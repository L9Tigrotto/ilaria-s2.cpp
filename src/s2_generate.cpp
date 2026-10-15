#include "../include/s2_generate.h"
#include "../include/s2_log.h"
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace s2 {

GenerateResult generate(
    SlowARModel & model,
    const TokenizerConfig & config,
    const PromptTensor & prompt,
    const GenerateParams & params
) {
    return std::move(generate_batch(model, config, { &prompt }, params)[0]);
}

std::vector<GenerateResult> generate_batch(
    SlowARModel & model,
    const TokenizerConfig & config,
    const std::vector<const PromptTensor *> & prompts,
    const GenerateParams & params
) {
    const auto generate_t0 = std::chrono::steady_clock::now();
    const int32_t n_seq = static_cast<int32_t>(prompts.size());
    std::vector<GenerateResult> outs(n_seq);
    const int32_t num_cb = std::max(1, model.hparams().num_codebooks);
    for (auto & o : outs) o.num_codebooks = num_cb;

    const int32_t sem_begin    = model.hparams().semantic_begin_id;
    const int32_t sem_end      = model.hparams().semantic_end_id;
    const int32_t n_sem        = sem_end - sem_begin + 1;
    const int32_t codebook_size = model.hparams().codebook_size;
    const int32_t im_end_id    = config.im_end_id;

    // The model scores only what may be sampled (see StepResult::logits).
    model.set_im_end_id(im_end_id);

    SamplerParams sparams;
    sparams.temperature     = params.temperature;
    sparams.top_p           = params.top_p;
    sparams.top_k           = params.top_k;

    // One sequence per prompt, in the model's slot of the same index.
    struct Seq {
        StepResult state;
        std::mt19937 gen;
        int32_t main_token = 0;
        std::vector<int32_t> ras_window;
        std::vector<int32_t> codebooks_cb;
        bool live = true;
        int32_t prefilled = 0, reused = 0;
    };
    std::vector<Seq> seqs(n_seq);

    const auto prefill_t0 = std::chrono::steady_clock::now();
    for (int32_t i = 0; i < n_seq; ++i) {
        const PromptTensor & prompt = *prompts[i];
        const int32_t rows = prompt.rows;
        const int32_t cols = prompt.cols;
        std::vector<int32_t> prompt_tm(static_cast<size_t>(rows) * cols);
        for (int32_t r = 0; r < rows; ++r) {
            for (int32_t c = 0; c < cols; ++c) {
                prompt_tm[static_cast<size_t>(c) * rows + r] = prompt.data[static_cast<size_t>(r) * cols + c];
            }
        }
        if (params.verbose && log_enabled(LogLevel::Info)) {
            std::cout << "[Generate] Prefilling " << prompt.cols << " tokens..." << std::endl;
        }
        model.use_slot(i);
        const int32_t reused = model.reuse_prefix(prompt_tm, prompt.prefix_cols);
        if (reused > 0) prompt_tm.erase(prompt_tm.begin(), prompt_tm.begin() + static_cast<size_t>(reused) * rows);
        if (!model.prefill_fast(prompt_tm, prompt.cols - reused, params.n_threads, seqs[i].state)) {
            std::cerr << "[Generate] Prefill failed." << std::endl;
            model.clear_kv_cache(); // the prefix it was writing is not whole
            return outs;
        }
        seqs[i].prefilled = prompt.cols;
        seqs[i].reused = reused;
        // The same seed for every sequence: two equal prompts make the same speech, alone or side by side.
        seqs[i].gen.seed(params.seed >= 0 ? static_cast<uint32_t>(params.seed) : std::random_device{}());
    }
    model.use_slot(0);
    const auto prefill_t1 = std::chrono::steady_clock::now();

    // Entry i of state.logits is semantic token sem_begin + i; an entry past them is the end token.
    auto sample_main = [&](Seq & s, const SamplerParams & sp, bool block_im_end) -> int32_t {
        std::vector<float> biased = s.state.logits;
        const int32_t end_idx = (im_end_id >= sem_begin && im_end_id <= sem_end) ? im_end_id - sem_begin
                              : (int32_t)biased.size() > n_sem ? n_sem : -1;
        if (block_im_end && end_idx >= 0) {
            biased[end_idx] = -std::numeric_limits<float>::infinity();
        }
        const int32_t idx = sample_token(biased.data(), (int32_t)biased.size(), sp, s.gen);
        return idx < n_sem ? sem_begin + idx : im_end_id;
    };

    for (auto & s : seqs) s.main_token = sample_main(s, sparams, params.min_tokens_before_end > 0);
    for (auto & o : outs) o.codes.resize(static_cast<size_t>(num_cb) * params.max_new_tokens, 0);

    const int32_t ras_window_size = 10;
    const float ras_high_temp     = 1.0f;
    const float ras_high_top_p    = 0.9f;

    if (params.verbose && log_enabled(LogLevel::Info)) {
        std::cout << "[Generate] Generating (max " << params.max_new_tokens << " tokens)..." << std::endl;
    }

    // All sequences make their frames in lockstep: frame `step` of each in one Slow-AR step and one Fast-AR pass
    // per codebook. A sequence that ends (or whose callback says stop) leaves the batch.
    std::vector<int32_t> live, slots, step_input;
    std::vector<const std::vector<float> *> hiddens;
    std::vector<const std::vector<int32_t> *> prefixes;
    std::vector<float> fast_logits;
    std::vector<StepResult> results;
    int32_t step = 0;
    int64_t seq_frames = 0;
    const auto loop_t0 = std::chrono::steady_clock::now();
    for (;;) {
        live.clear();
        for (int32_t i = 0; i < n_seq; ++i) {
            Seq & s = seqs[i];
            if (s.live && s.main_token != im_end_id && step < params.max_new_tokens) live.push_back(i);
            else s.live = false;
        }
        if (live.empty()) break;

        hiddens.clear();
        prefixes.clear();
        for (int32_t i : live) {
            Seq & s = seqs[i];
            if (!s.ras_window.empty() &&
                std::find(s.ras_window.begin(), s.ras_window.end(), s.main_token) != s.ras_window.end() &&
                s.main_token >= sem_begin && s.main_token <= sem_end)
            {
                SamplerParams ras_sparams;
                ras_sparams.temperature = ras_high_temp;
                ras_sparams.top_p       = ras_high_top_p;
                ras_sparams.top_k       = params.top_k;
                s.main_token = sample_main(s, ras_sparams, step < params.min_tokens_before_end);
            }

            s.ras_window.push_back(s.main_token);
            if ((int32_t)s.ras_window.size() > ras_window_size) {
                s.ras_window.erase(s.ras_window.begin());
            }

            int32_t sem_code = s.main_token - sem_begin;
            if (sem_code < 0)           sem_code = 0;
            if (sem_code >= codebook_size) sem_code = codebook_size - 1;

            s.codebooks_cb.clear();
            s.codebooks_cb.reserve(num_cb);
            s.codebooks_cb.push_back(sem_code);
            hiddens.push_back(&s.state.hidden);
            prefixes.push_back(&s.codebooks_cb);
        }

        for (int32_t cb_idx = 1; cb_idx < num_cb; ++cb_idx) {
            if (!model.fast_decode_batch(hiddens, prefixes, params.n_threads, fast_logits)) {
                std::cerr << "[Generate] fast_decode failed at cb " << cb_idx << std::endl;
                for (int32_t i : live) seqs[i].codebooks_cb.resize(num_cb, 0);
                break;
            }
            const size_t n_logits = fast_logits.size() / live.size();
            for (size_t j = 0; j < live.size(); ++j) {
                Seq & s = seqs[live[j]];
                s.codebooks_cb.push_back(sample_token(fast_logits.data() + j * n_logits, (int32_t)n_logits, sparams, s.gen));
            }
        }

        slots.clear();
        step_input.clear();
        for (int32_t i : live) {
            Seq & s = seqs[i];
            GenerateResult & out = outs[i];
            for (int32_t cb = 0; cb < num_cb; ++cb) {
                out.codes[static_cast<size_t>(cb) * params.max_new_tokens + step] = s.codebooks_cb[cb];
            }
            out.n_frames++;
            seq_frames++;

            if (params.on_frame) {
                FrameCallbackData fcd;
                fcd.codes = s.codebooks_cb.data();
                fcd.frame_index = step;
                fcd.total_frames = out.n_frames;
                fcd.num_codebooks = num_cb;
                fcd.sequence = i;
                if (!params.on_frame(fcd)) {
                    if (params.verbose && log_enabled(LogLevel::Info)) {
                        std::cout << "\n[Generate] Aborted by callback at frame " << step << std::endl;
                    }
                    s.live = false;
                    continue;
                }
            }

            slots.push_back(i);
            step_input.push_back(s.main_token);
            step_input.insert(step_input.end(), s.codebooks_cb.begin(), s.codebooks_cb.end());
        }
        if (slots.empty()) break;

        if (!model.step_batch(slots, step_input, params.n_threads, results)) {
            std::cerr << "[Generate] step() failed at step " << step << std::endl;
            break;
        }
        for (size_t j = 0; j < slots.size(); ++j) seqs[slots[j]].state = std::move(results[j]);

        step++;
        if (params.verbose && log_enabled(LogLevel::Info) && step % 50 == 0) {
            std::cout << "\r[Generate] " << step << " / " << params.max_new_tokens << " tokens..." << std::flush;
        }

        for (int32_t i : slots) seqs[i].main_token = sample_main(seqs[i], sparams, step < params.min_tokens_before_end);
    }

    if (params.verbose && log_enabled(LogLevel::Info)) {
        const auto loop_t1 = std::chrono::steady_clock::now();
        const auto generate_t1 = std::chrono::steady_clock::now();
        const double prefill_ms = std::chrono::duration<double, std::milli>(prefill_t1 - prefill_t0).count();
        const double loop_ms = std::chrono::duration<double, std::milli>(loop_t1 - loop_t0).count();
        const double total_ms = std::chrono::duration<double, std::milli>(generate_t1 - generate_t0).count();
        int32_t prefilled = 0, reused = 0;
        for (const auto & s : seqs) { prefilled += s.prefilled; reused += s.reused; }
        std::cout << std::endl;
        std::cout << "[Generate] Done: " << seq_frames << " frames generated." << std::endl;
        std::cout << "[Metrics] Generate: prefill=" << prefill_ms
                  << " ms (" << (prefilled - reused) << " of " << prefilled << " tokens), loop=" << loop_ms
                  << " ms, total=" << total_ms
                  << " ms, avg=" << (seq_frames > 0 ? loop_ms / seq_frames : 0.0)
                  << " ms/frame";
        if (n_seq > 1) {
            std::cout << ", sequences=" << n_seq << ", steps=" << step
                      << ", step_avg=" << (step > 0 ? loop_ms / step : 0.0) << " ms";
        }
        std::cout << std::endl;
    }

    for (GenerateResult & out : outs) {
        const int32_t n_frames = out.n_frames;
        std::vector<int32_t> compacted(static_cast<size_t>(num_cb) * n_frames);
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            for (int32_t t = 0; t < n_frames; ++t) {
                compacted[static_cast<size_t>(cb) * n_frames + t] =
                    out.codes[static_cast<size_t>(cb) * params.max_new_tokens + t];
            }
        }
        out.codes = std::move(compacted);
    }

    return outs;
}

}
