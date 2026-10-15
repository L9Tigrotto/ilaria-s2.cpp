#include "../include/s2_generate.h"
#include "../include/s2_log.h"
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace s2 {

static SamplerParams sampler_params(const GenerateParams & p) {
    SamplerParams sp;
    sp.temperature = p.temperature;
    sp.top_p       = p.top_p;
    sp.top_k       = p.top_k;
    return sp;
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

Generator::Generator(SlowARModel & model, const TokenizerConfig & config)
    : model_(model), im_end_id_(config.im_end_id) {}

bool Generator::add(int32_t slot, const PromptTensor & prompt, const GenerateParams & params) {
    if (slot >= static_cast<int32_t>(seqs_.size())) seqs_.resize(slot + 1);
    Seq & s = seqs_[slot];
    s = Seq{};
    s.params = params;
    s.t0 = std::chrono::steady_clock::now();

    // The model scores only what may be sampled (see StepResult::logits).
    model_.set_im_end_id(im_end_id_);

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
    model_.begin_slot(slot);
    const int32_t reused = model_.reuse_prefix(prompt_tm, prompt.prefix_cols);
    if (reused > 0) prompt_tm.erase(prompt_tm.begin(), prompt_tm.begin() + static_cast<size_t>(reused) * rows);
    if (!model_.prefill_fast(prompt_tm, prompt.cols - reused, params.n_threads, s.state)) {
        std::cerr << "[Generate] Prefill failed." << std::endl;
        model_.drop_prefix(slot);
        return false;
    }
    s.prefill_ms = ms_since(s.t0);
    s.prefilled = prompt.cols;
    s.reused = reused;

    const int32_t num_cb = std::max(1, model_.hparams().num_codebooks);
    s.out.num_codebooks = num_cb;
    s.out.codes.resize(static_cast<size_t>(num_cb) * params.max_new_tokens, 0);
    s.gen.seed(params.seed >= 0 ? static_cast<uint32_t>(params.seed) : std::random_device{}());
    s.main_token = sample_main(s, sampler_params(params), params.min_tokens_before_end > 0);
    s.live = true;
    if (params.verbose && log_enabled(LogLevel::Info)) {
        std::cout << "[Generate] Generating (max " << params.max_new_tokens << " tokens)..." << std::endl;
    }
    s.loop_t0 = std::chrono::steady_clock::now();
    return true;
}

// Entry i of state.logits is semantic token sem_begin + i; an entry past them is the end token.
int32_t Generator::sample_main(Seq & s, const SamplerParams & sp, bool block_im_end) {
    const int32_t sem_begin = model_.hparams().semantic_begin_id;
    const int32_t sem_end   = model_.hparams().semantic_end_id;
    const int32_t n_sem     = sem_end - sem_begin + 1;
    std::vector<float> biased = s.state.logits;
    const int32_t end_idx = (im_end_id_ >= sem_begin && im_end_id_ <= sem_end) ? im_end_id_ - sem_begin
                          : (int32_t)biased.size() > n_sem ? n_sem : -1;
    if (block_im_end && end_idx >= 0) {
        biased[end_idx] = -std::numeric_limits<float>::infinity();
    }
    const int32_t idx = sample_token(biased.data(), (int32_t)biased.size(), sp, s.gen);
    return idx < n_sem ? sem_begin + idx : im_end_id_;
}

bool Generator::any_live() const {
    for (const Seq & s : seqs_) if (s.live) return true;
    return false;
}

bool Generator::done(int32_t slot) const {
    return slot < static_cast<int32_t>(seqs_.size()) && seqs_[slot].done;
}

GenerateResult Generator::take(int32_t slot) {
    seqs_[slot].done = false;
    return std::move(seqs_[slot].out);
}

void Generator::finish(Seq & s) {
    s.live = false;
    s.done = true;
    GenerateResult & out = s.out;
    const int32_t num_cb = out.num_codebooks;
    const int32_t max = s.params.max_new_tokens;
    std::vector<int32_t> compacted(static_cast<size_t>(num_cb) * out.n_frames);
    for (int32_t cb = 0; cb < num_cb; ++cb) {
        for (int32_t t = 0; t < out.n_frames; ++t) {
            compacted[static_cast<size_t>(cb) * out.n_frames + t] = out.codes[static_cast<size_t>(cb) * max + t];
        }
    }
    out.codes = std::move(compacted);

    if (s.params.verbose && log_enabled(LogLevel::Info)) {
        const double loop_ms = ms_since(s.loop_t0);
        std::cout << std::endl;
        std::cout << "[Generate] Done: " << out.n_frames << " frames generated." << std::endl;
        std::cout << "[Metrics] Generate: prefill=" << s.prefill_ms
                  << " ms (" << (s.prefilled - s.reused) << " of " << s.prefilled << " tokens), loop=" << loop_ms
                  << " ms, total=" << ms_since(s.t0)
                  << " ms, avg=" << (out.n_frames > 0 ? loop_ms / out.n_frames : 0.0)
                  << " ms/frame" << std::endl;
    }
}

void Generator::step() {
    const int32_t sem_begin     = model_.hparams().semantic_begin_id;
    const int32_t sem_end       = model_.hparams().semantic_end_id;
    const int32_t codebook_size = model_.hparams().codebook_size;
    const int32_t ras_window_size = 10;
    const float ras_high_temp     = 1.0f;
    const float ras_high_top_p    = 0.9f;

    std::vector<int32_t> live;
    for (int32_t i = 0; i < static_cast<int32_t>(seqs_.size()); ++i) {
        Seq & s = seqs_[i];
        if (!s.live) continue;
        if (s.main_token == im_end_id_ || s.out.n_frames >= s.params.max_new_tokens) finish(s);
        else live.push_back(i);
    }
    if (live.empty()) return;
    const int32_t num_cb   = seqs_[live[0]].out.num_codebooks;
    const int32_t n_threads = seqs_[live[0]].params.n_threads;

    std::vector<const std::vector<float> *> hiddens;
    std::vector<const std::vector<int32_t> *> prefixes;
    for (int32_t i : live) {
        Seq & s = seqs_[i];
        const int32_t frame = s.out.n_frames;
        if (!s.ras_window.empty() &&
            std::find(s.ras_window.begin(), s.ras_window.end(), s.main_token) != s.ras_window.end() &&
            s.main_token >= sem_begin && s.main_token <= sem_end)
        {
            SamplerParams ras_sparams;
            ras_sparams.temperature = ras_high_temp;
            ras_sparams.top_p       = ras_high_top_p;
            ras_sparams.top_k       = s.params.top_k;
            s.main_token = sample_main(s, ras_sparams, frame < s.params.min_tokens_before_end);
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

    std::vector<float> fast_logits;
    for (int32_t cb_idx = 1; cb_idx < num_cb; ++cb_idx) {
        if (!model_.fast_decode_batch(hiddens, prefixes, n_threads, fast_logits)) {
            std::cerr << "[Generate] fast_decode failed at cb " << cb_idx << std::endl;
            for (int32_t i : live) seqs_[i].codebooks_cb.resize(num_cb, 0);
            break;
        }
        const size_t n_logits = fast_logits.size() / live.size();
        for (size_t j = 0; j < live.size(); ++j) {
            Seq & s = seqs_[live[j]];
            s.codebooks_cb.push_back(sample_token(fast_logits.data() + j * n_logits, (int32_t)n_logits,
                                                  sampler_params(s.params), s.gen));
        }
    }

    std::vector<int32_t> slots, step_input;
    for (int32_t i : live) {
        Seq & s = seqs_[i];
        GenerateResult & out = s.out;
        const int32_t frame = out.n_frames;
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            out.codes[static_cast<size_t>(cb) * s.params.max_new_tokens + frame] = s.codebooks_cb[cb];
        }
        out.n_frames++;

        if (s.params.on_frame) {
            FrameCallbackData fcd;
            fcd.codes = s.codebooks_cb.data();
            fcd.frame_index = frame;
            fcd.total_frames = out.n_frames;
            fcd.num_codebooks = num_cb;
            if (!s.params.on_frame(fcd)) {
                if (s.params.verbose && log_enabled(LogLevel::Info)) {
                    std::cout << "\n[Generate] Aborted by callback at frame " << frame << std::endl;
                }
                finish(s);
                continue;
            }
        }

        slots.push_back(i);
        step_input.push_back(s.main_token);
        step_input.insert(step_input.end(), s.codebooks_cb.begin(), s.codebooks_cb.end());
    }
    if (slots.empty()) return;

    std::vector<StepResult> results;
    if (!model_.step_batch(slots, step_input, n_threads, results)) {
        std::cerr << "[Generate] step() failed at step " << seqs_[slots[0]].out.n_frames - 1 << std::endl;
        for (int32_t i : slots) finish(seqs_[i]);
        return;
    }
    for (size_t j = 0; j < slots.size(); ++j) {
        Seq & s = seqs_[slots[j]];
        s.state = std::move(results[j]);
        const int32_t frames = s.out.n_frames;
        if (s.params.verbose && log_enabled(LogLevel::Info) && frames % 50 == 0) {
            std::cout << "\r[Generate] " << frames << " / " << s.params.max_new_tokens << " tokens..." << std::flush;
        }
        s.main_token = sample_main(s, sampler_params(s.params), frames < s.params.min_tokens_before_end);
    }
}

GenerateResult generate(
    SlowARModel & model,
    const TokenizerConfig & config,
    const PromptTensor & prompt,
    const GenerateParams & params
) {
    Generator g(model, config);
    if (!g.add(0, prompt, params)) {
        GenerateResult out;
        out.num_codebooks = std::max(1, model.hparams().num_codebooks);
        return out;
    }
    while (g.any_live()) g.step();
    return g.take(0);
}

Engine::Engine(SlowARModel & model, const TokenizerConfig & config, int32_t n_slots)
    : model_(model), gen_(model, config), n_slots_(std::max(1, n_slots)), thread_([this] { loop(); }) {}

Engine::~Engine() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable()) thread_.join();
}

GenerateResult Engine::run(const PromptTensor & prompt, const GenerateParams & params) {
    Job job{ &prompt, &params };
    std::unique_lock<std::mutex> lock(mtx_);
    queue_.push_back(&job);
    cv_.notify_one();
    done_cv_.wait(lock, [&] { return job.done; });
    if (job.result.num_codebooks == 0) job.result.num_codebooks = std::max(1, model_.hparams().num_codebooks);
    return std::move(job.result);
}

void Engine::loop() {
    std::vector<Job *> in_slot(n_slots_, nullptr);
    std::unique_lock<std::mutex> lock(mtx_);
    for (;;) {
        cv_.wait(lock, [&] { return stop_ || !queue_.empty() || gen_.any_live(); });
        if (stop_ && !gen_.any_live()) {
            for (Job * j : queue_) j->done = true;
            queue_.clear();
            done_cv_.notify_all();
            return;
        }

        // A waiting request takes a free slot between two steps; the others pause for its prefill.
        while (!queue_.empty()) {
            const auto free_slot = std::find(in_slot.begin(), in_slot.end(), nullptr);
            if (free_slot == in_slot.end()) break;
            Job * j = queue_.front();
            // At least room for a voice prompt and 1024 frames, so that a sentence a little longer than the
            // ones before does not wait for the others to end before the cache can grow.
            const int32_t needed = std::max(1536, j->prompt->cols + j->params->max_new_tokens);
            bool ok = true;
            if (model_.kv_capacity(n_slots_) < needed) {
                if (gen_.any_live()) break; // the cache grows once the others end
                lock.unlock();
                ok = model_.init_kv_cache(needed, n_slots_);
                lock.lock();
            }
            queue_.pop_front();
            const int32_t slot = static_cast<int32_t>(free_slot - in_slot.begin());
            if (ok) {
                lock.unlock();
                ok = gen_.add(slot, *j->prompt, *j->params);
                lock.lock();
            }
            if (ok) {
                *free_slot = j;
            } else {
                j->done = true;
                done_cv_.notify_all();
            }
        }

        lock.unlock();
        if (gen_.any_live()) gen_.step();
        lock.lock();
        for (int32_t s = 0; s < n_slots_; ++s) {
            if (in_slot[s] && gen_.done(s)) {
                in_slot[s]->result = gen_.take(s);
                in_slot[s]->done = true;
                in_slot[s] = nullptr;
                done_cv_.notify_all();
            }
        }
    }
}

}
