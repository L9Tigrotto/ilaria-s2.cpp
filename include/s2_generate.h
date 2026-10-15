#pragma once

#include "s2_model.h"
#include "s2_sampler.h"
#include "s2_tokenizer.h"
#include "s2_prompt.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
#include <functional>

namespace s2 {

struct FrameCallbackData {
    const int32_t* codes;
    int32_t frame_index;
    int32_t total_frames;
    int32_t num_codebooks;
};

using FrameCallback = std::function<bool(const FrameCallbackData&)>;

struct GenerateParams {
    int32_t max_new_tokens          = 1024;
    float   temperature             = 0.8f;
    float   top_p                   = 0.8f;
    int32_t top_k                   = 30;
    int32_t min_tokens_before_end   = 0;
    int32_t n_threads               = 0;
    int32_t seed                    = -1;   // the sampler's seed; -1 picks one at random
    bool    verbose                 = true;
    FrameCallback on_frame;
};

struct GenerateResult {
    std::vector<int32_t> codes;
    int32_t num_codebooks = 0;
    int32_t n_frames      = 0;
};

// The sequences generated side by side, one KV slot each. add prefills a sequence into its slot; step makes the
// next frame of every live sequence in one batched pass (decoding is bound by reading the weights, so two cost
// little more than one). Sequences join and leave between steps.
class Generator {
public:
    Generator(SlowARModel & model, const TokenizerConfig & config);
    bool add(int32_t slot, const PromptTensor & prompt, const GenerateParams & params);
    void step();
    bool any_live() const;
    bool done(int32_t slot) const; // ended and not taken yet
    GenerateResult take(int32_t slot);

private:
    struct Seq {
        GenerateParams params;
        GenerateResult out;
        StepResult state;
        std::mt19937 gen;
        int32_t main_token = 0;
        std::vector<int32_t> ras_window;
        std::vector<int32_t> codebooks_cb;
        bool live = false, done = false;
        int32_t prefilled = 0, reused = 0;
        std::chrono::steady_clock::time_point t0, loop_t0;
        double prefill_ms = 0.0;
    };
    int32_t sample_main(Seq & s, const SamplerParams & sp, bool block_im_end);
    void finish(Seq & s);

    SlowARModel & model_;
    int32_t im_end_id_;
    std::vector<Seq> seqs_; // by slot
};

GenerateResult generate(
    SlowARModel & model,
    const TokenizerConfig & config,
    const PromptTensor & prompt,
    const GenerateParams & params
);

// Owns the model on a thread of its own and generates every request given to it side by side, up to n_slots at
// once; a request that comes while every slot is busy waits for one. Nothing else may use the model meanwhile.
class Engine {
public:
    Engine(SlowARModel & model, const TokenizerConfig & config, int32_t n_slots);
    ~Engine();
    // Blocks until the prompt is generated; params.on_frame runs on the engine's thread.
    GenerateResult run(const PromptTensor & prompt, const GenerateParams & params);

private:
    struct Job {
        const PromptTensor * prompt;
        const GenerateParams * params;
        GenerateResult result;
        bool done = false;
    };
    void loop();

    SlowARModel & model_;
    Generator gen_;
    const int32_t n_slots_;
    std::mutex mtx_;
    std::condition_variable cv_;      // wakes the engine
    std::condition_variable done_cv_; // wakes the callers of run
    std::deque<Job *> queue_;
    bool stop_ = false;
    std::thread thread_;
};

}
