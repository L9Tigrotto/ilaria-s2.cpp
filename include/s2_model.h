#pragma once

#include "s2_backend.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace s2 {

struct ModelLayer {
    ggml_tensor * attention_norm = nullptr;
    ggml_tensor * ffn_norm       = nullptr;
    ggml_tensor * q_norm         = nullptr;
    ggml_tensor * k_norm         = nullptr;
    ggml_tensor * wqkv           = nullptr;
    ggml_tensor * wo             = nullptr;
    ggml_tensor * w1             = nullptr;
    ggml_tensor * w2             = nullptr;
    ggml_tensor * w3             = nullptr;
};

struct ModelHParams {
    int32_t context_length     = 0;
    int32_t vocab_size         = 0;
    int32_t embedding_length   = 0;
    int32_t feed_forward_length = 0;
    int32_t block_count        = 0;
    int32_t head_count         = 0;
    int32_t head_count_kv      = 0;
    int32_t codebook_size      = 0;
    int32_t num_codebooks      = 0;
    int32_t semantic_begin_id  = 0;
    int32_t semantic_end_id    = 0;
    float   rope_freq_base     = 10000.0f;
    float   rms_norm_eps       = 1e-5f;
    bool    tie_word_embeddings = true;
    bool    attention_qk_norm  = false;
    bool    scale_codebook_embeddings = false;

    int32_t fast_context_length     = 0;
    int32_t fast_embedding_length   = 0;
    int32_t fast_feed_forward_length = 0;
    int32_t fast_block_count        = 0;
    int32_t fast_head_count         = 0;
    int32_t fast_head_count_kv      = 0;
    int32_t fast_head_dim           = 0;
    float   fast_rope_freq_base     = 10000.0f;
    float   fast_rms_norm_eps       = 1e-5f;
    bool    fast_attention_qk_norm  = false;
    bool    fast_has_project_in     = false;
    bool    has_fast_decoder        = false;
};

struct ModelWeights {
    ggml_context * ctx_w = nullptr;
    std::vector<ggml_backend_buffer_t> model_bufs_cpu;
    std::vector<ggml_backend_buffer_t> model_bufs_gpu;

    ggml_tensor * embeddings           = nullptr;
    ggml_tensor * codebook_embeddings  = nullptr;
    ggml_tensor * norm                 = nullptr;
    ggml_tensor * fast_project_in      = nullptr;
    ggml_tensor * fast_embeddings      = nullptr;
    ggml_tensor * fast_norm            = nullptr;
    ggml_tensor * fast_output          = nullptr;

    std::vector<ModelLayer> layers;
    std::vector<ModelLayer> fast_layers;
};

struct StepResult {
    std::vector<float> hidden;
    // Scores over the whole vocab, unless the model was given the end token (set_im_end_id): then the
    // semantic tokens only (semantic_begin_id..semantic_end_id), followed, when the end token lies outside
    // that range, by the end token's score as the last entry.
    std::vector<float> logits;
};

class SlowARModel {
public:
    SlowARModel();
    ~SlowARModel();

    bool load(const std::string & gguf_path, int32_t gpu_device = -1, BackendType backend_type = BackendType::CPU, int32_t n_gpu_layers = -1);

    bool load_shared(gguf_context * gguf_ctx, const std::string & gguf_path, int32_t gpu_device = -1, BackendType backend_type = BackendType::CPU, int32_t n_gpu_layers = -1);

    bool read_tensor_data(const std::string & gguf_path, gguf_context * gguf_ctx);

    ggml_context * weights_ctx() { return weights_.ctx_w; }
    const std::unordered_set<ggml_tensor *> & weight_tensor_set() const { return weight_tensor_set_; }

    // A slot of max_seq_len tokens for each of n_seq sequences. Keeps the cache (and the prefixes in it) while it
    // holds that; else allocates a new one. Every slot starts empty, and slot 0 is the one in use.
    bool init_kv_cache(int32_t max_seq_len, int32_t n_seq = 1);

    // The slot that reuse_prefix, prefill and step work on.
    void use_slot(int32_t slot) { slot_ = slot; }
    // Empties a slot for a new sequence, keeping its prefix for reuse_prefix; the other slots go on.
    void begin_slot(int32_t slot) { slot_ = slot; if (slot < n_seq_) n_past_[slot] = 0; }
    // After a failed prefill: the prefix it was writing is not whole.
    void drop_prefix(int32_t slot) { if (slot < n_seq_) kept_prefix_[slot].clear(); }
    // Tokens a slot holds when the cache has at least n_seq slots; else 0.
    int32_t kv_capacity(int32_t n_seq) const { return kv_buf_ && n_seq <= n_seq_ ? max_seq_len_ : 0; }

    // A request whose first n_prefix tokens equal those of the slot's request before (a voice's reference prompt)
    // starts after them: their keys and values are still in the cache. Returns how many tokens need no prefill.
    int32_t reuse_prefix(const std::vector<int32_t> & flat_tokens, int32_t n_prefix);

    void reset();

    void clear_kv_cache();

private:
    bool eval_cached(const std::vector<int32_t> & flat_tokens,
        int32_t n_tokens, int32_t n_threads,
        StepResult & result);
public:
    bool prefill_fast(const std::vector<int32_t> & flat_tokens, int32_t n_tokens,
        int32_t n_threads, StepResult & result);

    bool prefill(const std::vector<int32_t> & flat_tokens, int32_t n_tokens,
                 int32_t n_threads, StepResult & result);

    bool step(const std::vector<int32_t> & flat_tokens, int32_t n_threads,
              StepResult & result);

    bool fast_decode(const std::vector<float> & hidden,
                     const std::vector<int32_t> & prefix_codes,
                     int32_t n_threads,
                     std::vector<float> & logits_out);

    // step and fast_decode for several sequences at once, which costs little more than one: decoding is bound by
    // reading the weights. step_batch: one token per sequence (num_codebooks + 1 ints each, in the order of
    // slots), one result each. fast_decode_batch: every sequence at the same codebook position, codebook_size
    // scores each in logits_out.
    bool step_batch(const std::vector<int32_t> & slots, const std::vector<int32_t> & flat_tokens,
                    int32_t n_threads, std::vector<StepResult> & results);
    bool fast_decode_batch(const std::vector<const std::vector<float> *> & hiddens,
                           const std::vector<const std::vector<int32_t> *> & prefix_codes,
                           int32_t n_threads, std::vector<float> & logits_out);

    const ModelHParams & hparams() const { return hparams_; }

    // Narrows StepResult::logits to what can be sampled (see there); a negative id restores the whole vocab.
    void set_im_end_id(int32_t id) {
        if (id != im_end_id_) drop_graph(slow_graph_, sched_); // its output head depends on it
        im_end_id_ = id;
    }

    ModelWeights   weights_;

private:
    ModelHParams   hparams_;
    ggml_backend_t backend_cpu_  = nullptr;
    ggml_backend_t backend_gpu_  = nullptr;
    ggml_backend_sched_t sched_       = nullptr;
    ggml_backend_sched_t fast_sched_  = nullptr;
    ggml_context * ctx_kv_      = nullptr;
    ggml_backend_buffer_t kv_buf_ = nullptr;
    ggml_tensor *  memory_k_   = nullptr;
    ggml_tensor *  memory_v_   = nullptr;
    int32_t        max_seq_len_ = 0; // per slot; slot s holds positions s * max_seq_len_ on
    int32_t        n_seq_      = 0;
    int32_t        slot_       = 0;
    int32_t        n_gpu_layers_ = -1;
    std::vector<int32_t> n_past_;                   // per slot
    std::vector<std::vector<int32_t>> kept_prefix_; // per slot: the tokens whose keys and values head it (see reuse_prefix)
    bool step_run(int32_t slot0, int32_t n_seq, const int32_t * flat_tokens, int32_t n_threads, StepResult * results);
    static void token_inputs(const ModelHParams & hp, const int32_t * flat_tokens, int32_t n_tokens,
                             std::vector<int32_t> & semantic, std::vector<float> & semantic_mask,
                             std::vector<float> & token_scale, std::vector<std::vector<int32_t>> & cb);
    size_t         ctx_size_   = 0;
    std::vector<uint8_t> ctx_buf_;
    size_t         fast_ctx_size_ = 0;
    std::vector<uint8_t> fast_ctx_buf_;
    std::vector<uint8_t> fast_ctx_buf0_;

    // The Fast-AR's keys and values within one frame, so each codebook step computes only its own
    // token; fast_cached_ is how many positions it holds (-1: unusable, take the full path).
    // Batched: hiddens[i] and prefix_codes[i] are sequence i's, all at the same position.
    bool fast_decode_one(const std::vector<const std::vector<float> *> & hiddens,
                         const std::vector<const std::vector<int32_t> *> & prefix_codes,
                         int32_t n_threads, std::vector<float> & logits_out);
    ggml_context * fast_ctx_kv_ = nullptr;
    ggml_backend_buffer_t fast_kv_buf_ = nullptr;
    ggml_tensor *  fast_k_ = nullptr;   // num_codebooks positions for each of fast_n_seq_ sequences
    ggml_tensor *  fast_v_ = nullptr;
    int32_t        fast_n_seq_ = 0;
    int32_t        fast_cached_ = -1;

    int32_t        im_end_id_ = -1;

    // A decode graph kept with its scheduler allocation, so the next call only uploads input values
    // (and the CUDA backend replays it as a CUDA graph). key: what its shape depends on.
    struct CachedGraph {
        ggml_context * ctx = nullptr;
        ggml_cgraph  * gf  = nullptr;
        int32_t        key = -1;
        std::vector<ggml_tensor *> in;
        ggml_tensor *  hidden = nullptr;
        ggml_tensor *  logits = nullptr;
    };
    CachedGraph slow_graph_;
    CachedGraph fast_graph_[2];                // position 0, positions after it
    ggml_backend_sched_t fast_sched0_ = nullptr; // position 0's own, so both allocations stay
    // slot0 < 0: a prefill of n_tokens into slot_. Else a step of n_tokens sequences, one token each, in the
    // slots from slot0 on.
    bool build_slow_graph(int32_t n_tokens, int32_t n_kv, int32_t slot0);
    static void drop_graph(CachedGraph & g, ggml_backend_sched_t sched);
    static bool run_graph(CachedGraph & g, ggml_backend_sched_t sched,
                          const std::vector<std::pair<const void *, size_t>> & in_vals,
                          std::vector<float> & logits_out);

    std::unordered_set<ggml_tensor *> weight_tensor_set_;

};

}
