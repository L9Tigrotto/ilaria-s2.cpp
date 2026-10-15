#include "../third_party/filesystem.hpp"
namespace fs = ghc::filesystem;
#include "s2_log.h"
#include "s2_pipeline.h"
#include "s2_server.h"
#include "s2_audio.h"
#include "s2_codec.h"
#include "s2_voice.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <io.h>
#include <fcntl.h>
#else
#include <clocale>
#endif

static void safe_print(const char* msg) {
    fputs(msg, stdout);
}

static void safe_print(const std::string& msg) {
    fputs(msg.c_str(), stdout);
}

static void safe_print_error(const char* msg) {
    fputs(msg, stderr);
}

static void safe_print_error(const std::string& msg) {
    fputs(msg.c_str(), stderr);
}

static bool parse_log_level(const std::string& value, s2::LogLevel& out) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "error" || lower == "errors" || lower == "0") {
        out = s2::LogLevel::Error;
        return true;
    }
    if (lower == "warn" || lower == "warning" || lower == "warnings" || lower == "1") {
        out = s2::LogLevel::Warning;
        return true;
    }
    if (lower == "info" || lower == "2") {
        out = s2::LogLevel::Info;
        return true;
    }
    if (lower == "debug" || lower == "3") {
        out = s2::LogLevel::Debug;
        return true;
    }
    return false;
}

void print_uso() {
    safe_print("Usage: s2 [options]\n");
    safe_print("Options:\n");
    safe_print("  -m, --model        <path>   Path to GGUF model\n");
    safe_print("  -t, --tokenizer    <path>   Path to tokenizer.json\n");
    safe_print("  -text, --text      <text>   Text to synthesize\n");
    safe_print("  -pa, --prompt-audio <path>  Path to reference audio for cloning\n");
    safe_print("  -pt, --prompt-text <text>   Text of the reference audio\n");
    safe_print("  --voice            <id>     Load a saved voice profile\n");
    safe_print("  --save-voice                Save the encoded reference as a voice profile\n");
    safe_print("  --voice-dir        <path>   Directory used for saved voice profiles\n");
    safe_print("  --list-voices               List available saved voice profiles and exit\n");
    safe_print("  -o, --output       <path>   Output WAV path\n");
    safe_print("  -v, --vulkan     <id>   Vulkan device index (-1 = CPU)\n");
    safe_print("  -c, --cuda       <id>   CUDA device index (-1 = CPU)\n");
    safe_print("  -M, --metal             Use Metal backend (macOS only)\n");
    safe_print("  -ngl, --gpu-layers  <n>     Transformer layers on GPU (-1 = auto, 0 = CPU only)\n");
    safe_print("  -threads, --threads <n>     CPU threads (0 = auto)\n");
    safe_print("  -max-tokens, --max-tokens <n>  Max tokens to generate\n");
    safe_print("  --min-tokens-before-end <n> Minimum tokens before EOS is allowed\n");
    safe_print("  -temp, --temp, --temperature <f>  Temperature\n");
    safe_print("  -top-p, --top-p    <f>      Top-p sampling\n");
    safe_print("  -top-k, --top-k    <n>      Top-k sampling\n");
    safe_print("  --seed             <n>      Sampler seed (-1 = random)\n");
    safe_print("  --dynamic-normalize         Apply dynamic RMS normalization\n");
    safe_print("  --no-dynamic-normalize      Disable dynamic RMS normalization\n");
    safe_print("  --no-trim-silence           Keep trailing silence in output WAV\n");
    safe_print("  --trim-silence              Trim trailing silence in output WAV\n");
    safe_print("  --no-normalize              Keep original output peak level\n");
    safe_print("  --normalize                 Peak-normalize output WAV to 0.95\n");
    safe_print("  --codec-auto                Benchmark codec backends and keep the fastest (default)\n");
    safe_print("  --codec-follow-backend      Force codec to follow the selected GPU backend\n");
    safe_print("  --codec-cpu                 Force codec on CPU even when model uses GPU\n");
    safe_print("  --stream-file               Write output WAV through the streaming path\n");
    safe_print("  --batch-text       <text>   Repeatable: generate the texts side by side, <output>-<n>.wav each\n");
    safe_print("  --slots            <n>      Requests generated side by side (default 2; 1 = one at a time)\n");
    safe_print("  --stream-decode-stride <n>  Decode cadence in frames (0 = auto: server 4, file/offline 16)\n");
    safe_print("  --codec-context-frames <n>  Override codec decode history of offline decodes (streams keep state)\n");
    safe_print("  --codec-stream-check <file> Decode a .s2voice's codes whole and streamed, print the difference\n");
    safe_print("  --log-level <level>         error, warn, info, or debug (default: info)\n");
    safe_print("  --server                    Start HTTP server\n");
    safe_print("  -H, --host         <host>   Server host (default: 127.0.0.1)\n");
    safe_print("  -P, --port         <port>   Server port (default: 3030)\n");
    safe_print("  --cors             <origin> Allow browser requests from this origin, \"*\" for any (default: none)\n");
    safe_print("  -h, --help                  Show this help\n");
}

// Decodes a voice's codes once whole, then streamed in chunks of a few sizes, and prints how far apart they are.
// Writes <output>-whole.wav and <output>-stream<n>.wav for the ears.
// Decodes codes dumped by S2_DUMP_DIR (see s2_pipeline.cpp) in one piece, with the codec of -m on the backend asked.
static int run_decode_codes(const s2::PipelineParams & params, const std::string & codes_path) {
    std::ifstream in(codes_path, std::ios::binary);
    int32_t head[2] = { 0, 0 };
    in.read(reinterpret_cast<char *>(head), sizeof(head));
    std::vector<int32_t> codes(static_cast<size_t>(head[0]) * head[1]);
    in.read(reinterpret_cast<char *>(codes.data()), sizeof(int32_t) * codes.size());
    if (!in || codes.empty()) {
        safe_print_error("Error: cannot read codes from " + codes_path + "\n");
        return 1;
    }
    s2::AudioCodec codec;
    if (!codec.load(params.model_path, params.gpu_device, params.backend_type)) return 1;
    std::vector<float> pcm;
    if (params.codec_decode_context_frames < 0) {
        if (!codec.decode(codes.data(), head[1], params.gen.n_threads, pcm)) return 1;
    } else {
        // The streamed decode before the codec kept its state (holdback 0): every stride a window of the new frames
        // and up to context frames before them, emitted from the first new frame to the window's end.
        const int32_t T = head[1], stride = (std::max)(1, params.stream_decode_stride_frames);
        const int32_t ctx = params.codec_decode_context_frames;
        const size_t spf = static_cast<size_t>(codec.samples_per_code_frame());
        std::vector<int32_t> window;
        std::vector<float> part;
        for (int32_t done = 0; done < T;) {
            const int32_t end = (std::min)(T, done + stride), start = (std::max)(0, done - ctx), n = end - start;
            window.clear();
            for (int32_t cb = 0; cb < head[0]; ++cb) {
                const auto row = codes.begin() + static_cast<size_t>(cb) * T;
                window.insert(window.end(), row + start, row + end);
            }
            if (!codec.decode(window.data(), n, params.gen.n_threads, part)) return 1;
            pcm.insert(pcm.end(), part.begin() + static_cast<size_t>(done - start) * spf,
                       part.begin() + (std::min)(part.size(), static_cast<size_t>(n) * spf));
            done = end;
        }
    }
    s2::audio_write_wav(params.output_path, pcm.data(), pcm.size(), codec.sample_rate());
    safe_print(std::to_string(head[1]) + " frames decoded to " + params.output_path + "\n");
    return 0;
}

static int run_codec_stream_check(const s2::PipelineParams & params, const std::string & voice_path) {
    const s2::VoiceProfile voice = s2::VoiceProfile::load(voice_path);
    // The first 160 frames: more than the codec transformer's window of 128, and a whole decode of all of them fits.
    const int32_t T = (std::min)(voice.T_prompt, 160);
    if (T <= 0 || voice.codes.size() < static_cast<size_t>(voice.num_codebooks) * T) {
        safe_print_error("Error: no codes in " + voice_path + "\n");
        return 1;
    }
    s2::AudioCodec codec;
    if (!codec.load(params.model_path, params.gpu_device, params.backend_type)) return 1;
    const std::string stem = params.output_path.size() > 4 ? params.output_path.substr(0, params.output_path.size() - 4)
                                                           : params.output_path;
    std::vector<int32_t> codes;
    for (int32_t cb = 0; cb < voice.num_codebooks; ++cb) {
        const auto row = voice.codes.begin() + static_cast<size_t>(cb) * voice.T_prompt;
        codes.insert(codes.end(), row, row + T);
    }
    std::vector<float> whole;
    if (!codec.decode(codes.data(), T, params.gen.n_threads, whole)) return 1;
    s2::audio_write_wav(stem + "-whole.wav", whole.data(), whole.size(), codec.sample_rate());
    double energy = 0.0;
    for (float x : whole) energy += static_cast<double>(x) * x;
    safe_print(std::to_string(T) + " frames, " + std::to_string(whole.size()) + " samples decoded whole\n");
    {
        // For comparison: what a window of the old streamed decode costs (stride 32 and context 16, 48 frames).
        std::vector<int32_t> window;
        for (int32_t cb = 0; cb < voice.num_codebooks; ++cb) {
            window.insert(window.end(), codes.begin() + static_cast<size_t>(cb) * T,
                          codes.begin() + static_cast<size_t>(cb) * T + 48);
        }
        std::vector<float> pcm;
        codec.decode(window.data(), 48, params.gen.n_threads, pcm);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 5; ++i) codec.decode(window.data(), 48, params.gen.n_threads, pcm);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5;
        safe_print("windowed decode of 48 frames: " + std::to_string(ms) + " ms\n");
    }

    int bad = 0;
    for (int32_t chunk : { 1, 4, 8, 13, 32 }) {
        s2::AudioCodec::Stream * stream = codec.stream_begin();
        std::vector<float> streamed, pcm;
        std::vector<int32_t> part;
        double ms = 0.0, first_ms = -1.0; // the first call also builds the graph
        for (int32_t at = 0; at < T; at += chunk) {
            const int32_t n = (std::min)(chunk, T - at);
            part.resize(static_cast<size_t>(voice.num_codebooks) * n);
            for (int32_t cb = 0; cb < voice.num_codebooks; ++cb) {
                std::copy_n(voice.codes.begin() + static_cast<size_t>(cb) * voice.T_prompt + at, n,
                            part.begin() + static_cast<size_t>(cb) * n);
            }
            const auto t0 = std::chrono::steady_clock::now();
            if (!codec.stream_decode(stream, part.data(), n, params.gen.n_threads, pcm)) {
                safe_print_error("Error: stream decode failed\n");
                codec.stream_end(stream);
                return 1;
            }
            const double call_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (first_ms < 0.0) first_ms = call_ms; else ms += call_ms;
            streamed.insert(streamed.end(), pcm.begin(), pcm.end());
        }
        codec.stream_end(stream);
        s2::audio_write_wav(stem + "-stream" + std::to_string(chunk) + ".wav", streamed.data(), streamed.size(),
                            codec.sample_rate());
        double diff = 0.0, max_diff = 0.0;
        const size_t n = (std::min)(whole.size(), streamed.size());
        for (size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(whole[i]) - streamed[i];
            diff += d * d;
            max_diff = (std::max)(max_diff, std::fabs(d));
        }
        const double snr = diff > 0.0 ? 10.0 * std::log10(energy / diff) : 999.0;
        char line[200];
        std::snprintf(line, sizeof(line), "chunk %2d: %zu samples, max diff %.2e, SNR %.1f dB, first call %.0f ms, then %.2f ms a frame\n",
                      chunk, streamed.size(), max_diff, snr, first_ms, ms / (std::max)(1, T - chunk));
        safe_print(line);
        if (streamed.size() != whole.size() || snr < 60.0) bad = 1;
    }
    return bad;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    fflush(stdout);
    fflush(stderr);
    if (!_isatty(_fileno(stdout))) _setmode(_fileno(stdout), _O_BINARY);
    if (!_isatty(_fileno(stderr))) _setmode(_fileno(stderr), _O_BINARY);

    int argc_w;
    LPWSTR* argv_w = CommandLineToArgvW(GetCommandLineW(), &argc_w);
    static std::vector<std::string> args_utf8;
    static std::vector<char*> new_argv;
    if (argv_w != NULL) {
        for (int i = 0; i < argc_w; i++) {
            int size = WideCharToMultiByte(CP_UTF8, 0, argv_w[i], -1, NULL, 0, NULL, NULL);
            std::vector<char> arg(size);
            WideCharToMultiByte(CP_UTF8, 0, argv_w[i], -1, arg.data(), size, NULL, NULL);
            args_utf8.emplace_back(arg.data());
        }
        LocalFree(argv_w);
        for (auto& s : args_utf8) new_argv.push_back(s.data());
        argv = new_argv.data();
        argc = argc_w;
    }
#endif

#ifndef _WIN32
    if (!std::setlocale(LC_ALL, "")) {
        std::setlocale(LC_ALL, "C.UTF-8");
    }
#endif

    if (argc < 2) {
        print_uso();
        return 1;
    }

    fs::u8arguments u8guard(argc, argv);

    s2::PipelineParams params;
    params.model_path = "model.gguf";
    params.tokenizer_path = "tokenizer.json";
    params.output_path = "out.wav";
    params.text = u8"Hello world";
    params.gpu_device = -1;
    params.backend_type = s2::BackendType::CPU;

    bool use_server = false;
    bool use_stream_file = false;
    bool list_voices = false;
    std::string codec_stream_check;
    std::string decode_codes;
    s2::ServerParams serverParams;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-m"  || arg == "--model")        { if (i+1 < argc) params.model_path        = argv[++i]; }
        else if (arg == "-t"  || arg == "--tokenizer")    { if (i+1 < argc) params.tokenizer_path    = argv[++i]; }
        else if (arg == "-text" || arg == "--text")       { if (i+1 < argc) params.text               = argv[++i]; }
        else if (arg == "-pa" || arg == "--prompt-audio") { if (i+1 < argc) params.prompt_audio_path = argv[++i]; }
        else if (arg == "-pt" || arg == "--prompt-text")  { if (i+1 < argc) params.prompt_text        = argv[++i]; }
        else if (arg == "--voice")                        { if (i+1 < argc) params.voice_id           = argv[++i]; }
        else if (arg == "--save-voice")                   { params.save_voice = true; }
        else if (arg == "--voice-dir")                    { if (i+1 < argc) params.voice_storage_dir  = argv[++i]; }
        else if (arg == "--list-voices")                  { list_voices = true; }
        else if (arg == "-o"  || arg == "--output")       { if (i+1 < argc) params.output_path        = argv[++i]; }
        else if (arg == "-v"  || arg == "--vulkan")       { if (i+1 < argc) { try { params.gpu_device = std::stoi(argv[++i]); } catch(...) {} params.backend_type = s2::BackendType::Vulkan; } }
        else if (arg == "-c"  || arg == "--cuda")         { if (i+1 < argc) { try { params.gpu_device = std::stoi(argv[++i]); } catch(...) {} params.backend_type = s2::BackendType::CUDA; } }
        else if (arg == "-M"  || arg == "--metal")        { params.gpu_device = 0; params.backend_type = s2::BackendType::Metal; }
        else if (arg == "-ngl" || arg == "--gpu-layers")  { if (i+1 < argc) { try { params.n_gpu_layers = std::stoi(argv[++i]); } catch(...) {} } }
        else if (arg == "-threads" || arg == "--threads") { if (i+1 < argc) { try { params.gen.n_threads      = std::stoi(argv[++i]); } catch(...) {} } }
        else if (arg == "-max-tokens" || arg == "--max-tokens") {
            if (i+1 < argc) {
                try { params.gen.max_new_tokens = std::stoi(argv[++i]); } catch(...) {}
            }
        }
        else if (arg == "--min-tokens-before-end")        {
            if (i+1 < argc) {
                try {
                    params.gen.min_tokens_before_end = std::stoi(argv[++i]);
                    if (params.gen.min_tokens_before_end < 0) params.gen.min_tokens_before_end = 0;
                } catch(...) {}
            }
        }
        else if (arg == "-temp" || arg == "--temp" || arg == "--temperature") { if (i+1 < argc) { try { params.gen.temperature = std::stof(argv[++i]); } catch(...) {} } }
        else if (arg == "-top-p" || arg == "--top-p")      { if (i+1 < argc) { try { params.gen.top_p       = std::stof(argv[++i]); } catch(...) {} } }
        else if (arg == "-top-k" || arg == "--top-k")      { if (i+1 < argc) { try { params.gen.top_k       = std::stoi(argv[++i]); } catch(...) {} } }
        else if (arg == "--seed")                          { if (i+1 < argc) { try { params.gen.seed        = std::stoi(argv[++i]); } catch(...) {} } }
        else if (arg == "--dynamic-normalize")     { params.normalize_dynamic = true;  }
        else if (arg == "--no-dynamic-normalize")  { params.normalize_dynamic = false; }
        else if (arg == "--no-trim-silence")  { params.trim_silence     = false; }
        else if (arg == "--trim-silence")     { params.trim_silence     = true;  }
        else if (arg == "--no-normalize")     { params.normalize_output = false; }
        else if (arg == "--normalize")        { params.normalize_output = true;  }
        else if (arg == "--codec-auto")       { params.codec_auto_backend = true;  params.codec_follow_backend = true;  }
        else if (arg == "--codec-follow-backend") { params.codec_auto_backend = false; params.codec_follow_backend = true; }
        else if (arg == "--codec-cpu")        { params.codec_auto_backend = false; params.codec_follow_backend = false; }
        else if (arg == "--stream-file")      { use_stream_file = true; }
        else if (arg == "--batch-text")       { if (i+1 < argc) params.batch_texts.push_back(argv[++i]); }
        else if (arg == "--slots")            { if (i+1 < argc) { try { params.n_slots = std::stoi(argv[++i]); } catch(...) {} } }
        else if (arg == "--stream-decode-stride") {
            if (i+1 < argc) {
                try { params.stream_decode_stride_frames = std::stoi(argv[++i]); } catch(...) {}
            }
        }
        else if (arg == "--codec-context-frames") {
            if (i+1 < argc) {
                try { params.codec_decode_context_frames = (std::max)(0, std::stoi(argv[++i])); } catch(...) {}
            }
        }
        else if (arg == "--log-level") {
            if (i+1 < argc) {
                s2::LogLevel log_level;
                if (parse_log_level(argv[++i], log_level)) {
                    s2::set_log_level(log_level);
                } else {
                    safe_print_error("Warning: unknown --log-level value; expected error, warn, info, or debug.\n");
                }
            }
        }
        else if (arg == "--server")           { use_server = true; }
        else if (arg == "--codec-stream-check") { if (i+1 < argc) codec_stream_check = argv[++i]; }
        else if (arg == "--decode-codes") { if (i+1 < argc) decode_codes = argv[++i]; }
        else if (arg == "-H" || arg == "--host") { if (i+1 < argc) serverParams.host = argv[++i]; }
        else if (arg == "-P" || arg == "--port") { if (i+1 < argc) { try { serverParams.port = std::stoi(argv[++i]); } catch(...) {} } }
        else if (arg == "--cors") { if (i+1 < argc) serverParams.cors_origin = argv[++i]; }
        else if (arg == "-h" || arg == "--help") { print_uso(); return 0; }
    }

    if (!decode_codes.empty()) {
        return run_decode_codes(params, decode_codes);
    }

    if (!codec_stream_check.empty()) {
        return run_codec_stream_check(params, codec_stream_check);
    }

    if (list_voices) {
        std::vector<std::string> voice_ids;
        const fs::path dir(params.voice_storage_dir);
        if (fs::exists(dir)) {
            for (const auto & entry : fs::directory_iterator(dir)) {
                if (entry.path().extension() == ".s2voice") {
                    voice_ids.push_back(entry.path().stem().string());
                }
            }
        }

        std::sort(voice_ids.begin(), voice_ids.end());
        if (voice_ids.empty()) {
            safe_print("No saved voice profiles found.\n");
        } else {
            safe_print("Saved voice profiles:\n");
            for (const std::string & voice_id : voice_ids) {
                safe_print("  " + voice_id + "\n");
            }
        }
        return 0;
    }

    const bool has_prompt_audio = !params.prompt_audio_path.empty();
    const bool has_prompt_text = !params.prompt_text.empty();
    const bool has_voice_id = !params.voice_id.empty();

    if (params.save_voice) {
        if (!has_voice_id) {
            safe_print_error("Error: --save-voice requires --voice <id>.\n");
            return 1;
        }
        if (!has_prompt_audio || !has_prompt_text) {
            safe_print_error("Error: --save-voice requires both --prompt-audio and --prompt-text.\n");
            return 1;
        }
    }

    if (has_voice_id && has_prompt_audio && !params.save_voice) {
        safe_print_error("Warning: prompt audio takes precedence over --voice for this request.\n");
    }

    if (has_voice_id && !has_prompt_audio && has_prompt_text) {
        safe_print_error("Warning: --prompt-text is ignored when loading a saved --voice profile.\n");
    }

    if (params.n_gpu_layers > 0 &&
        params.gpu_device < 0 &&
        params.backend_type != s2::BackendType::Metal) {
        safe_print_error("Error: --gpu-layers requires --cuda, --vulkan, or --metal to select a GPU backend.\n");
        return 1;
    }

    if (params.tokenizer_path == "tokenizer.json") {
        std::string model_path = params.model_path;
        size_t slash = model_path.find_last_of("/\\");
        if (slash != std::string::npos) {
            std::string model_dir = model_path.substr(0, slash + 1);
            std::string candidate = model_dir + "tokenizer.json";
            if (FILE * f = std::fopen(candidate.c_str(), "r")) {
                std::fclose(f);
                params.tokenizer_path = candidate;
            } else {
                size_t parent_slash = model_dir.find_last_of("/\\", slash - 1);
                if (parent_slash != std::string::npos) {
                    candidate = model_dir.substr(0, parent_slash + 1) + "tokenizer.json";
                    if (FILE * f2 = std::fopen(candidate.c_str(), "r")) {
                        std::fclose(f2);
                        params.tokenizer_path = candidate;
                    }
                }
            }
        }
    }

    if (params.gen.max_new_tokens > 1024) {
        if (s2::log_enabled(s2::LogLevel::Warning)) {
            safe_print_error("Warning: --max-tokens > 1024 may cause voice quality degradation. Consider splitting long texts.\n");
        }
    }

    if (use_server) {
        serverParams.pipeline = params;
        s2::Server server;
        if (!server.serve(serverParams)) {
            safe_print_error("Server initialization failed.\n");
            return 1;
        }
        return 0;
    }

    s2::Pipeline pipeline;
    if (!pipeline.init(params)) {
        safe_print_error("Pipeline initialization failed.\n");
        return 1;
    }

    if (!params.batch_texts.empty()) {
        if (!pipeline.synthesize_batch(params)) {
            safe_print_error("Synthesis failed.\n");
            return 1;
        }
        return 0;
    }

    if (use_stream_file) {
        if (!pipeline.synthesize_streaming_file(params)) {
            safe_print_error("Synthesis failed.\n");
            return 1;
        }
        return 0;
    }

    if (!pipeline.synthesize(params)) {
        safe_print_error("Synthesis failed.\n");
        return 1;
    }

    return 0;
}
