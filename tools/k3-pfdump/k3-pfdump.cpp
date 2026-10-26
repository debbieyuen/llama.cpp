// k3-pfdump: decode a chat prompt greedily and dump, per decode step and layer, the speculative prefetch
// predictor's scores, the real router's selection scores and the real selected expert ids, for an offline
// evaluation of prediction precision (kimi-k3-spark expert store).
//
// Output file records: int32 step, int32 layer, int32 kind, int32 n, then n elements
//   kind 1: moe_prefetch_probs (f32)   kind 2: ffn_moe_probs_biased (f32)   kind 3: ffn_moe_topk (i32)
//   kind 4: ffn_moe_probs (f32)        kind 5: generated token id (i32, n = 1, layer = -1)
// Usage: llama-k3-pfdump <common args> -p "<user prompt>" -n <decode tokens> --out <file>

#include "arg.h"
#include "common.h"
#include "chat.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct dump_state {
    FILE * f = nullptr;
    int step = -1; // -1 during the prefill: nothing is recorded
    std::vector<uint8_t> buf;
};

static int kind_of(const char * name, int & il) {
    static const struct { const char * prefix; int kind; } kinds[] = {
        { "moe_prefetch_probs-",   1 },
        { "ffn_moe_probs_biased-", 2 },
        { "ffn_moe_topk-",         3 },
        { "ffn_moe_probs-",        4 },
        { "pfv2_probs-",           6 },
        { "pfv3_probs-",           7 },
        { "pfv4_probs-",           8 },
        { "pfv5_probs-",           9 },
    };
    for (const auto & k : kinds) {
        const size_t n = strlen(k.prefix);
        if (strncmp(name, k.prefix, n) == 0) {
            il = atoi(name + n);
            return k.kind;
        }
    }
    return 0;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (dump_state *) ud;
    int il = -1;
    const int kind = kind_of(t->name, il);
    if (ask) {
        return st->step >= 0 && kind != 0;
    }
    if (st->step < 0 || kind == 0) {
        return true;
    }
    const int64_t n = ggml_nelements(t);
    if (!(t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_I32) || !ggml_is_contiguous(t)) {
        LOG_ERR("%s: unexpected tensor %s type %s\n", __func__, t->name, ggml_type_name(t->type));
        return true;
    }
    st->buf.resize((size_t) n * 4);
    ggml_backend_tensor_get(t, st->buf.data(), 0, st->buf.size());
    const int32_t hdr[4] = { st->step, il, kind, (int32_t) n };
    fwrite(hdr, sizeof(hdr), 1, st->f);
    fwrite(st->buf.data(), 4, (size_t) n, st->f);
    return true;
}

int main(int argc, char ** argv) {
    std::string out_path = "pfdump.bin";
    std::vector<char *> args;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
            continue;
        }
        args.push_back(argv[i]);
    }
    common_params params;
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    dump_state st;
    st.f = fopen(out_path.c_str(), "wb");
    if (st.f == nullptr) {
        LOG_ERR("cannot open %s\n", out_path.c_str());
        return 1;
    }
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // the chat template, like the server with --jinja
    auto tmpls = common_chat_templates_init(model, "");
    common_chat_templates_inputs in;
    common_chat_msg msg;
    msg.role = "user";
    msg.content = params.prompt;
    in.messages.push_back(msg);
    in.add_generation_prompt = true;
    in.use_jinja = true;
    std::string text;
    for (const auto & part : common_chat_templates_apply(tmpls.get(), in).prompt.parts) {
        text += part.text;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, text, true, true);
    LOG_INF("prompt: %zu tokens, decoding %d\n", tokens.size(), params.n_predict);

    st.step = -1;
    {
        common_batch batch = common_batch_get_one(ctx, tokens);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("prefill failed\n");
            return 1;
        }
    }
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::string out_text;
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < params.n_predict; i++) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) {
            LOG_INF("eog after %d tokens\n", i);
            break;
        }
        out_text += common_token_to_piece(ctx, tok);
        st.step = i;
        const int32_t hdr[4] = { i, -1, 5, 1 };
        fwrite(hdr, sizeof(hdr), 1, st.f);
        fwrite(&tok, 4, 1, st.f);
        std::vector<llama_token> one = { tok };
        common_batch batch = common_batch_get_one(ctx, one);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("decode failed at step %d\n", i);
            return 1;
        }
    }
    const double dt = (ggml_time_us() - t0) / 1e6;
    LOG_INF("decode: %d steps in %.1f s (%.2f tok/s, with the dump callback)\n", st.step + 1, dt, (st.step + 1) / dt);
    fclose(st.f);
    FILE * ft = fopen((out_path + ".txt").c_str(), "wb");
    if (ft != nullptr) {
        fwrite(out_text.data(), 1, out_text.size(), ft);
        fclose(ft);
    }
    llama_sampler_free(smpl);
    llama_backend_free();
    return 0;
}
