# Correctness plan: the two engines against llama.cpp at 3cf03257

_Results (step 12, run 2026-10-18 on the A6000 VM): `bench/unbiased/correctness/README.md` in git history at commit
7551f138; the run directories were removed from the tree on 2026-10-19, and the harness moved from
`bench/unbiased/correctness/scripts/` to `bench/correctness/`. The `bench/unbiased/...` paths below are as they were at the time._

PLAN.md step 11 (`PLAN.md:318`), written 2026-10-18. This is a plan: nothing in it has been run, no code was
changed, and every line number was read from this tree, the submodule `third_party/llama.cpp` (HEAD
`3cf03257`, "CUDA: enable sparse fa for qwen4 (#28770)") or the upstream checkout
`~/Code/unbiasedstrata2/PolyStrata-v0.1.45-poly.1`. Step 12 (`PLAN.md:319`) runs what is signed off here and
writes the results under `bench/unbiased/correctness/`.

Where a claim could not be settled by reading, it says so. Three such items matter for the run order and are
listed in section 2.1 and section 5.3.

## 0. Scope

**Target.** Linux x86-64, one NVIDIA GPU of Pascal or newer, CUDA 12.x. The numbers and run lines below are
the RTX 3090 pod's (`bench/unbiased/2026-10-17-3090pod-*-step6/server-command.sh`, `engine-info.txt`): 24 GB,
driver 580.126.20, CUDA 12.8, a CPU without AVX-512.

**Configuration A: `strata`, Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS.** Run line
`bench/unbiased/2026-10-17-3090pod-flash-next-iq3xxs-step6/server-command.sh:9-12`: `--pack` (the expert pack
from `tools/iq_pack.py`), `--native` shard 1, `--ple-gguf` shard 2, `--expert-profile`, `--expert-cache auto`,
`--prefill auto`, `--spec 4 --spec-min-p 0.5`, `--mtp` (the draft layer from `tools/mtp_pack.py` and
`tools/mtp_rt.py`), `--max-context 65536`, `--kv int8`, `--ple-io ram`, `--pool-workers 15`,
`--kv-resident 32768`. On the pod 10,531 of 24,576 routed experts are in VRAM (`engine-info.txt`: "expert
cache 10531 slots", 48 layers x 512 experts), so the other experts are computed on the CPU's AVX-2 kernels or
streamed over PCIe (`pcie_frac=0.55`), and `STRATA_ROUTE_TAIL_SKIP=7` and the prefill CPU share are on by
default (both printed in `engine-info.txt`).

**Configuration B: `strata-poly`, Qwen3.6-35B-A3B UD-Q3_K_M, and UD-IQ4_XS.** Run line
`bench/unbiased/2026-10-17-3090pod-qwen36-ud-q3km-mtp-step6/server-command.sh:7-8`: `--model` (the Unsloth
MTP file, draft block inside), `--max-context 32768`, `--expert-profile`, `--tail-skip 6`. On the pod every
routed expert is in VRAM (`engine-info.txt`: `expert_slots=10496` = 41 layers x 256, including the draft
block's), so `--tail-skip 6` has nothing to skip and no CPU expert runs in decode.

**Reference.** llama.cpp at the pinned commit, built from `third_party/llama.cpp` with CUDA on, run twice per
sequence: all layers on the GPU and all on the CPU backend. The two reference runs against each other are the
noise floor of "two correct implementations", which is how upstream frames every comparison it reports
(`docs/QWEN36.md:33-35`, `tools/qwen35/kl.py:1-8`).

**What "the same" means here.** Three tiers, used by every check below:

| Tier | Statement | Where it is owed |
| --- | --- | --- |
| T0, bitwise | Two runs of the same engine write the same bytes | Wherever the engine claims it: draft on against draft off, a verify window against one token, streamed KV against resident KV, prompt chunks of different sizes with the CPU share off |
| T1, argmax at a margin | At every position where llama.cpp's two likeliest tokens differ by 0.5 nats or more, the engine's argmax is llama.cpp's | Every engine-versus-llama.cpp comparison |
| T2, distribution | Mean KL(llama.cpp GPU \|\| engine) over the answer's positions is at most 1.5x mean KL(llama.cpp GPU \|\| llama.cpp CPU) on the same positions | Every engine-versus-llama.cpp comparison; lossy options (int8 KV, Q8_0 KV) against the same engine's fp16 run |

Why these and not "the same token ids": section 3.6.

**Out of scope.** Pictures (`GENI`), batch slots (`--batch`), HIP, other quantizations, sampled decoding
(the sampler is checked by `sampler_parity` alone, section 2.2), the front end's HTTP layer
(`frontend/tests/`).

## 1. Kernel and code paths in use

### 1.1 What the engines take from the pinned llama.cpp

The engines call no `llama_*` function. They link `ggml-base` and `ggml-cpu` from the submodule
(`CMakeLists.txt:1418`, `:1479`; `GGML_CUDA` forced off, `:1430`) and compile these ggml-cuda sources into
their own libraries:

| Library | Sources from the submodule | `CMakeLists.txt` | Who calls it |
| --- | --- | --- | --- |
| `strata_mmq` (static) | `ggml-cuda/quantize.cu`; `template-instances/mmq-instance-{q2_0,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_nl,iq4_xs,q8_0}.cu` | `:1568`, `:1585-1587`, `:1603-1605` | `strata`'s prompt path for routed experts (`src/prefill/moe_mmq.cu:404-460`) |
| `strata_kq` (shared, `libstrata_kq.so`, opened at run time by `STRATA_MMQ_KQ=1`) | `mmq-instance-{q4_k,q5_k,q5_1,q6_k}.cu` | `:1595`, `:1685-1687` | `strata` prompt path for K-quant packs; not configuration A |
| `strata_poly_mmq` (static) | `quantize.cu`; 18 instances `q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k iq1_s iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs` | `:2016-2018`, `:2045-2046` | `strata-poly`'s prompt path for every quantized matrix of 32 rows or more (`src/poly/mmq.cu:83-119`) |
| `strata_kernels`, `strata_kernels_cpu` | `third_party/ggml/ggml-common.h` (a copy; `third_party/ggml/VERSION.txt` says `3cf03257`, and `diff -q` against the submodule's copy is empty) | `:503`, `:1486` | Every "native" GPU kernel and the CPU expert kernels: the block layouts and codebook grids are llama.cpp's |

The decode-time dot products are transcriptions, not links: `src/kernels/cuda/iq_kernels.cu:3-6` says the
`vec_dot_*_q8_1` functions, the dequantizers and the q8_1 quantizer are transcribed from
`ggml-cuda/vecdotq.cuh`, `dequantize.cuh` and `quantize.cu` at the `VERSION.txt` commit;
`include/strata/kernels/iq_kernels.hpp:1-7` says activations are q8_1 (32 values, fp16 scale and sum), "the
llama.cpp CUDA contract". `src/kernels/cuda/native_mmvq.cu:3110-3130` dispatches Q3_K, Q4_K, Q5_K, Q6_K,
Q8_0 and IQ4_XS to per-type kernels and the i-quants to `iq_mmvq`. The CPU side links ggml-cpu for single-token
experts (`src/kernels/cpu/native_expert.cpp:1-3`: "the activation quantizers and the row dot products are
ggml-cpu's") and uses its own AVX-2 multi-token kernels above a group size (`:125-142`, default
`STRATA_IQ_MT_MIN=2`), whose arithmetic is ggml's "only the order of the float additions differs"
(`src/kernels/cpu/iq_avx2.cpp:9-10`).

The native kernels are compiled with `--use_fast_math` to "match the pinned llama.cpp CUDA arithmetic
contract" (`CMakeLists.txt:506-519`); ggml-cuda itself uses `-use_fast_math`
(`third_party/llama.cpp/ggml/src/ggml-cuda/CMakeLists.txt:186`).

The submodule's branch (robertkarl/llama.cpp `UnbiasedStrata`; at the time of these checks the same change was a
configure-time patch) carries tool-call parser changes only (`common/parsers/*`, `common/chat-auto-parser-generator.cpp`;
UPSTREAM.md "Changes to llama.cpp"). They touch nothing the engines or the reference arithmetic use.

Both model architectures load at the pin: `qwen35moe` (`src/llama-arch.h:47`, graph
`src/models/qwen35moe.cpp`) and `qwen4exp` (`src/llama-arch.h:48`, graph `src/models/qwen4exp.cpp`;
`conversion/qwen4exp.py:16-19` names "Qwen/Qwen3.8-Flash-Next"). Q2_0, the pack's down-expert format in 30
of 48 layers (`tests/data/native_experts/iq3_xxs.txt`), exists in the pin as `GGML_TYPE_Q2_0 = 42`
(`ggml/include/ggml.h:432`) with an MMVQ dot (`ggml-cuda/mmvq.cu:43`, `vecdotq.cuh:733`) and an MMQ case
(`ggml-cuda/mmq.cu:13-14`).

### 1.2 Configuration A: `strata` with the Flash-Next pack

A native pack never runs a per-token loop: every decode step is a verify window of T tokens through
`Verifier::run` (`src/program/generate.cpp:15624`; `:6562` "a native pack runs verify windows only"), and
`--spec 4` becomes a window of 6 because the suffix drafter adds 2 (`generate.cpp:2989-2992`; `--suffix-draft 0`
keeps 4). The emitted token after each window row is the head's argmax; a draft is kept while it equals that
argmax (`generate.cpp:15653`: `while (a < T - 1 && window[a + 1] == outv[a]) ++a;`), so the drafter decides
which tokens get checked, never which are emitted (`include/strata/core/verify.hpp:1-8`,
`include/strata/core/mtp.hpp:13-14`). The pack mixes formats: gate/up IQ2_XXS (9 layers), IQ2_XS (10),
IQ3_XXS (6), IQ3_S (13), IQ2_S (10); down IQ4_NL (18) or Q2_0 (30). That is the checked-in layout
`tests/data/native_experts/iq3_xxs.txt` (48 lines), and it matches the GGUF itself: the header of the local copy
of shard 1 (`/Volumes/models/Qwen3.8-Flash-Next-GSQ-RCO/IQ3_XXS/`, sha256 `219ea929…` in its `SHA256SUMS`, read
with the vendored `gguf-py`) gives the same gate/up and down type for every one of the 48 layers. The pod runs
the same bytes (`server-command.sh:3-4`) and the pack keeps every quantized tensor in GGUF form
(`tools/iq_pack.py`), so the pod pack's `native_experts.txt` is expected to match; one `diff` in step 12 records
it. The rest of the file, from the same header: `general.architecture` is `qwen4exp`, 1,223 tensors in shard 1
and one in shard 2 (`per_layer_token_embd.weight`, IQ4_NL, the PLE table); `token_embd` IQ3_S; routers,
hyper-connection and indexer projections BF16 with F32 norms; attention and SSM projections a per-layer mix of
IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K; shared-expert gate/up IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K and down IQ4_NL or Q2_0;
`output.weight` Q5_K; `ple_conv1d` F16; no tensor named `mtp`, `nextn` or `draft`.

| Stage | `strata` path | llama.cpp path at the pin | Relation |
| --- | --- | --- | --- |
| Token embedding | Gather from the pack's embedding in mapped host memory (`engine-info.txt`: "token embedding IQ3_S"), `src/core/verify.cpp:1096-1099`, `src/core/native_head.cpp:206-210` | `get_rows` over the GGUF `token_embd` (IQ3_S in the file) | Same codebooks (`ggml-common.h`); the file's and the engine's embedding type agree (IQ3_S in both the header and `engine-info.txt`) |
| Hyper-connections (4 residual streams, `include/strata/core/geometry.hpp:27`) | `fused_gr_read_multi`, `src/core/verify.cpp:1248` (attention and FFN halves; `fused_gr_read_persist` `:1247` under `STRATA_HC_PERSIST=1`), output mixer `:2123-2156`; the per-token `gr_read` in `src/core/layer.cpp` belongs to the non-native loop, which a native pack never runs | `src/models/qwen4exp.cpp` ("hyper-connections in place of every layer norm", `conversion/qwen4exp.py:19-24`); the lines were not located in this pass | Own kernels on both sides |
| Gated DeltaNet layers | Decode: `gdn_ab_multi` `src/core/verify.cpp:1311`, `:1372`, `gdn_conv_ab_multi` `:1359`, `gdn_step_norm_multi` `:1412`; `include/strata/kernels/verify_kernels.hpp:1-74` says they are bitwise the `fused_gdn_*` single-token kernels. Prompt: `gdn_recurrence`, `src/prefill/prefill.cpp:3036` to `src/prefill/kernels.cu:2554` | `ggml_gated_delta_net` (`ggml/include/ggml.h:2652`), CUDA `ggml-cuda/gated_delta_net.cu:4-30`, all f32; fused on by force, `src/llama-context.cpp:232-234`; used at `src/models/qwen4exp.cpp:970` | Same recurrence in f32; different kernels and reduction order |
| QSA attention, decode | Norm and rope `src/core/verify.cpp:1451`; block scores and top-k `:1545-1548`; `qsa_kv_resolve` `:1767` (`src/core/layer.cpp:954`); `qsa_decode_attn_batch` `:1737`, `:1777`; output gate `:1782` | `build_qsa_top_k` `src/models/qwen4exp.cpp:542` (`ggml_top_k` `:684`), `build_attn_qsa` `:693-764` unmasks the chosen cells and runs the dense `build_attn_mha`; flash attention kernel choice `ggml-cuda/fattn.cu:529-704` | Same selection rule in different code; attention over the same cells with different kernels |
| QSA attention, prompt | `qsa_prompt_attn_batch`, `src/prefill/prefill.cpp:3259-3270`, tensor cores, "not bitwise equal" to the old path (`include/strata/kernels/qsa_prompt_attn.hpp:1-18`); selection `:3150-3160` (3xTF32) | MMA_F16 flash attention for a batch on sm_86 (`fattn.cu:626-652`) | Both tensor-core fp16 with f32 accumulation |
| KV cache | `--kv int8`: int8 codes, one fp16 scale per 64 values (`generate.cpp:805-806`, `include/strata/kernels/kv_q8.hpp:1-12`), appended by `kv_append_q8_steps` `src/core/verify.cpp:1641-1643` (per token `:1662`); `--kv-resident 32768`: pages streamed from a pinned host copy, `kv_stream_resolve` inside `qsa_kv_resolve` (`src/core/layer.cpp:954-957`), `include/strata/kernels/kv_stream.hpp:1-20` | f16 by default (`common/arg.cpp:2434-2458`); q8_0 (one fp16 scale per 32 values, `ggml-common.h:255`) has a compiled vector kernel (`ggml/CMakeLists.txt:207`, `fattn.cu:470`) | Different int8 formats, so int8 KV has no llama.cpp twin. Streaming has none either; `kv_stream.hpp:8-9` says readers are unchanged, so streamed must equal resident bitwise |
| Router, top-10 of 512 | `bf16_gemv_fp32_mmvf_multi` + `native_router_top10_multi`, `src/core/verify.cpp:1830-1832` | Router `mul_mat` + top-k in the graph (`src/models/qwen4exp.cpp`; line not located) | BF16 router weights both sides; f32 logits |
| Routed experts, decode, VRAM tier | `native_expert_grouped`, `src/core/verify.cpp:1962-1971` to `src/kernels/cuda/iq_kernels.cu:4291-4480`; dots `vec_dot_iq3_xxs_q8_1` `iq_kernels.cu:191` and the other formats under `STRATA_GU_FMTS`/`STRATA_D_FMTS` `:647-655`; activations q8_1 by `quantize_q8_1_rows` `verify.cpp:1918-1920` | `ggml_cuda_mul_mat_id` `ggml-cuda/ggml-cuda.cu:1924-1930`: MMVQ with ids while n_tokens is at most the per-type cap (`mmvq.cu:197-207`: IQ3_XXS 7, IQ3_S 6, IQ2_S 7, Q2_0 8); dots `vecdotq.cuh:1172` (iq3_xxs), `:733` (q2_0); `quantize_q8_1` `quantize.cu:54-100` | Transcribed dots, same q8_1 activation contract; expert sums combined in strata's order |
| Routed experts, decode, PCIe tier | The same GPU kernels on staged copies once the copy engine lands them (`verify.cpp:1960`); share of misses `(nmiss * pcie_num) >> 8`, `src/core/expert_source.cpp:3875` | n/a | Placement only; same arithmetic as the VRAM tier |
| Routed experts, decode, CPU tier | `src/kernels/cpu/pool.cpp:732`, `:742` to `native_gu_rows` / `native_down_rows` `src/kernels/cpu/native_expert.cpp:144-248`, `:250-305`: ggml-cpu `vec_dot` for a group below `STRATA_IQ_MT_MIN` (default 2, `:125-142`), the AVX-2 multi-token kernels (`src/kernels/cpu/iq_avx2.cpp`) otherwise; activations Q8_K (`:113-123`) | CPU backend: `ggml_vec_dot_iq3_xxs_q8_K` `ggml/src/ggml-cpu/arch/x86/quants.c:3260`, `quantize_row_q8_K` `:505` (ref `ggml/src/ggml-quants.c:2768`, blocks of 256) | ggml's own code for a lone token; Q8_K (256-block) activations, so a CPU expert rounds like llama.cpp's CPU backend and unlike the GPU tier |
| Tail skip, adaptive tier | A missed expert that every token of a window ranks 7th or lower is dropped, `src/core/expert_source.cpp:3850-3869`, default 7 in serve `generate.cpp:9375-9420`; the tier re-places experts every 4 rounds (`generate.cpp:677`, off when 0 `:9423`); `pcie_frac` refined in the first windows when not given (`:671-672`) | n/a | Approximations; section 1.4 says how to hold them |
| Shared expert | `shared_expert_multi`, `src/core/verify.cpp:1917` (`src/kernels/cuda/shared_expert.cu`) | Dense `mul_mat` on the `shexp` tensors; MMVQ at one token | Gate/up IQ3_S, IQ4_XS, Q4_K, Q5_K or Q6_K and down IQ4_NL or Q2_0 by layer (shard-1 header); the K-quant ones take `native_mmvq`'s per-type kernels, so `mmvq_multi_parity` and `iq_parity` cover them |
| Dense projections, decode | BF16: `bf16_gemv_fp32_mmvf_multi` `src/core/verify.cpp:1163`; quantized: `native_quantize_q8_1` + `native_mmvq` `:1166-1167` (`native_mmvq.cu:3110-3130`) | `mmvf` for BF16 (`ggml-cuda.cu:1844-1848`), MMVQ for quantized (`:1868-1870`) | Transcribed |
| Dense projections, prompt | `src/prefill/gemm.cu:1042-1045` (BF16 `cublasGemmEx`), `:1109-1131` (dequantize to FP16, FP32 compute) | MMQ for quantized batches (`ggml-cuda.cu:1872-1874`, `mmq.cu:319`), cuBLAS for floats (`:1876`, `:1622-1670`) | Different: strata dequantizes quantized projections to FP16 for the prompt, llama.cpp runs int8 MMQ |
| Routed experts, prompt | `src/prefill/prefill.cpp:3366-3383` picks MMQ (fused int8 is off by default on CUDA, `src/prefill/moe_fused_iq.cu:1386-1407`); `src/prefill/moe_mmq.cu:404-460` calls `mul_mat_q_case<T>` (`:442`), activations `quantize_scatter_mmq_q8_1_cuda` (`:384-395`, `quantize.cu:606`); CPU share of the chunk `prefill.cpp:185-215` | `ggml_cuda_mul_mat_id` MMQ branch `ggml-cuda.cu:1940-1943`, same instances, scatter quantizer `mmq.cu:233` | Same ggml kernel and quantizer; tile and stream-k choice may differ with the row count (`mmq.cuh:173`, `:346-351`) |
| PLE (n-gram embeddings, one layer) | Rows on the host `src/core/verify.cpp:2682-2685`; key and value projections `:1152-1170`; post-ops and `ple_block` `:1194-1219`; needs `--ple-gguf` or "LAYER 1's PLE IS SILENTLY SKIPPED" (`generate.cpp:943`) | `src/models/qwen4exp.cpp` ("PLE n-gram hash embeddings on a single layer", `conversion/qwen4exp.py:19-24`); line not located | Own kernels both sides; a missing shard 2 is a silent difference |
| Head and argmax | Native Q5_K head (`generate.cpp:838`), `native_mmvq` over the window's rows `src/core/verify.cpp:2177`; greedy `argmax_rows` `:2187` (or `sample_tokens` with `greedy = true`, `:2188-2193`), ties to the lowest index (`src/kernels/cuda/sampler.cu:114-118`) | Output `mul_mat` (MMVQ at one token, `vec_dot_q5_K_q8_1` `vecdotq.cuh:968`); logits f32 (`src/llama-context.h:295`); greedy keeps the first argmax (`src/llama-sampler.cpp:270-285`) | Same tie rule |
| MTP draft layer | `src/core/mtp.cpp`: shared expert `:1099`, router `:1112-1116`, argmax `:1216`; weights Q8_0 dense and q2_0 experts from `tools/mtp_rt.py` and `tools/mtp_pack.py` (not the GGUF's: the converter drops the head, `conversion/qwen4exp.py:28-30`) | None for `qwen4exp` (`src/models/models.h:2382-2420` has no MTP graph) | Drafts only; the exact-match acceptance makes it invisible in the ids |

### 1.3 Configuration B: `strata-poly` with Qwen3.6-35B-A3B

Dispatch is one function, `Engine::lin()` (`src/poly/engine.cpp:2072-2104`): a quantized matrix with 32 rows
or more (`kStageRows`, `src/poly/runtime.hpp:29`) goes to ggml's MMQ through `Mmq::quantize` and `mmq_->mul`
when `Mmq::fits` (`src/poly/mmq.cu:40-59`); fewer rows go to `native_quantize_q8_1` + `native_mmvq`, at most 8
columns per launch, each column bitwise the single-column result (`g_multi_exact = true`,
`src/kernels/cuda/native_mmvq.cu:1342`). F32 and BF16 matrices use `gemv_kernel` below 32 rows and
`gemm_small_kernel` or `cublasSgemm` above (`src/poly/kernels.cu:2179-2209`). Tensors stay in GGUF form when
`native_mmvq` supports the type (`engine.cpp:1101-1102`, list `native_mmvq.cu:3074-3079`); F16 is converted
to BF16 (`:1103`, `:1126-1128`). The UD-Q3_K_M file is not on this Mac, so its per-tensor types were not read;
the local UD-IQ4_XS (no draft block, `docs/MODEL-FILES.md:15`) has attention, shared-expert and embedding
tensors Q8_0, routers and norms F32, output Q6_K, gate/up experts IQ3_S, down experts IQ4_XS in 37 layers and
Q6_K in 3. Step 12 records the UD-Q3_K_M types from the pod file with `tools/strata_inspect.py` before the run.

| Stage | `strata-poly` path | llama.cpp path at the pin | Relation |
| --- | --- | --- | --- |
| Token embedding | One row on the host with ggml `to_float`, `src/poly/engine.cpp:2021-2025` | `get_rows` | Same dequantizer |
| Norm + activation quantization | `rms_norm_kernel` then `native_quantize_q8_1`, `src/poly/qwen35.cpp:456-465` (`kernels.cu:159`) | `rms_norm` f32, `quantize_q8_1` inside MMVQ (`quantize.cu:54-100`) | Same q8_1 contract |
| Gated DeltaNet (30 of 40 layers) | `qwen35.cpp:279-299`; `conv4_silu` `kernels.cu:2673-2683`, `gdn_step` `:2699-2722`, f32 state (`engine.cpp:1798-1801`) | `ggml_gated_delta_net` at `src/models/qwen35moe.cpp:472`, `gated_delta_net.cu` f32 | Same recurrence, different kernels |
| Full attention (10 of 40, head dim 256, 8 query heads per KV head) | Decode `attend()` `kernels.cu:2449-2550` (`attn_score_reg_kernel<8,2,q8>`, softmax, value kernels); prompt `attend_batch()` `:2568-2657`: cuBLAS fp16 GEMMs with f32 accumulation, online softmax with half weights | Flash attention: VEC at one token, MMA_F16 for a batch on sm_86 (`fattn.cu:626-652`) | Different kernels; both fp16 products, f32 sums |
| KV cache | fp16 by default (`src/poly/engine.hpp:39-40`); `--kv q8_0` opt-in, int8 codes + one fp16 scale per 32 values (`qwen35.cpp:354-368`, `kernels.cu:516-584`), qwen35moe only (`engine.cpp:989-990`); under q8 the prompt path dequantizes blocks to fp16 first (`kernels.cu:2626-2631`) | f16 default; q8_0 kernels compiled | Same block size as llama.cpp's q8_0; byte layout equality not verified |
| Router, top-8 of 256 | `lin(gate_inp)` F32 (`engine.cpp:2106`); `route_rows` softmax, top-8, renormalized (`kernels.cu:907`, `:2419-2432`; host copy `qwen35.cpp:478-496`) | `build_moe_ffn` in `src/llama-graph.cpp` (line not located) | Same rule; `poly_route_test` holds the GPU to the host copy |
| Routed experts, decode, every expert in VRAM (the pod) | `moe_card()` `engine.cpp:2264-2285`: `native_mmvq_ids` gate/up, `swiglu_clamp_rows`, `native_quantize_q8_1`, `native_mmvq_ids` down, `moe_sum`; whole pass one CUDA graph when every layer is card-routed (`:2817-2839`) | `mul_mat_id` MMVQ with ids (`ggml-cuda.cu:1924-1930`; caps `mmvq.cu:197-207`: Q3_K 5, IQ3_S 6, IQ4_XS 8, Q6_K 8); fused gate+up MMVQ (`:1794-1821`); dots `vecdotq.cuh:891` (q3_K), `:918` (q4_K), `:1019` (q6_K), `:1357` (iq4_xs) | Transcribed dots; strata-poly sums experts in its own order |
| Routed experts, decode, some experts on the CPU (smaller cards) | `moe()` `engine.cpp:2119-2234`; `src/poly/cpu_experts.cpp:163-229` through ggml-cpu `vec_dot` with Q8_K activations; `--tail-skip` drops low-ranked missing experts (`engine.cpp:2158-2169`) | CPU backend `arch/x86/quants.c:1766` (q3_K), `:4004` (iq4_xs) | Not exercised on 24 GB; relevant to the 1080 Ti rung |
| Routed experts, prompt (32 rows or more) | `moe_rows()` `engine.cpp:2323-2606`: `Mmq::quantize` (ggml `quantize_mmq_q8_1_cuda`, `src/poly/mmq.cu:66-81`) and `mmq_->mul` (`:83-119`, `mul_mat_q_case<T>`), gate/up `:2528-2530`, down `:2536-2538` | `mul_mat_id` MMQ branch (`ggml-cuda.cu:1940-1943`) with the scatter quantizer (`quantize.cu:606`) | Same kernel templates; poly fills `mmq_args` itself (`mmq.cuh:1378-1386`) and picks the tile from the largest row count (`mmq.cu:40-59`) |
| Prompts shorter than 32 tokens, and every prompt's last token | Decode kernels (`src/poly/main.cpp:399-414`) | One batch through MMVQ or MMQ by size | A chunk boundary moves which kernel a token meets (section 1.4) |
| Shared expert | `qwen35.cpp:504-511`: `lin` on Q8_0 gate/up (`native_q8_0_mmvq` or MMQ q8_0), `swiglu_clamp`, `lin` down, sigmoid gate | `build_moe_ffn` shared path; `vec_dot_q8_0_q8_1` `vecdotq.cuh:851` | Same dots |
| Dense attention and SSM projections (Q8_0 in the local file) | `native_q8_0_mmvq` / MMQ q8_0 through `lin()` | MMVQ / MMQ q8_0 | Same dots |
| Head and argmax | `lin(output_)` always one token in `eval` (`engine.cpp:3047`), up to 8 in a window (`:3339`), Q6_K via `native_mmvq`; `most_likely` on the host, first index of the largest logit (`engine.cpp:3493-3496`, `main.cpp:98`) | `vec_dot_q6_K_q8_1`; first argmax (`llama-sampler.cpp:270-285`) | Same tie rule |
| Draft block (MTP, 1 extra layer) | Found by `nextn_predict_layers` (`engine.cpp:887-901`), loaded by `load_drafter` (`:1753-1765`); `draft_input` `:3411-3433`, `block_attend` `:3451-3470` with the head over the first 131,072 rows (`src/poly/qwen35.hpp:28-30`); up to 3 guesses (`main.cpp:66-68`), suffix lookup up to the window (`:70-72`, `--lookup 3` default `:1152`); acceptance `main.cpp:516-529`: kept while `pick(logits_of(right)) == w[right+1]`, `pick` = `most_likely` at temperature 0 | `--spec-type draft-mtp` loads and runs the block (`common/arg.cpp:4236-4244`, `src/models/qwen35moe.cpp:106-138`, `:550`); without it the block's tensors are skipped (`qwen35moe.cpp:41`) | The reference runs without a draft; `engine.hpp:226-229` claims a kept token's logits equal `eval()`'s, so draft on must equal draft off bitwise (T0) |
| Sampling | Defaults greedy (`main.cpp:79-86`, `:98`); `GEN` keys `:541-582`; the front end sends no keys at temperature 0 (`frontend/strata_context.cpp:330-343`) | `--temp 0` first argmax | Same |

### 1.4 Settings on the pod run lines that move the output

Each of these changes bits or ids on purpose. A comparison holds them still, and the table says how. A row marked
"inert on 24 GB" is held anyway so the same command works on a smaller card.

| Engine | Setting | Default on the pod | Effect | Hold for a check |
| --- | --- | --- | --- | --- |
| A | `STRATA_ROUTE_TAIL_SKIP` | 7 (serve, CUDA; `generate.cpp:9375-9420`) | Drops low-ranked missed experts, KL ~0.003-0.006 (`:9376-9377`) | `STRATA_ROUTE_TAIL_SKIP=0` |
| A | `STRATA_PREFILL_CPU_SHARE` | auto for chunks under 1024 tokens (`engine-info.txt`; `generate.cpp:7812-7840`, `src/prefill/prefill.cpp:185-215`) | CPU computes some prompt experts with Q8_K rounding, mean KL ~0.004 (`engine-info.txt`) | `STRATA_PREFILL_CPU_SHARE=0`; `docs/DETAILS.md:1989` records first logits byte-identical across chunk splits with it off |
| A | adaptive expert tier | every 4 rounds, 96 swaps (`generate.cpp:677`, `:707`) | Moves experts between GPU and CPU tiers, which round differently | `--adapt-every 0` (`:9423` disables the tier at 0; upstream used `--adapt-every 100000`, `docs/UNSLOTH_Q4.md:198`) |
| A | `pcie_frac` | 0.55 then refined (`generate.cpp:671-672`) | Same arithmetic as the VRAM tier, but the split changes which experts are GPU-rounded | `--pcie-frac 0.55` explicitly, so it is not refined |
| A | `STRATA_IQ_MT_MIN` | 2 (`native_expert.cpp:131`) | A CPU expert's rounding depends on how many window tokens share it (`docs/DETAILS.md:99-104`) | `STRATA_IQ_MT_MIN=1` |
| A | suffix drafter | on, widens `--spec 4` to 6 (`generate.cpp:2989-2992`) | Changes window composition only | Leave on for the "draft on" arm; `--suffix-draft 0 --spec 4` is the T=4 arm |
| A | prompt cache | 6 checkpoints (`generate.cpp:708-711`) | A resumed prefix carries the earlier run's prompt-path bits | `--prompt-cache 0` and one engine process per sequence |
| A | `--kv int8`, `--kv-resident` | int8, 32768 | Lossy format; streaming claimed exact | Ladder: `--kv fp16` and resident first, then int8, then streamed (section 3.4) |
| B | `--tail-skip 6` | inert on 24 GB (every expert resident) | On a smaller card drops missed experts | `--tail-skip 0` |
| B | draft block and suffix lookup | on, window 8 (`engine.cpp:1687`), lookup 3 | Window composition only, claimed bitwise inert | Base arm `--no-draft --lookup 0 --window 1`; second arm defaults |
| B | prompt cache | 6 checkpoints; a prompt that extends the engine's sequence resumes even at 0 (`src/poly/conversations.hpp:234`) | Which tokens share a chunk, so MMQ versus MMVQ and `attend_batch` versus `attend` (`conversations.hpp:106-127`, `main.cpp:399-414`) | One `--tokens` process per sequence, same `--prompt-batch` both arms |
| B | expert placement | all resident on 24 GB | On a smaller card, `STRATA_FREEZE_EXPERTS=1` holds the split (`main.cpp:1094-1096`) | Set it anyway |
| B | `--kv q8_0` | off | Lossy format | Ladder as for A |
| B | `--kv k16v8` (fp16 keys, Q8_0 values) | off | Lossy in V only; the scores are `--kv fp16`'s | Ladder as for `q8_0`; the scores' half is T2-exact against `base` by construction, so the KL against it is V's error alone |

## 2. Upstream's parity and bench programs

### 2.1 How they are built, and what cannot be

- `STRATA_BUILD_TESTS` defaults on only when both `tests/CMakeLists.txt` and `bench/micro` exist
  (`CMakeLists.txt:107-111`); `tests/CMakeLists.txt` is absent in this tree and upstream's, and `bench/micro`
  holds one file. `make build` configures with `-DSTRATA_BUILD_TESTS=OFF` (`Makefile:57`) and builds only
  `strata strata-poly strata_kq unbiased-strata` (`Makefile:49`, `:63`). Step 12 passes
  `CMAKE_EXTRA=-DSTRATA_BUILD_TESTS=ON` (`Makefile:60`) and names the test targets to `cmake --build`.
- Most GPU parity programs sit inside `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` (`CMakeLists.txt:320`)
  at `:850-1234` and are not gated by `STRATA_BUILD_TESTS`. Settled by reading the nesting: between `:320` and
  its `endif()` at `:1297` the only conditions around `:850-1234` are `STRATA_ENABLE_CUDA` and
  `STRATA_HIP_GFX906` (`:909`, `:921`, `:937`, `:953`, `:1011`, `:1025`, `:1063`); the first
  `STRATA_BUILD_TESTS` gate after `:850` is `:1237`. `enable_testing()` at `:589` sits inside
  `if(EXISTS tests/core/file_expert_source_test.cpp)` (`:588`), and that file exists. So the normal `make build`
  configure defines these targets; they are not built because `make build` names only its four targets.
- The CUDA oracle block (`CMakeLists.txt:527-549`) cannot build: its 16 sources
  `bench/micro/native_*_parity.cpp` do not exist in either tree, so the one set of tests that would hold the
  `native_*` kernels to a pinned llama.cpp CUDA build (`:537-543`, `--selftest` at `:544`) is unavailable.
  Setting `STRATA_ORACLE_SOURCE_DIR` fails the configure on the missing files.
- Also missing sources, skipped by `EXISTS` guards: `native_mmvq_multi` (`:1237-1242`),
  `graph_registry_test` (`:1273-1278`), `pinned_capture` (`:1285-1290`), `hit_cpu_order_parity`
  (`:1899-1902`).
- `ple_parity` (`:1180-1183`) needs `bench/micro/ple_in.bin` and `ple_out.bin` (`:1178`), absent; it fails
  by construction, as `docs/Q8_KV.md:151` already records upstream.
- `iq_parity` is registered only when a python3 with numpy is found (`:870-905`); the fixture generator is
  `tools/iq_fixture.py` over the vendored `gguf-py`, exit 3 (skip) when numpy is missing.
- Many fixtures hard-code the old model's width: n_embd 2560, expert width 640
  (`src/kernels/native_expert_parity.cpp:47`, `tests/cuda/prefill_fused_iq_test.cpp:41`,
  `prefill_mmq_kquant_test` 1280x2560 / 2560x640). They test the kernels, not Qwen3.6's or Flash-Next's shapes
  (Flash-Next: `include/strata/core/geometry.hpp:25-49`; Qwen3.6: n_embd 2048, `tools/qwen35/parity.py:24`).
- Two scripts that look like gates never fail: `tools/qwen35/parity.py` returns 0 after printing (`:108`),
  `tools/qwen35/kl.py` states a gate in its docstring ("KL within 1.5x of the floor", `:4`) and returns 0
  without testing it (`:96-102`); `compare_dumps.py` likewise. Step 12 applies the criteria from their output.

### 2.2 Programs covering a path in use

"Registered" means a CTest entry exists; "hand" means the target builds and is run by hand. Times are
estimates from the registered timeouts and the fixture sizes, not measurements.

| Program (source) | Registration | Path covered (section 1) | Compares against | Needs | Time | Pass criterion |
| --- | --- | --- | --- | --- | --- | --- |
| `iq_parity` (`src/kernels/iq_parity.cpp`) | `CMakeLists.txt:897-903` | A decode: GPU dequant and `native_mmvq` for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0, Q3_K (`:29`); B decode: Q3_K, IQ4_XS, IQ3_S | gguf-py dequantized synthetic fixture; double-precision product | GPU, python3 + numpy | under a minute | dequant rel L1 < 1e-6, mmvq rel L1 < 2e-2, multi-column bitwise single-column (`:96-99`, `:112`) |
| `iq_multi_parity` (`src/kernels/iq_multi_parity.cpp`) | `:950` | A decode: `iq_mmvq`, `native_expert_grouped` new vs old kernels, q8_1 quantizers | The engine's own previous kernels; GPU dequantizer as a gross-error guard | GPU | minutes | memcmp per output, finite, ref rel < 1e-2 (`:146`, `:158`, `:259`) |
| `native_grouped_parity` (`src/kernels/native_grouped_parity.cpp`) | `:990` | A decode: grouped expert kernels and fused SwiGLU + q8_1, gate/up {16,17,18,21,22,23,29,42,12,13,8}, down {20,23,42,7,8} | v1 launches | GPU | minutes | bitwise, including rows it must not write (`:10-11`) |
| `native_expert_parity` (`src/kernels/native_expert_parity.cpp`) | gated, `:1858-1883`; GGUF mode by hand | A decode: float ref vs CPU path vs GPU grouped path vs dequantizers; AVX-2 IQ3_XXS/IQ3_S/IQ2_S exact singleton vs ggml (`:103-119`) | ggml `to_float` float expert; ggml-cpu `vec_dot` | GPU, CPU; synthetic pairs are K-quant/Q8_0 (not the pack's formats) | 120 s timeout each | CPU and GPU rel L1 < 3e-2 vs float (`:396`); dequant rel < 1e-6; multi vs one memcmp (`:195`, `:199`, `:287`, `:344`) |
| `iq_avx2_parity` (`src/kernels/cpu/iq_avx2_parity.cpp`) | gated, `:1550-1551` | A decode, CPU tier: every AVX-2 variant vs scalar, 1..8 tokens, gate/up {16,17,22,18,21,23}, down {20,42} | ggml-cpu `vec_dot` | CPU with AVX-2 (the pod) | minutes | rel <= 1e-5 vs ggml (`:250`); exact singleton memcmp for 16,17,18,21,22 (`:251-268`); multi-token rows bitwise the one-token rows (`:270-284`) |
| `q8k_quant_parity` (`src/kernels/cpu/q8k_quant_parity.cpp`) | gated, `:1565-1566` | A CPU tier: Q8_K activation quantizer | ggml `from_float_ref` | CPU | seconds | byte for byte (`:1-4`) |
| `prefill_fused_iq_test` (`tests/cuda/prefill_fused_iq_test.cpp`) | gated, `:1717-1720` | A prompt: MMQ for IQ3_XXS/IQ4_NL, IQ3_XXS/Q2_0, IQ4_XS/Q2_0 pairs (`:661-673`), and the fused kernels against it | double product of ggml-dequantized weights | sm_80+ GPU (exit 77 otherwise) | 600 s timeout | fused rms <= 2e-3, worst <= 3e-2 (`:408`); fused <= 1.5x MMQ rms (`:423`, `:631`). Note: MMQ itself is held to no absolute bound here (`:419` bounds the path without MMQ at rms 0.03) |
| `prefill_mmq_scatter_test` (`tests/cuda/prefill_mmq_scatter_test.cpp`) | gated, `:1723-1726` | A prompt: scatter quantizer vs per-row | ggml per-row q8_1 bytes | GPU | 120 s | byte diff == 0 (`:86-88`) |
| `prefill_mmq_kquant_test` (`tests/cuda/prefill_mmq_kquant_test.cpp`) | gated, `:1737-1741`, `STRATA_MMQ_KQ=1` | `strata` K-quant prompt path (not A, not B) | double product | GPU | 120 s | rel_l2 <= 0.04, max/rms <= 0.35 (`:141`) |
| `mmvq_multi_parity` (`src/kernels/mmvq_multi_parity.cpp`) | `:915` | B decode: `native_mmvq` T columns vs T single columns, Q4_K, Q5_K, Q6_K, IQ4_XS, Q8_0 (no Q3_K, `:58-65`) | itself (negative control with exact mode off) | GPU | minutes | 0 differing bits in exact mode; control must differ (`:216-262`) |
| `poly_expert_parity` (`src/poly/expert_parity.cpp`) | hand, `:2075-2079`; `poly_expert_parity MODEL.gguf` | B decode and prompt on the real file: `native_mmvq` and `cpu_experts` per (gate/up, down) type pair, experts 0, n/2, n-1; every matrix kind through `native_mmvq` and MMQ on a 40-token batch, 1 MiB of 0xFF past the weights (`:203-274`) | double product over ggml `to_float` (`:41-55`) | GPU, the model file | minutes per file | relative L2 < 0.05 CPU and GPU, two experts in one `native_mmvq_slots` launch bitwise one (`:187-194`); vec and mat < 0.05 (`:275`) |
| `poly_route_test` (`src/poly/route_test.cu`) | hand, `:2094` | B router: `route_rows` softmax 256 experts top-8 with norm (`:115`), 4096 random rows with ties | host `route()` | GPU | seconds | picks identical except rounding ties (rel <= 4e-7, `:97`); weights rel < 1e-5 (`:104`) |
| `poly_norm_q8_test` (`src/poly/norm_q8_test.cu`) | hand, `:2099` | B norm + q8_1 fused kernel, n includes 2048 | `axpy` + `rms_norm` + `native_quantize_q8_1` | GPU | seconds | bytes equal (`:51-55`) |
| `poly_gemm_small_test` (`src/poly/gemm_small_test.cu`) | hand, `:2105` | B router and small F32 projections in a prompt chunk | host `fma` chain | GPU | seconds | memcmp (`:41`) |
| `poly_attend_test` (`src/poly/attend_test.cu`) | hand, `:2110` | B attention kernels, but at dim 64, 32 heads, 8 KV heads (`:23`): LFM2's shape, not Qwen3.6's | the engine's other kernels; `attend_batch` vs `attend` | GPU | seconds | <= 1e-5 and bitwise alone vs on-card (`:66-70`); batch <= 4e-3 (`:96`) |
| `poly_kv_q8_test` (`src/poly/kv_q8_test.cpp`) | gated, `:2136-2141` | B `--kv q8_0` store, and the `--kv k16v8` read | host arithmetic | GPU | seconds | round trip within half_scale + 5e-4·x + 1e-3 (`:87`); codes and scales memcmp (`:139`, `:150`); ring memcmp (`:198`, `:207`); `attend()` fp16-K + Q8_0-V within 5e-3 of the same attention over the dequantized values as fp16; `attend_batch()` the same within 1e-5 (one workspace half, `kwork` null) |
| `poly_draft_race_test` (`src/poly/draft_race_test.cpp`) | hand, `:2083-2090` | B draft block: repeated reads with the block give the same guesses | run 1 of 8 | GPU, the model file | minutes | guesses equal, probs memcmp, state equal (`:94-100`); **SKIPs (exit 3) with every expert cached (`:84-88`), so it skips on the pod** |
| `kv_q8_parity` (`src/kernels/kv_q8_parity.cpp`) | `:1188` | A int8 KV append and gather | host reference; the FP16 path | GPU | seconds | codes and scales bitwise (`:96`); gather bitwise (`:128`); int8 vs fp16 within 0.5 + 2·127/2048 + 1e-3 steps (`:138`) |
| `kv_stream_parity` (`src/kernels/kv_stream_parity.cpp`) | `:1194` | A KV streaming: streamed vs resident attention, int8/fp16/q4_0, ring restore | resident path | GPU | seconds | memcmp (`:190`, `:232`) |
| `kv_steps_parity`, `kv_window_parity` (`src/kernels/kv_steps_parity.cpp`, `kv_window_parity.cpp`) | `:1107`, `:1201` | A verify window: batched KV appends n = 1..8 and the catch-up window, graph replay, streamed layer | per-cell path | GPU | seconds | every pool byte memcmp (`:45`, `:124`); streamed attention memcmp resident (`:280`), ring (`:285`), slots (`:305`) |
| `gdn_parity` (`src/kernels/gdn_parity.cpp`) | `:1038` | A GDN decode: recurrence, conv, norms; multi-token vs fused | `ref/gdn.py` reference; `fused_gdn_*` | GPU | seconds | rel <= 1e-5 (`:202-203`), conv and norms <= 1e-6 (`:299-300`, `:341`, `:374`), multi memcmp (`:451`, `:508`) |
| `gdn_rec_parity` (`src/prefill/gdn_rec_parity.cu`) | `:966` | A GDN prompt recurrence (copies of the kernels, `:3-9`, not the live `src/prefill/kernels.cu`) | kh kernels; FP64 | GPU | minutes | bitwise; chunked < 1e-4 of max (`:880`) |
| `rope_parity`, `elementwise_parity` (`src/kernels/rope_parity.cpp`, `elementwise_parity.cpp`) | `:1073`, `:1140` | A rope tables and rotation; silu, softplus, rms norm | float64 `ref/qsa.py`; host | GPU | seconds | rotation rel <= 1e-6 (`:147`, `:341`); <= 1e-6 / 1e-7 (`:116`, `:177`, `:290`) |
| `qsa_parity`, `qsa_topk_parity`, `qsa_select_bench`, `qsa_prompt_attn_parity` (`src/kernels/qsa_*.cpp`) | `:1155`, `:1162-1164`, `:493-494`, hand `:427-432` | A QSA selection and attention, decode and prompt | host refs; FP64; the warp kernel | GPU | minutes | `qsa_parity` <= 1e-6 .. 1e-3 by stage (`:611`, `:733`, `:1117-1154`); top-k ids identical (`:1159-1161`); prompt attention err <= max(4x old, 1e-6·scale) (`:318`, `:326`) |
| `native_multi_parity`, `router_top10_parity`, `route_window_parity` (`src/kernels/native_multi_parity.cpp`, `router_top10_parity.cpp`, `route_window_parity.cpp`) | `:1102`, `:1097`, `:1084` | A router: multi-token top-10 vs per token, combine vs host replay (2560-wide fixtures) | per-token kernels; host | GPU | seconds | memcmp ids and weights (`:85`, `:165`, `:176`); memcmp (`:48`, `:65`) |
| `verify_parity`, `spec_verify_parity`, `verify_batch_parity` (`src/kernels/verify_parity.cpp`, `spec_verify_parity.cpp`, `verify_batch_parity.cpp`) | `:1015`, `:1054`, `:1218` | A head argmax and probabilities over vocab 248320 with ties, NaN, -inf; spec sampler; PLE batch snapshot | `sample_tokens` greedy; host `spec_prob.hpp` | GPU | seconds | id equal and p memcmp (`:93`); `out[t] == want` (`:233`); bitwise (`:69-72`) |
| `sampler_parity` (`src/kernels/sampler_parity.cpp`) | `:1048`, `:1057`, `:1059` | A sampler chain, greedy fixtures: plain argmax, greedy over fixtures, greedy consumes no random number, penalties, logit bias (`:535-610`, `:645`, `:487-530`) | host reference | GPU | seconds | token equality (`:124`, `:512`, `:1185`, `:1261`); returns 1 on any failure (`:1427-1428`) |
| `decode_cluster_parity` (`src/kernels/decode_cluster_parity.cpp`) | `:1067` | A greedy cluster sampler, QSA top-k cluster | one-block kernels | sm_90 (exits 0 below it: a skip on the pod) | seconds | bitwise (`:152-153`, `:230`, `:291`) |
| `quantize_act_parity`, `dequant_wide_parity`, `pdl_parity`, `dma_batch_parity`, `batch_rows_parity` | `:1078`, `:932`, `:941`, `:958`, `:1223` | A: Q8_0 activation bytes; wide dequant; PDL; DMA batch; batch rows | host transcription; per-group; plain | GPU | seconds | bytes / bitwise |
| `strata-poly --check-batch K` (`src/poly/main.cpp:968-1021`) | built in | B: K prompts alone then together, token by token | itself | GPU, model | minutes | returns 1 on a difference |
| `poly_workers_test`, `poly_guesses_test` (`src/poly/workers_test.cpp`; `CMakeLists.txt:131-133`) | `:2125-2128`, `:131-133` | B thread pool; guess policy | fixed counts | CPU | seconds | counts; policy |

### 2.3 Paths in use that no in-tree program holds to an external reference

- Configuration A: hyper-connections (`gr_*`, only self-consistency and the old model's `gr_parity` family
  `:982-1124`), PLE (`ple_parity` cannot run), the QSA decode attention against anything but the engine's own
  older kernels, the Q5_K native head, the pack's embedding gather, the MTP draft layer's arithmetic (by design
  unobservable in the ids), MMQ for IQ3_XXS/Q2_0 pairs to an absolute bound (`prefill_fused_iq_test` bounds the
  fused kernel relative to MMQ), the PCIe tier's staging, the expert combine order, and the whole decode window
  against a per-token run at the pack's shapes (no T=1 path exists for a native pack, so the T0 claim of
  `verify.hpp:1-8` is only testable window-against-window).
- Configuration B: attention at head dim 256 with 8 heads per group (`poly_attend_test` is dim 64), the GDN
  kernels (`gdn_parity` tests `src/kernels/cuda/gdn.cu`, not `src/poly/kernels.cu`), `moe_rows()`'s MMQ
  call with its own `mmq_args` (only `poly_expert_parity`'s 40-token batch at rel 0.05), the draft block's
  equality claim on 24 GB (`poly_draft_race_test` skips), the Q8_0 KV format's byte layout against llama.cpp's
  q8_0, `most_likely`, the host router `route()` (only the GPU copy is tested against it), the prompt-cache
  resume's chunking.
- Both: nothing in the tree compares either measured engine's logits with llama.cpp's; `tools/qwen35` targets
  `strata-q36ref` and `strata-q36` (`src/qwen35/`), which refuse Q3_K experts (`docs/BACKENDS.md:1897`) and
  are not what the fork runs (`docs/MODEL-FILES.md:13-14`). The fork's own upstream comparison so far is one
  12-token greedy answer compared as text
  (`bench/unbiased/2026-10-17-3090pod-flash-next-iq3xxs-step6/upstream-compare/flash-next-upstream-compare/fork-greedy.json`).

### 2.4 `tools/qwen35` and other scripts

| Script | What it drives | What it compares | Usable here |
| --- | --- | --- | --- |
| `tools/qwen35/q36_llama_dump.cpp` | llama.cpp's API over a checkout (`:6-8` build line); `--model --tokens|--prompt --n --ngl --dump-layers --dump-logits --force --dump-full-logits`; the prompt in one `llama_decode` batch, then one token per step (`:130-147`); `--force` feeds the given ids (teacher forcing, `:108-109`, `:142`); `--dump-full-logits` writes f32 `[step][vocab]` (`:140`); `--ngl 0` is the CPU backend with op offload off (`:100`); `STRATA_DUMP_CTX` sets n_ctx and n_batch (default 4096, `:94-99`); `STRATA_DUMP_NGL` for partial offload (`:72-74`); layer outputs captured by name `l_out-N` (`:30-40`) | nothing; it is the reference side | Yes, for both models: it uses no model-specific code; `l_out-N` capture for `qwen4exp` needs a check that the graph names its layer outputs so |
| `tools/qwen35/parity.py` | `strata-q36ref` vs the dump, 10 ChatML prompts (`parity_prompts.json`), `--n 64` | greedy ids up to llama.cpp's first `<|im_end|>` (`:75-80`), first divergence with both top-2 margins (`:83-86`), per-layer relative error (`:88-98`) | Not as is: `strata-poly` has no `--dump-logits`/`--dump-layers` file writers (section 3.7); the metric definitions are reused |
| `tools/qwen35/kl.py` | the dump at `--ngl 99` and `--ngl 0`, the reference engine `--force`d on llama.cpp's answer (`:62-75`) | full-vocab KL in float64, top-1 agreement, ratio engine/floor (`:85-101`) | The metric and the floor; the driver needs a `--top-probs` reader for `strata-poly` and a `STRATA_LOGPOS` reader for `strata` |
| `tools/qwen35/compare_dumps.py`, `q36_lgcmp.py` | two dump directories; two full-logit files | noise floor; max dlogit, KL, top-1 | `q36_lgcmp.py` compares the two llama.cpp runs directly |
| `tools/qwen35/q36_harness.py` | `exe --serve` over the engine protocol (`:21-22`, `:56`, `:76`) | n/a | Sends the `GEN` lines for the `strata` side without the removed `serve/` package |
| `tools/split_mtp_batch_parity.py` | an HTTP server | texts identical with and without the drafter (`:44-53`) | Text only, and a hard-coded model name (`:35`); superseded by the id comparisons below |
| `docs/UNSLOTH_Q4.md:192-228` | `strata --serve` with `STRATA_LOGPOS` + `STRATA_LOGPOS_TOPK=20`, `--short-read` covering the positions, `--adapt-every 100000`; llama.cpp CPU build at `3cf0325` over `llama_get_logits_ith` | argmax agreement, top-5/10 overlap, KL over llama.cpp's top 20 + rest, perplexity; near-tie analysis by llama.cpp's top-2 gap (`:213-221`) | The procedure for configuration A, repeated with a CUDA reference added |
| `docs/BACKENDS.md:1022-1051` | `strata-poly` vs llama.cpp, five chat prompts, greedy 64 | same ids (Qwen3.6 UD-Q3_K_M: 64, 64, 40, 64, 64 of 64), largest top-5 probability difference 0.10; "Two correct implementations do not write the same tokens for long" (`:1048`); llama.cpp's own GPU and CPU runs agree 17 of 50, 7 of 64, 15 of 64 on Ornith (`:1051`) | The script is not in the tree; the method is section 3.3 |

## 3. Teacher-forced greedy comparison against llama.cpp

### 3.1 The reference side

- **Build.** `third_party/llama.cpp` at `3cf03257` with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`
  (the pod's card), targets `llama`, `ggml`, `ggml-base`, `ggml-cpu`, `ggml-cuda` and `llama-server`. The pod's
  existing build is `797de63` (`PLAN.md:57`) and is not used: a different commit is a different reference.
- **Program.** `tools/qwen35/q36_llama_dump.cpp` compiled against that build (`:6-8`). It reads the same
  token ids both engines get (`--tokens`, no tokenizer), runs the prompt as one batch and the answer one token at
  a time, and with `--force` writes every step's full f32 logits (`--dump-full-logits`). Greedy is its own loop,
  first index of the maximum (`:133-136`), the same rule as `llama_sampler` (`src/llama-sampler.cpp:270-285`),
  `strata` (`sampler.cu:116`) and `strata-poly` (`engine.cpp:3493-3496`).
- **Two runs per sequence.** `--ngl 99` (GPU: MMQ for the prompt batch, MMVQ for the steps, flash attention
  f16 KV) and `--ngl 0` (CPU backend: Q8_K activations). KL and top-1 between them is the floor
  (`tools/qwen35/kl.py:1-8`, `docs/QWEN36.md:33-35`).
- **Flash-Next on 24 GB.** The IQ3_XXS shards are 47.0 GB + 28.8 GB (`docs/MODEL-FILES.md:12`): `--ngl 99`
  does not fit. The CPU run is the primary reference, as in `docs/UNSLOTH_Q4.md:194`; the GPU run uses
  `STRATA_DUMP_NGL=<n>` with as many whole layers as fit beside the compute buffers. A copy of the dump tool with
  llama-server's `--override-tensor "exps=CPU"` semantics (`common/arg.cpp:2750`, `tensor_buft_overrides`)
  would put every dense layer on the GPU and the experts on the CPU; that is a step-12 tool, listed in 5.3.
  The GSQ-RCO file's tensor set is the pinned loader's: every `create_tensor` of `src/models/qwen4exp.cpp:163-257`
  (plus the q/k/v, gate/up-experts and `ssm_a` helpers at `:218`, `:234`, `:252`) has its tensor in the shard-1
  header, and the header holds nothing the loader does not create, so `done_getting_tensors`
  (`src/llama-model-loader.cpp:1388-1398`, which throws on a count mismatch) is satisfied; the converter drops
  the MTP head (`conversion/qwen4exp.py:28-30`) and the file has no such tensor. What the header cannot show is
  op support for the types at load, which the first load shows.
- **Qwen3.6 on 24 GB.** 17.1 GB fits with `--ngl 99` at n_ctx 4096. The Unsloth file's draft-block tensors are
  skipped without `--spec-type draft-mtp` (`src/models/qwen35moe.cpp:41`); the reference runs without a draft.
- **Settings.** No chat template (ids are given), no BOS (Qwen adds none, `q36_llama_dump.cpp:12`), n_ctx and
  n_batch = `STRATA_DUMP_CTX` large enough for prompt + answer, defaults otherwise. If `llama-server` is used
  for anything, `/completion` with `"prompt": [ids]` passes ids through untouched
  (`tools/server/server-common.cpp:988-991`), with `"cache_prompt": false`, `"temperature": 0`, `"top_k": 1`
  set explicitly (GGUF `general.sampling.*` overrides unset values, `common/common.cpp:1217-1267`), `-fit off`.

### 3.2 Inputs

Sequences are prompt ids followed by llama.cpp's own greedy answer ids (the GPU run's), so both engines are
scored on the same continuation, the one llama.cpp reached (`kl.py:62-64`). Per model:

| Set | Prompts | Tokens scored | Why |
| --- | --- | --- | --- |
| Short chat | the 10 of `tools/qwen35/parity_prompts.ids` (20-40 tokens each, ChatML, thinking off) for Qwen3.6; the same texts re-tokenized through the front end for Flash-Next (`llama-tokenize --ids`) | 128 answer tokens each, 1,280 per model | Decode kernels; matches upstream's own sets |
| Code and reasoning | 2 prompts: an LRU cache in Python; a reasoning puzzle with thinking on (`docs/UNSLOTH_Q4.md:204-205`) | 400 and 324 | Long answers at low near-tie rates |
| Long prompt | 3 documents of 4K, 16K and 31K tokens from this repository's docs (the step-6 smoke texts, `bench/unbiased/*-step6/smoke/`), each followed by its own 256-token greedy continuation | 768 | The prompt path (MMQ, batched attention, chunking) and the tail of the context: KV streaming on A needs more than 20,480 cells (`generate.cpp:811-813`), so the 31K prompt with `--kv-resident 20480` is the streamed case; B's context is 32768 |
| Real text | the last 510 tokens of the 16K document, scored as prompt positions | 510 | Positions the model did not write itself (`docs/UNSLOTH_Q4.md:207-208`) |

Every id list is written to `bench/unbiased/correctness/inputs/` once and used by both engines and both
reference runs.

### 3.3 Procedure for configuration B (`strata-poly`)

For each sequence, a fresh process per arm:

```
strata-poly --model F --max-context 32768 --tail-skip 0 --no-draft --lookup 0 --window 1 \
            --tokens <prompt ids> --follow <answer ids> --max-new <len(answer)> --top-probs 20 \
            2> top.txt > ids.txt
```

- `--follow` writes the given answer in place of the picks (`src/poly/main.cpp:334`, `:463-464`), so every
  step is scored on llama.cpp's token; `--top-probs 20` prints `top id:prob ...` per step from the same logits
  (`:927-940`, softmax over the full vocabulary, `%.6g`). `ids.txt` is the engine's own argmax per step.
- The first step's distribution comes after the prompt path (chunks of `--prompt-batch` 1024 through MMQ and
  `attend_batch`, the last token alone through MMVQ and `attend`, `main.cpp:399-414`); every later step comes
  through the decode kernels. The long-prompt set therefore scores the prompt path at step 0 and the decode path
  after it.
- Arm 2, the same command without `--no-draft --lookup 0 --window 1` (defaults: draft block, lookup 3, window
  8): `top.txt` must be byte-identical to arm 1's (T0). `--follow` keeps the sequence the same, so only the
  window composition differs, which `engine.hpp:226-229` says changes nothing.
- Arm 3, `--kv q8_0` on arm 1's flags: T2 against arm 1, T1 against llama.cpp. The `k16v8` arm (`--kv k16v8`,
  the same flags) is T2 against arm 1 and against arm 3: its scores are arm 1's, so its KL against arm 1 is the
  value cache's error with no score error in it, and it must sit between arm 1 and arm 3 on every sequence.
- Arm 4, UD-IQ4_XS, arms 1 and 2 again. Arm 5 (smaller cards only, not the pod): `--vram-reserve-mib` large
  enough to push a third of the experts to the CPU, `STRATA_FREEZE_EXPERTS=1`, `--tail-skip 0`: scores the
  `cpu_experts` path (section 1.3).
- Scoring: a script (section 5.3) reads `top.txt` and the reference's full logits, and per step writes the
  argmax agreement, llama.cpp's top-2 gap, top-20 overlap, and KL(llama GPU || engine) over llama.cpp's top 20
  plus one bucket for the rest (the engine prints 20 probabilities, so the rest is `1 - sum`), the same KL
  definition as `docs/UNSLOTH_Q4.md:198-200`; and the same for KL(llama GPU || llama CPU) from the two
  reference files (full vocab, `kl.py:85`).
- `--prompt-batch` is left at 1024 in every arm; a run with `--prompt-batch 512` on the 16K document is one
  extra arm, scored at step 0 only, to show how far the chunk size alone moves the distribution (upstream's
  figure for `strata-q36`: KL 3.6e-3 to 4.2e-3, `docs/QWEN36.md:172-174`).

### 3.4 Procedure for configuration A (`strata`)

`strata` has no `--follow`; its teacher-forced hook is the serve path's `STRATA_LOGPOS` (`generate.cpp:12550-12562`):
every token a request reads through the decode windows is scored against the prompt's own next token, and
`Verifier::window_logprobs` (`src/core/verify.cpp:3072-3137`) appends one line per position: position, target
id, log p(target), top id, log p(top), top == target, log p(`<|im_end|>`), log p(target) without that token, and
with `STRATA_LOGPOS_TOPK=20` the 20 likeliest `id:logprob` pairs (`:3122-3134`). So the sequence (prompt + answer)
is sent as one prompt, and the positions to score are read through the windows by setting `--short-read` to at
least the sequence length (`generate.cpp:909`, `:2524`; default 64, `:748`); positions above `--short-read` go
through the batched prompt path and are not scored.

Per sequence, a fresh engine process (the pod's run line with these changes), driven over the protocol by
`tools/qwen35/q36_harness.py` or a 20-line replacement:

```
STRATA_LOGPOS=pos.tsv STRATA_LOGPOS_TOPK=20 STRATA_ROUTE_TAIL_SKIP=0 STRATA_PREFILL_CPU_SHARE=0 STRATA_IQ_MT_MIN=1 \
strata --serve --pack P --native S1 --ple-gguf S2 --expert-profile E --expert-cache auto --prefill auto \
       --mtp M --spec 4 --spec-min-p 0.5 --max-context 65536 --ple-io ram --pool-workers 15 \
       --prompt-cache 0 --adapt-every 0 --pcie-frac 0.55 --short-read 4096 --kv fp16
GEN 1 <prompt ids + answer ids>
```

- Arm 1 (above): fp16 KV, resident, drafts on. Scores the decode windows on every position of the short sets.
  `--max-new 1` style requests (`GEN 1`) so nothing free-running is scored.
- Arm 2: `--suffix-draft 0` (T=4 windows): T0 against arm 1 on `pos.tsv` (the log-probabilities are printed to
  9 decimals, `verify.cpp:3119`; equality of those is the check, since the window composition is the only
  difference and `verify.hpp:1-8` claims per-token arithmetic is the single-token kernel's). If this fails with
  CPU experts in play even at `STRATA_IQ_MT_MIN=1`, rerun with `--expert-cache` sized so no expert misses
  (impossible for the whole model on 24 GB) or compare only positions whose experts all hit, from `--dump-routing`
  (`generate.cpp:864`); otherwise the claim is narrowed to the GPU tiers in the write-up.
- Arm 3: `--kv int8`: T2 against arm 1, T1 against llama.cpp.
- Arm 4: `--kv int8 --kv-resident 20480` on the 31K sequence with `--short-read 1024` (the last 1,024
  positions through the windows, the rest through the prompt path): T0 against arm 3 on those positions
  (`kv_stream.hpp:8-9`: readers unchanged).
- Arm 5: the long-prompt set with `--short-read 512`: step-0 and the 510 real-text positions score the prompt
  path (MMQ experts, FP16 dense GEMMs, tensor-core attention) through the first windows after it. A second run
  with `--prefill 512` (a chunk the user names is not capped, `generate.cpp:617`) shows the chunk effect, as in
  `docs/DETAILS.md:1989`.
- `STRATA_DUMP_FIRST_LOGITS` (`generate.cpp:14338-14346`) writes each request's first-window fp32 logits: it is
  the per-request probe when `pos.tsv` is not enough.
- Scoring: the same script as 3.3 over `pos.tsv` (log-probabilities, top 20) and the reference files.
  `STRATA_LOGPOS_EXTRA`'s column is reported, not used in the gate.

### 3.5 Keeping prompt caching and the draft layer from changing results

- One process per sequence on both engines, `--prompt-cache 0` on `strata` (every request reads from 0,
  `generate.cpp:708-709`); on `strata-poly` a `--tokens` run has no earlier sequence to extend
  (`conversations.hpp:234` is the serve path's exact-extension resume).
- Drafts: acceptance is exact match on both engines (`generate.cpp:15653`; `main.cpp:516-529`), and the
  teacher-forced hooks score the model's logits at every position regardless of what was drafted. The draft
  arms (B arm 2, A arms 1 vs 2) exist to show the T0 claim rather than to assume it.
- The expert tiers on A are frozen (`--adapt-every 0`, fixed `--pcie-frac`, `STRATA_IQ_MT_MIN=1`,
  `STRATA_ROUTE_TAIL_SKIP=0`), so two runs of one arm are expected to be byte-identical; each arm is run twice
  on one short sequence to confirm it before the full set.
- The front end is not in the loop: it would send `GEN <max_new>` with no sampling keys at temperature 0
  (`frontend/strata_context.cpp:330-343`), the same request the harness sends, and is covered by
  `frontend/tests/`.

### 3.6 Tolerated divergence, and why

- **Identical ids are not the criterion.** Both engines quantize activations to 8 bits at every matrix
  (q8_1 on the GPU, Q8_K on the CPU) as llama.cpp does, with different block sums and different reduction orders.
  Upstream measured llama.cpp against itself, GPU versus CPU, agreeing on 17 of 50, 7 of 64 and 15 of 64 greedy
  tokens on Ornith (`docs/BACKENDS.md:1051`), and `strata`'s own greedy run against its own teacher-forced rerun
  differing at 6% of positions (`docs/UNSLOTH_Q4.md:220-221`). A free-running greedy comparison measures where
  the first near-tie falls. The ids are still recorded (`ids.txt`, `pos.tsv` column 6) and reported in the
  `docs/BACKENDS.md:1024-1034` style, with the top-2 margins at the first divergence as `parity.py:83-86` prints them.
- **T1, argmax at a margin of 0.5 nats.** Upstream's data: at every position where llama.cpp's top-2 gap is
  0.2 or more on short answers (695 of 695) and 0.5 or more after a 16K prompt (144 of 144, 254 of 254) the
  argmax agreed (`docs/UNSLOTH_Q4.md:213-217`), and the largest top-5 probability difference `strata-poly` showed
  against llama.cpp on Qwen3.6 was 0.10 (`docs/BACKENDS.md:1029`). A disagreement at 0.5 nats (a 62/38 split
  between the two candidates) is outside anything quantization rounding produced in those runs, so one such
  position is a finding, not noise. The gate is 100% at the margin; below the margin agreement is reported.
- **T2, KL within 1.5x of the floor.** `kl.py:4` states the gate upstream intended; `docs/QWEN36.md:37-44`
  shows ratios of 0.3 to 1.0 on the picture check and `:172-174` ratios near 1.0 for the chunk size. The floor is
  measured on the same positions in the same run, so it absorbs the model's own near-tie density (the 16K sets
  have KL 0.035-0.061 against 0.0008-0.0016 for code, `docs/UNSLOTH_Q4.md:204-208`). Configuration A's CPU tier
  rounds like the CPU reference and its GPU tier like the CUDA reference, so A is expected between the two
  references; the gate uses the GPU reference as P and the CPU-vs-GPU KL as the floor.
- **Lossy options.** int8 KV (A) and Q8_0 KV (B) change values by design (`kv_q8_parity.cpp:138` bounds
  the per-value step). They are gated the same way, with the engine's own fp16-KV arm in P's place: KL(fp16 arm
  || int8 arm) at most 1.5x the floor, and T1 against llama.cpp holding. `docs/Q8_KV.md:40-58` (15 of 15
  identical 64-token greedy answers for q8_0) and `docs/DETAILS.md:2269` (int8 attention paths at KL 0.0016
  against a floor of 0.0012) say this is attainable.
- **T0.** Byte equality where the engine claims it; no tolerance. A T0 failure that is within T2 is still a
  failure of the claim and is written up as such.
- **Perplexity.** Reported per set (`pos.tsv` column 3 summed) as upstream does, not gated: it moves with
  near ties and quantization alike.

### 3.7 Localizing a divergence to a layer or kernel

Order: cheapest first, each step narrowing the set of kernels that can be wrong.

1. **Which arm.** The ladder (fp16 KV resident, draft off; then each option) names the option that introduced
   it. An arm-to-arm T0 failure names the window, KV streaming or the draft block directly.
2. **Which tier (A).** `--dump-routing` (`generate.cpp:864`) gives each position's experts; the residency table
   says which tier computed them. Rerun the sequence with `--pcie-frac 0` (all misses on the CPU) and with
   `--pcie-frac 1.0` (all misses streamed to the GPU): a difference that follows the tier is in that tier's
   expert kernels, and `iq_avx2_parity` or `iq_parity` at the pack's exact (gate/up, down) pair is the next
   check. The `--native-*` toggles (`generate.cpp:822-832`, `:3673-3689`) switch each native kernel family
   back to the older path one at a time.
3. **Which path (B).** `--no-gpu-experts` (`main.cpp:1138`) computes every routed expert with ggml-cpu's
   `vec_dot` and Q8_K activations, the CPU reference's arithmetic; a divergence that disappears is in the GPU
   expert kernels (then `poly_expert_parity MODEL.gguf` and `iq_parity` for that type). `--no-graphs`,
   `--prompt-batch 16` (every token through the decode kernels) and `--kv q8_0` off split the rest.
4. **Which layer.** llama.cpp's side dumps every layer output for every position
   (`q36_llama_dump.cpp --dump-layers`, `[pos][layer][n_embd]` f32, `:111-123`). `strata-q36ref` writes the
   same format, `strata-poly` does not: `--dump-layers` prints one checksum line per stage (`engine.cpp:2056-2068`;
   stages `attn_in`, `attn`, `ffn_in`, `ffn`, `out_norm`) and changes the path while doing it (window 1, rows 1,
   no graphs, host routing: `:1595`, `:1794-1795`, `:2242`). For the prompt path `strata` writes R per layer
   with `STRATA_PREFILL_DUMP_R` / `_ALL` (`src/prefill/prefill.cpp:4631`, `:4644`); for the decode windows it
   has state hashes (`STRATA_STATE_HASH`, `generate.cpp:14490`) and window hashes (`--window-hashes`, `:956`),
   bitwise self-comparison only. A per-layer vector comparison against llama.cpp for `strata-poly` needs a
   small addition to `Engine::dbg()` that writes the vectors (step-12 decision; listed in 5.3), after which
   `parity.py:88-98`'s per-layer relative error applies. For Flash-Next the names differ: `qwen4exp` names each
   layer's output `l_last-N` (`src/models/qwen4exp.cpp:441-442`, "the layer output name that build_cvec and
   imatrix look for"), and that tensor is the four-stream hyper-connection residual `[n_embd, hc, tokens]`, with
   the single-stream FFN output as `ffn_out-N` (`:437`). The dump tool matches `l_out-` only
   (`q36_llama_dump.cpp:33`) and writes rows of `n_embd` (`:111-123`), so for Flash-Next it needs the name and
   the `hc` dimension (a step-12 edit, 5.3). On the engine side the matching dump is `--dump-layers`, which
   writes `(n_layers + 1) x hc x n_embd` floats per position (`generate.cpp:6701`) and is bisected against
   `l_last-<il>` (`:862`); it belongs to the per-token loop, so for a native pack only the prompt-path dumps
   above apply.
5. **Which kernel.** With the layer known, the stage checksums (B) or the `--native-*` toggles and the
   component parities (A) point at one kernel; the kernel's parity program is run at that layer's exact shape and
   type, which for most of section 2.2 means editing a fixture size.

## 4. What a failure means, and the run order

| Check | A failure means | Action |
| --- | --- | --- |
| Section 2.1 configure and build of the test targets | The test inventory in 2.2 is not reachable from `make build`; the plan's coverage claims rest on upstream's runs | Fix the build (a fork change under `CMakeLists.txt`) before anything else; a test that cannot build is "uncovered", not "passed" |
| Component parities (2.2) at the pack's / file's formats | A kernel in use differs from ggml's arithmetic beyond its own bound, in the fork's build of the same source upstream ships; since the sources are unmodified since the import commit (`UPSTREAM.md:10`), this is a toolchain or arch-flag difference (`CUDA_ARCHS` `61;75;86;89`, `Makefile:13`) or an upstream defect | A SEV-3 ticket with the program's output; the engine's measured numbers stand only under "same output as upstream PolyStrata" until the teacher-forced check passes or fails |
| T0 (draft on vs off, window sizes, streaming) | A claimed exact path is not exact; speculation or streaming changes answers | The claim comes out of the README; the lossy option is gated as a lossy one (T2) instead, or turned off in the shipped run lines |
| T1 at a margin | A wrong token at a position where the model is not near a tie: a kernel, a format conversion in the pack, a missing PLE shard, a wrong tensor, or a llama.cpp defect | Section 3.7 until one side is named; a divergence in llama.cpp is reported upstream with the ids; the README says nothing beyond "same output as upstream PolyStrata" until resolved (`PLAN.md:319`) |
| T2 above 1.5x the floor with T1 holding | Systematic rounding worse than the reference's own CPU-vs-GPU spread: an activation quantizer, a reduction order, or a lossy default still on | Find the arm that raised it; if no arm isolates it, the kernel-level parities at the exact shape |
| Reference GPU vs CPU floor itself far above upstream's figures (0.001-0.06 by set) | The reference build is not what upstream measured against (wrong commit, FA off, a type without a compiled kernel) | Fix the reference before judging the engines |

**Run order.** Each step is skipped by nothing before it failing, but its result is read in the light of it.

1. Configure with tests on; build the test targets and the pinned llama.cpp CUDA build; build the dump tool.
   Records which targets configure and build, against the reading in 2.1.
2. The component parities whose formats are in use: `iq_parity`, `iq_avx2_parity`, `q8k_quant_parity`,
   `native_grouped_parity`, `iq_multi_parity`, `mmvq_multi_parity`, `prefill_fused_iq_test`,
   `prefill_mmq_scatter_test`, `kv_q8_parity`, `kv_stream_parity`, `kv_steps_parity`, `kv_window_parity`,
   `gdn_parity`, `gdn_rec_parity`, `qsa_parity`, `qsa_topk_parity`, `verify_parity`, `spec_verify_parity`,
   `sampler_parity`, `native_multi_parity`, `rope_parity`, `elementwise_parity`; for B: `poly_expert_parity`
   on both files, `poly_route_test`, `poly_norm_q8_test`, `poly_gemm_small_test`, `poly_kv_q8_test`,
   `poly_attend_test` (recorded as not at Qwen3.6's shape), `poly_draft_race_test` (recorded as SKIP).
3. Reference runs for every sequence of 3.2, GPU and CPU, and the floor per set.
4. Configuration B, arms 1 to 4 (3.3).
5. Configuration A, arms 1 to 5 (3.4).
6. `strata-poly --check-batch 4` on the short set (batching, outside the two configurations but one command).
7. Write-up: per check a `result.md` with the criterion and the numbers; `docs/MODEL-FILES.md` gets a column.

Steps 4 and 5 are independent of each other and of step 2, but a step-2 failure at a format in use is the
first thing to look at when 4 or 5 fails.

## 5. Machine time and files per check

### 5.1 Time on the RTX 3090 pod

Estimates; nothing was timed. Engine start-up from `engine-info.txt`: `strata` about 2.5 minutes to ready
(the pinned 40 GB arena, the profile pre-fill), `strata-poly` 8 s. The reference's CPU run of Flash-Next reads
76 GB of shards through the page cache; its speed depends on the pod's RAM, which is not recorded in this tree.

| Check | Runs | Time each | Total |
| --- | --- | --- | --- |
| Builds (5.3) | 1 | llama.cpp CUDA build 10-20 min; fork test targets 10-20 min on top of the engine build; dump tool seconds | under 1 h |
| Component parities, A set (step 2) | 22 programs | seconds to a few minutes; `prefill_fused_iq_test` up to 10 min by its timeout | about 30 min |
| Component parities, B set | 7 programs, `poly_expert_parity` on 2 files | seconds; `poly_expert_parity` minutes per file | 15 min |
| Reference, Qwen3.6 | 18 sequences x 2 runs | GPU: prompt in one batch plus 128-256 single steps, under a minute; CPU: a few minutes for the short set, longer for the 16K and 31K prompts (n_batch = `STRATA_DUMP_CTX`) | 1-2 h, mostly the CPU runs of the long prompts |
| Reference, Flash-Next | 18 x 2 | CPU: minutes per short sequence, tens of minutes for the long prompts; partial-GPU: similar | 3-5 h |
| B arms 1-4 | 4 arms x 18 sequences (+ IQ4_XS: 2 x 18) | 8 s start + seconds per sequence | under 1 h |
| A arms 1-5 | 5 arms x 18 sequences | 2.5 min start per process + seconds to a minute per sequence | 4-5 h, dominated by start-up; a `--serve` process reused per arm with `--prompt-cache 0` cuts it to under 1 h if the first repeat shows byte-identical results |
| Scoring and write-up | | | 1 h |

Total: one pod-day, of which the engine time is small and the reference's CPU runs of Flash-Next are the
largest block. The two engines never run at the same time as a reference run on the same card.

### 5.2 Files per check, under `bench/unbiased/correctness/`

| Path | Content |
| --- | --- |
| `README.md` | the pod, driver, CUDA, commits of the fork and of llama.cpp, the builds' sha256 |
| `inputs/<set>/<seq>.prompt.ids`, `.answer.ids` | the id lists both sides get |
| `reference/<model>/<seq>.gpu.full`, `.cpu.full` | f32 `[step][vocab]` from the dump tool, 254 MB each for 256 steps at V = 248,320; kept on the pod, with their sha256 and a derived `.top20.tsv` (ids and log-probabilities) committed |
| `reference/<model>/floor.md` | KL and top-1, GPU vs CPU, per set |
| `parity/<program>/command.sh`, `stdout.txt`, `stderr.txt`, `result.md` | the component parities, one directory each, with the exit code and the criterion |
| `poly/<file>/<arm>/<seq>.top.txt`, `.ids.txt`, `command.sh` | 3.3's outputs |
| `strata/<arm>/<seq>.pos.tsv`, `command.sh`, `engine.log` | 3.4's outputs |
| `scores/<model>/<arm>/<set>.tsv`, `summary.md` | per position: llama.cpp top-2 gap, argmax equal, top-20 overlap, KL; per set the T1/T2 verdict |
| `findings/<n>.md` | one per divergence: first diverging position, the inputs, both sides' top 20, which side is wrong once known |

### 5.3 What must be built or written, and what is undecided

- **On the pod:** llama.cpp `3cf03257` with CUDA (the submodule checkout, `-DGGML_CUDA=ON`,
  `-DCMAKE_CUDA_ARCHITECTURES=86`), the dump tool against it, the fork's test targets
  (`CMAKE_EXTRA=-DSTRATA_BUILD_TESTS=ON`), python3 with numpy for `iq_parity`'s fixture.
- **New scripts (step 12, small):** a scorer reading `top.txt` / `pos.tsv` and the reference `.full` files
  (the metrics of `kl.py:85-101` and `docs/UNSLOTH_Q4.md:198-200`); a `GEN` sender for `strata --serve` if
  `tools/qwen35/q36_harness.py` needs more than its current arguments.
- **Possible code changes, each a step-12 decision, not assumed here:** (a) a dump-tool variant with tensor
  overrides so Flash-Next's dense layers run on the GPU while the experts stay on the CPU, and with the capture
  name `l_last-` and the `hc` dimension for `qwen4exp` (3.7); (b) an `Engine::dbg()` that writes per-layer
  vectors for `strata-poly`, needed only if a B divergence has to be localized by layer; (c) a `poly_attend_test`
  case at head dim 256 with 8 heads per group, if the attention is the suspect.
- **Settled by reading (2026-10-18), no run needed:** the `:850-1234` programs are defined without
  `STRATA_BUILD_TESTS` (2.1); `qwen4exp` names its layer outputs `l_last-N`, not `l_out-N` (3.7); the GSQ-RCO
  shard-1 header carries exactly the tensor set the pinned loader creates and no MTP tensor (3.1); the shard's
  per-layer expert types equal the checked-in layout `tests/data/native_experts/iq3_xxs.txt` for all 48 layers
  (1.2).
- **Not settleable from this Mac:** the UD-Q3_K_M file's per-tensor types (the file is on the pod only; the
  local `/Volumes/models/qwen3.6/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS-MTP.gguf` is a 1,067-byte pointer, and
  `docs/MODEL-FILES.md:14` should say so), the pod pack's `native_experts.txt` (one `diff` against the fixture
  on the pod), and op support at a real load of either file. Each is the first line of its check's `result.md`.
