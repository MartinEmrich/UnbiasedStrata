# The Q8_0 KV cache (`strata-poly --kv q8_0`)

`strata-poly` can store its attention K and V caches in the ggml **Q8_0** format instead of fp16. It is
opt-in: `--kv q8_0`. The default, `--kv fp16`, is unchanged and byte-identical to the build before this
option existed. The format halves the KV cache's VRAM and, on the models tested, leaves the answers alone.

## What the format is

A cache row holds one value a token for each KV head and each head dimension (`n_kv * head_dim` values).
Stored as fp16 that is `2 * n_kv * head_dim` bytes a token for K and, again, for V. Stored as Q8_0 it is cut
into 32-value blocks: for every block of 32 values, one fp16 **scale** (the block's largest magnitude over 127)
and 32 **int8 codes**. That is `n_kv * (head_dim / 32) * 2` bytes of scales plus `n_kv * head_dim` bytes of codes,
a row for K and the same for V. Attention dequantizes a block with its scale on the fly while it computes the
scores and the outputs.

For the three tested models (`n_kv = 2`, `head_dim = 256`, 11 attention layers for Tiel/Ornith, 10 for Occamy)
the engine prints its row size at load:

```
strata-poly: q8_0 KV cache: 544 bytes a K row, 544 a V row
```

544 = 512 codes + 2 blocks × 2 scales, a head, K and V. `--kv q8_0` needs `n_kv * head_dim` to be a multiple
of 32; the models here are. A grammar that is not gets a clear error rather than a wrong answer.

## Mixing the two: `--kv k16v8`

K and V are stored one at a time, and `--kv k16v8` uses both formats: the **keys stay fp16** and only the
**values become Q8_0**. The attention scores are therefore the same numbers `--kv fp16` computes, and only the
weighted sum of the values carries quantization error - the half of the cache whose error is averaged by the
attention weights rather than fed through the softmax.

For the same models (512 values a row), what a token costs an attention layer:

| `--kv` | a K row | a V row | a token, 11 layers | 131072 tokens |
|---|---|---|---|---|
| `fp16` | 1024 | 1024 | 22,528 B | 2.95 GB |
| `k16v8` | 1024 | 544 | 17,248 B | 2.26 GB |
| `q8_0` | 544 | 544 | 11,968 B | 1.57 GB |

What to expect from it, against the two it sits between:

- **Prompt: cheaper than `--kv q8_0`, level with `fp16`.** The batch's products take fp16 operands, so a Q8_0
  cache's rows are dequantized into a workspace before each block is read (`kernels.cu`'s `dequant_q8_kernel`).
  `k16v8` dequantizes only the values and reads its keys where they lie.
- **Decode: between the two, nearer `fp16`.** The score kernel reads the fp16 keys (twice the bytes of Q8_0)
  and the value kernel reads the Q8_0 values (half of fp16).
- **The win is VRAM the experts can have.** As under `--kv q8_0`, what the cache gives back goes to the expert
  cache; that, and not the attention kernels, is where the tokens per second move.

The load line says which pair is in use (`k16v8`, `q8_0`, `fp16`), and `poly_kv_q8_test` covers the mixed read:
`attend()` with fp16 keys and Q8_0 values against the same attention over the dequantized values as an fp16
cache, and `attend_batch()` the same way with the values' workspace half.

All three models loaded and answered with `--kv q8_0` at the full 131072-token context, and with a picture
through the vision encoder (`strata-vision`), each replying `KV OK` to the fixed prompt.

## What it was measured on

| | |
| --- | --- |
| Graphics card | NVIDIA RTX 3070, 8192 MiB |
| CPU | 24 threads |
| Built with | CUDA (the CUDA build of `strata-poly`); `/opt/cuda` |
| Models | `Tiel-Coder-35B-A3B.APEX-I-MiniPlus-V2.1.gguf`, `Ornith-1.5-35B-A3B.APEX-I-MiniPlus-V2.1.gguf`, `Occamy-1.0.APEX-I-MiniPlus-V2.1.gguf`, all `qwen35moe`, Q4_K - Medium |
| Context | 131072 tokens |
| Server | `serve/server.py --engine strata --config strata-<model>.json --port 8099` |

## The answers do not move

With greedy sampling (`temperature: 0`, which the engine reads as its own greedy), `--no-draft` so the
speculative path cannot change a token (`timings.draft_n` was 0 on every request), and `max_tokens: 64`, the
fp16 and Q8_0 arms were compared over five prompts each: the four short factual ones and one open-ended
sentence about the colour blue.

| model | exact matches | mean character agreement | longest common prefix |
| --- | ---: | ---: | --- |
| Tiel Coder 35B-A3B | 5/5 | 1.0000 | full reply (137 chars) |
| Ornith 1.5 35B-A3B | 5/5 | 1.0000 | full reply (129 chars) |
| Occamy 1.0 | 5/5 | 1.0000 | full reply (173 chars) |
| **all three** | **15/15** | **1.0000** | — |

Every Q8_0 answer equals the fp16 one character for character. The short factual prompts (`KV OK`, `42`,
`Paris`, `2, 3, 5, 7, 11`) also equal the pre-change fp16 baseline byte for byte on all three models
(12/12), which is how the default path's byte-identity was checked; the open-ended fifth prompt is not
bit-stable at greedy on Ornith and Occamy even before this change (speculative drafts on Ornith, `--tail-skip`
and the adaptive expert cache on Occamy), so it is not used for that check. The engine/API does not return
tokens, so agreement is measured on the reply text (longest common prefix and per-prompt agreement).

## What it saves

Per token, and in total at a full 131072-token context:

| model | attention layers | Q8_0 a token | fp16 a token | Q8_0 at 131072 | fp16 at 131072 | saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiel | 11 | 11 968 B | 22 528 B | 1.461 GiB | 2.750 GiB | 1.289 GiB |
| Ornith | 11 | 11 968 B | 22 528 B | 1.461 GiB | 2.750 GiB | 1.289 GiB |
| Occamy | 10 | 10 880 B | 20 480 B | 1.328 GiB | 2.500 GiB | 1.172 GiB |

**46.88% less KV VRAM** for all three, matching the per-token arithmetic exactly.

The engine's default KV cache grows as the conversation does, so a short prompt barely moves the load-time
expert cache — but with `--kv-whole`, which reserves the whole context's KV at load, the freed VRAM goes to
experts. At 131072 the expert cache under `--kv-whole` rose

```
Tiel   fp16: 1237 of 10496 experts (1.5 GiB)   q8_0: 2319 of 10496 (2.8 GiB)   +1082 experts
Ornith fp16: 1237 of 10496 experts (1.5 GiB)   q8_0: 2313 of 10496 (2.8 GiB)   +1076 experts
Occamy fp16: 1503 of 10240 experts (1.8 GiB)   q8_0: 2480 of 10240 (2.9 GiB)    +977 experts
```

with the same free VRAM left per model at the end.

## What it costs

At a 972-token prompt, 128-token answers, greedy, `--no-draft`, `--vram-reserve-mib 512` (see the caveat),
each arm a fresh process, median of four steps:

| model | arm | prompt tok/s | decode tok/s |
| --- | --- | ---: | ---: |
| Tiel | fp16 | 1014.6 | 105.8 |
| Tiel | q8_0 | 1008.0 | 105.2 |
| Ornith | fp16 | 1022.8 | 106.5 |
| Ornith | q8_0 | 1008.4 | 106.2 |
| Occamy | fp16 | 1031.8 | 107.1 |
| Occamy | q8_0 | 1034.1 | 106.9 |

Q8_0 against fp16: prompt **−1.4% to +0.2%**, decode **−0.6% to −0.2%** — within run-to-run scatter.

## A caveat that is not about Q8_0

At 131072 with the default 256 MiB reserve, long chats intermittently fail with
`cudaGraphInstantiate: out of memory`: the engine grows the expert cache down to 128 MiB free, then a newly
recorded CUDA graph needs more than what is left. This is **pre-existing and general**: it reproduces on the
unchanged fp16 path too, so it is not caused by `--kv q8_0`. `--vram-reserve-mib 512` removes it — all four
model/format combinations, eight requests each, with and without a picture, saw zero OOM. The numbers above
were taken with that 512 MiB reserve.

The three run configs (`strata-tiel.json`, `strata-ornith.json` and `strata-occamy.json`) now ship with
`--vram-reserve-mib 512`, so the OOM does not occur with them. With that reserve, Tiel at the full
131072-token context loads 3481 of 10496 routed experts into VRAM (4.1 GiB), measured on the RTX 3070 below.

## What was left alone

- The `strata` engine, and its own `--kv int8`. Nothing in it changed.
- `glm5next` / GLM 5.3 Flash (`dsa_attend`): its attention is not the one this cache quantizes, so `--kv q8_0`
  is not offered for it.
- **Gemma 4 is out of v1.** `--kv q8_0` with a `gemma4` file is rejected cleanly at load
  (`--kv q8_0 (and --kv k16v8) are only supported for qwen35moe models in v1; this is a 'gemma4' model`), not run.
- ROCm/HIP: the same kernels, measured below on an AMD card (it was rejected on HIP until that was done).
- The default path stays byte-identical to the last release.

## The comparison with another implementation

`llama.cpp` (the in-tree build `runners/llama.cpp/build/bin/llama-cli`) reads the same `qwen35moe` files and
accepts `-ctk q8_0 -ctv q8_0`. Given a Q8_0 KV cache it answered the two short fixed prompts
(`Reply with exactly: KV OK` → `KV OK`; `What is 17 + 25? Answer with just the number.` → `42`), the same
tokens the poly engine's `--kv q8_0` produced on all three models. It was run on the CPU (`-ngl 0`) because the
file is far larger than the card; only the short deterministic answers are a meaningful signal across engines.
Full transcript in the T8.5 report.

## Files touched

- `src/poly/engine.hpp` — `KvFmt` and `Options::kv_format_k` / `kv_format_v` (both `kFp16` by default; one
  format a cache half, which is what lets `--kv k16v8` mix them).
- `src/poly/main.cpp` — `--kv fp16|q8_0|k16v8` parsing and help text.
- `src/poly/engine.cpp` — the `qwen35moe`-only check and the clean rejection for other architectures.
- `src/poly/qwen35.hpp`, `src/poly/qwen35.cpp` — the Q8_0 cache pointers, allocation, save/load, the row-size
  print, and the attention path that reads codes+scales.  Each half is allocated, stored, saved and read on its
  own; `kv_runs()` is what keeps `card_row_bytes()` and the row movers in step.
- `src/poly/kernels.cu`, `include/strata/poly/kernels.hpp` — `store_q8` / `store_q8_at` and the Q8_0 attention
  readers (scores and outputs).  They took K's and V's formats independently from the start, so the mixed mode
  needed no kernel change.
- `src/poly/kv_q8_test.cpp` (new) — the kernel unit test: round-trip within half a block scale, byte-exact
  layout, the ring store, and the mixed (`k16v8`) read through `attend()` and `attend_batch()`.
- `CMakeLists.txt` — the `poly_kv_q8_test` target and its `add_test`.
- `strata-tiel.json`, `strata-ornith.json`, `strata-occamy.json` — the three run configs now carry
  `--kv q8_0` and `--vram-reserve-mib 512`.
- `gemma4.hpp` / `gemma4.cpp` — untouched.

## Builds

- **CUDA:** builds clean; the test-suite run is **116/117**. The one failure, `ple_parity`, is a missing model
  file (`.../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf: cannot open`), unrelated to this change.
  `poly_kv_q8_test` passes (0.23 s).
- **HIP:** builds clean for gfx1201; `poly_kv_q8_test` passes (91,530 values checked) on an RX 9070 XT.

## On an AMD card

RX 9070 XT (16 GB, gfx1201), Ryzen 7 9800X3D, 30 GB of RAM, Fedora, ROCm 7 from AMD's wheels;
`Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf`, `--kv-whole`, a 2 GiB conversation cache. fp16 at 131072 against Q8_0
at 196608, each a fresh server:

| | experts in VRAM | 256-token answer | 4.5K prompt | 64K prompt | answer after it |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16, 131072 | 4415 (7.9 GiB) | 110-127 tok/s | 1,814-1,822 tok/s | 2,065 tok/s | 66 tok/s |
| Q8_0, 196608 | 4729 (8.4 GiB) | 106-130 tok/s | 1,843-1,862 tok/s | 2,075 tok/s | 63 tok/s |

Greedy with `--no-draft`, seven prompts: the four short ones gave the same answers; a docstring, a two-sentence
summary of a 24K-character prompt and a list of names from a 60K-character one took other words from the 249th,
14th and first character on, each answer as right as the fp16 one.
- **SYCL:** not applicable; that is a separate engine.
