// src/poly/kv_q8_test.cpp - the Q8_0 KV cache kernels of the poly engine: store_q8 and store_q8_at against a host
// reference, and the read side of a MIXED cache (--kv k16v8: fp16 keys, Q8_0 values) against the same attention
// over an fp16 cache of the numbers the blocks stand for.  No model file and no card beyond CUDA device 0:
//   1. round trip: for several row counts and n in {32,128,256}, random floats of mixed magnitude are stored and
//      read back; every dequantized value must be within half a block scale (plus a hair) of the input;
//   2. byte layout: for n=256 the device codes and fp16 scales must match the host's own arithmetic byte for byte,
//      which pins codes[row*n + blk*32 + j] and scales[row*(n/32) + blk];
//   3. store_q8_at with a ring: two rows written from a device `at` land at (*at + r) % ring and equal what plain
//      store_q8 writes from the same position;
//   4. attend() with an fp16 key cache and a Q8_0 value cache (a null vcache and its Kvq8) against the same call
//      with the values as an fp16 cache of the dequantized blocks: within 5e-3 of the values' own size;
//   5. attend_batch() the same way, where the values' rows are dequantized into the workspace the engine hands out
//      for them (kwork null, vwork one half) and the products read the very halves the fp16 run reads: the same
//      numbers to the last bits.
// Exit 0 prints "poly_kv_q8_test: PASS"; any failed assertion prints the offending index and values and exits 1.
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "poly_kv_q8_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n ? n : 1), "cudaMalloc"); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() { return static_cast<T*>(p); }
};

uint16_t half_bits(__half h) {
    uint16_t b;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}

float half_value(uint16_t b) {
    __half h;
    std::memcpy(&h, &b, sizeof(h));
    return __half2float(h);
}

// A magnitude drawn over five orders (1e-4 .. 1e3) and either sign, so small and large values mix within a block.
float rand_val(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    const float mag = std::pow(10.f, -4.f + 7.f * u(rng));
    return (u(rng) < 0.5f ? -1.f : 1.f) * mag;
}

// 1. Round trip: dequantize with the fp16-rounded scale read back from `scales` and bound the error.
void round_trip(int rows, int n, std::mt19937& rng, int& fails, long& values) {
    const int nb = n / 32;
    std::vector<float> x((size_t) rows * n);
    for (float& v : x) v = rand_val(rng);

    Dev dx(x.size() * sizeof(float)), dcodes((size_t) rows * n), dscales((size_t) rows * nb * sizeof(uint16_t));
    ck(cudaMemcpy(dx.as<float>(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload x");
    strata::poly::store_q8(dx.as<float>(), dcodes.as<int8_t>(), dscales.as<uint16_t>(), n, 0, rows, nullptr);
    ck(cudaGetLastError(), "store_q8 launch");
    ck(cudaStreamSynchronize(nullptr), "store_q8 sync");

    std::vector<int8_t> codes((size_t) rows * n);
    std::vector<uint16_t> scales((size_t) rows * nb);
    ck(cudaMemcpy(codes.data(), dcodes.p, codes.size(), cudaMemcpyDeviceToHost), "download codes");
    ck(cudaMemcpy(scales.data(), dscales.p, scales.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "download scales");

    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const float d = half_value(scales[(size_t) r * nb + b]);
            const float half_scale = std::fabs(d) * 0.5f;
            for (int j = 0; j < 32; ++j) {
                const size_t i = (size_t) r * n + (size_t) b * 32 + j;
                const float got = (float) codes[i] * d;
                // A q8_0 value cannot land closer than half a block scale.  The scale is written fp16 from the
                // full-precision d the codes were rounded with (the kernel's id = 1/d), so decoding with the fp16
                // scale can miss by up to 127 * 2^-11 block scales at the rounding boundary: allow |x| * 5e-4 too.
                const float tol = half_scale + std::fabs(x[i]) * 5e-4f + 1e-3f;
                ++values;
                if (!(std::fabs(got - x[i]) <= tol)) {
                    std::fprintf(stderr,
                                 "poly_kv_q8_test: round-trip: n=%d rows=%d value [%d,%d,%d] x=%.9g got=%.9g "
                                 "d=%.9g err=%.9g tol=%.9g\n",
                                 n, rows, r, b, j, (double) x[i], (double) got, (double) d,
                                 (double) std::fabs(got - x[i]), (double) tol);
                    ++fails;
                }
            }
        }
    }
}

// 2. Byte layout: the host's own d, codes and fp16 scale must equal the device's bytes.
void byte_layout(int rows, int n, std::mt19937& rng, int& fails, long& values) {
    const int nb = n / 32;
    std::vector<float> x((size_t) rows * n);
    for (float& v : x) v = rand_val(rng);

    std::vector<int8_t> ecodes((size_t) rows * n);
    std::vector<uint16_t> escales((size_t) rows * nb);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const float* xr = x.data() + (size_t) r * n + (size_t) b * 32;
            float amax = 0.f;
            for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(xr[j]));
            const float d = amax / 127.0f;
            const float id = d > 0.f ? 1.0f / d : 0.0f;
            escales[(size_t) r * nb + b] = half_bits(__float2half(d));
            int8_t* cr = ecodes.data() + (size_t) r * n + (size_t) b * 32;
            for (int j = 0; j < 32; ++j) {
                float q = std::rint(xr[j] * id);
                q = std::fmin(std::fmax(q, -127.0f), 127.0f);
                cr[j] = (int8_t) q;
            }
        }
    }

    Dev dx(x.size() * sizeof(float)), dcodes((size_t) rows * n), dscales((size_t) rows * nb * sizeof(uint16_t));
    ck(cudaMemcpy(dx.as<float>(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload x");
    strata::poly::store_q8(dx.as<float>(), dcodes.as<int8_t>(), dscales.as<uint16_t>(), n, 0, rows, nullptr);
    ck(cudaGetLastError(), "store_q8 launch");
    ck(cudaStreamSynchronize(nullptr), "store_q8 sync");

    std::vector<int8_t> codes((size_t) rows * n);
    std::vector<uint16_t> scales((size_t) rows * nb);
    ck(cudaMemcpy(codes.data(), dcodes.p, codes.size(), cudaMemcpyDeviceToHost), "download codes");
    ck(cudaMemcpy(scales.data(), dscales.p, scales.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "download scales");

    ++values;
    if (std::memcmp(ecodes.data(), codes.data(), codes.size()) != 0) {
        size_t i = 0;
        while (i < codes.size() && ecodes[i] == codes[i]) ++i;
        const int r = (int) (i / n), rem = (int) (i % n);
        std::fprintf(stderr,
                     "poly_kv_q8_test: byte layout: codes differ at row=%d blk=%d j=%d: host=%d device=%d (n=%d "
                     "rows=%d)\n",
                     r, rem / 32, rem % 32, (int) ecodes[i], (int) codes[i], n, rows);
        ++fails;
    }
    ++values;
    if (std::memcmp(escales.data(), scales.data(), scales.size() * sizeof(uint16_t)) != 0) {
        size_t i = 0;
        while (i < scales.size() && escales[i] == scales[i]) ++i;
        const int r = (int) (i / nb), b = (int) (i % nb);
        std::fprintf(stderr,
                     "poly_kv_q8_test: byte layout: scales differ at row=%d blk=%d: host=0x%04x device=0x%04x "
                     "(n=%d rows=%d)\n",
                     r, b, (unsigned) escales[i], (unsigned) scales[i], n, rows);
        ++fails;
    }
}

// 3. store_q8_at with a ring: rows land at (*at + r) % ring and match plain store_q8 from the same position.
void ring_test(std::mt19937& rng, int& fails, long& values) {
    const int n = 128, nb = n / 32, rows = 2, ring = 5;
    const long long at_val = 4;   // rows go to 4 and 0: the second wraps

    std::vector<float> x((size_t) rows * n);
    for (float& v : x) v = rand_val(rng);

    Dev dx(x.size() * sizeof(float)), d_at(sizeof(long long));
    ck(cudaMemcpy(dx.as<float>(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload x");
    ck(cudaMemcpy(d_at.p, &at_val, sizeof(at_val), cudaMemcpyHostToDevice), "upload at");

    Dev rcodes((size_t) ring * n), rscales((size_t) ring * nb * sizeof(uint16_t));
    strata::poly::store_q8_at(dx.as<float>(), rcodes.as<int8_t>(), rscales.as<uint16_t>(), n, d_at.as<long long>(),
                              rows, ring, nullptr);
    ck(cudaGetLastError(), "store_q8_at launch");
    ck(cudaStreamSynchronize(nullptr), "store_q8_at sync");

    Dev pcodes((size_t) (at_val + rows) * n), pscales((size_t) (at_val + rows) * nb * sizeof(uint16_t));
    strata::poly::store_q8(dx.as<float>(), pcodes.as<int8_t>(), pscales.as<uint16_t>(), n, at_val, rows, nullptr);
    ck(cudaGetLastError(), "store_q8 launch");
    ck(cudaStreamSynchronize(nullptr), "store_q8 sync");

    std::vector<int8_t> rcode((size_t) ring * n), pcode((size_t) (at_val + rows) * n);
    std::vector<uint16_t> rscale((size_t) ring * nb), pscale((size_t) (at_val + rows) * nb);
    ck(cudaMemcpy(rcode.data(), rcodes.p, rcode.size(), cudaMemcpyDeviceToHost), "download ring codes");
    ck(cudaMemcpy(rscale.data(), rscales.p, rscale.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "download ring scales");
    ck(cudaMemcpy(pcode.data(), pcodes.p, pcode.size(), cudaMemcpyDeviceToHost), "download plain codes");
    ck(cudaMemcpy(pscale.data(), pscales.p, pscale.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "download plain scales");

    for (int r = 0; r < rows; ++r) {
        const size_t rr = (size_t) ((at_val + r) % ring);
        const size_t pr = (size_t) (at_val + r);
        ++values;
        if (std::memcmp(rcode.data() + rr * n, pcode.data() + pr * n, (size_t) n) != 0) {
            size_t i = 0;
            while (i < (size_t) n && rcode[rr * n + i] == pcode[pr * n + i]) ++i;
            std::fprintf(stderr,
                         "poly_kv_q8_test: ring: row %d at %zu != plain row %zu: code [%zu] ring=%d plain=%d\n",
                         r, rr, pr, i, (int) rcode[rr * n + i], (int) pcode[pr * n + i]);
            ++fails;
        }
        ++values;
        if (std::memcmp(rscale.data() + rr * nb, pscale.data() + pr * nb, (size_t) nb * sizeof(uint16_t)) != 0) {
            size_t i = 0;
            while (i < (size_t) nb && rscale[rr * nb + i] == pscale[pr * nb + i]) ++i;
            std::fprintf(stderr,
                         "poly_kv_q8_test: ring: row %d at %zu != plain row %zu: scale [%zu] ring=0x%04x "
                         "plain=0x%04x\n",
                         r, rr, pr, i, (unsigned) rscale[rr * nb + i], (unsigned) pscale[pr * nb + i]);
            ++fails;
        }
    }
}

// The Q8_0 codes and scales the device wrote, dequantized the way the read kernels do and rounded to halves: an
// fp16 cache holding exactly the numbers a Q8_0 cache stands for.  `codes`/`scales` are the device's own bytes,
// so the comparison that follows is about the plumbing and not about the quantization.
std::vector<uint16_t> dequantized(const std::vector<int8_t>& codes, const std::vector<uint16_t>& scales, int rows,
                                  int n) {
    const int nb = n / 32;
    std::vector<uint16_t> v((size_t) rows * n);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            const float d = half_value(scales[(size_t) r * nb + b]);
            for (int j = 0; j < 32; ++j) {
                const size_t i = (size_t) r * n + (size_t) b * 32 + j;
                v[i] = half_bits(__float2half((float) codes[i] * d));
            }
        }
    return v;
}

// A magnitude uniform in [lo, hi] and either sign: narrower than rand_val, so that a sum of them cannot cancel
// away the difference the test is looking at.
float rand_between(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    const float m = u(rng);
    return (u(rng) < 0.5f ? -1.f : 1.f) * m;
}

// 4. attend() over a MIXED cache: fp16 keys and Q8_0 values (--kv k16v8).  The same attention with the values
//    uploaded as an fp16 cache of the very numbers the blocks hold must come out the same within a hair: the two
//    runs share their scores and their weights, and differ only in the value kernel reading fp32-dequantized codes
//    where the other reads their halves.  A mis-wired half (the keys' scales for the values, a null cache read as
//    fp16, the wrong row stride) is off by the values themselves, orders above the bound.
void mixed_decode(int& fails, long& values, std::mt19937& rng) {
    const int dim = 256, n_head = 8, kv_heads = 1, keys = 37, rows = 1, chunk = 1;
    const long long pos = keys - rows;
    const float scale = 1.f / std::sqrt((float) dim);
    const int n = kv_heads * dim, nb = n / 32;

    std::vector<float> q((size_t) rows * n_head * dim), k((size_t) keys * n), v((size_t) keys * n);
    for (float& x : q) x = rand_between(rng, -1.f, 1.f);
    for (float& x : k) x = rand_between(rng, -1.f, 1.f);
    for (float& x : v) x = rand_between(rng, 0.5f, 1.5f);   // positive: the output cannot cancel to nothing

    std::vector<uint16_t> k16((size_t) keys * n);
    for (size_t i = 0; i < k.size(); ++i) k16[i] = half_bits(__float2half(k[i]));

    Dev dq(q.size() * sizeof(float)), dk(k16.size() * sizeof(uint16_t)), dvf(v.size() * sizeof(float));
    Dev dv((size_t) keys * n * sizeof(uint16_t));
    Dev dcodes((size_t) keys * n), dscales((size_t) keys * nb * sizeof(uint16_t));
    Dev d_out_a((size_t) rows * n_head * dim * sizeof(float)), d_out_b((size_t) rows * n_head * dim * sizeof(float));
    Dev d_score((size_t) 1 << 20);
    ck(cudaMemcpy(dq.as<float>(), q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "upload q");
    ck(cudaMemcpy(dk.as<uint16_t>(), k16.data(), k16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "upload k");
    ck(cudaMemcpy(dvf.as<float>(), v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "upload v");

    // the values as Q8_0 blocks, and the same numbers as an fp16 cache
    strata::poly::store_q8(dvf.as<float>(), dcodes.as<int8_t>(), dscales.as<uint16_t>(), n, 0, keys, nullptr);
    ck(cudaGetLastError(), "mixed: store_q8 launch");
    ck(cudaStreamSynchronize(nullptr), "mixed: store_q8 sync");
    std::vector<int8_t> codes((size_t) keys * n);
    std::vector<uint16_t> scales((size_t) keys * nb);
    ck(cudaMemcpy(codes.data(), dcodes.p, codes.size(), cudaMemcpyDeviceToHost), "download mixed codes");
    ck(cudaMemcpy(scales.data(), dscales.p, scales.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "download mixed scales");
    const std::vector<uint16_t> v16 = dequantized(codes, scales, keys, n);
    ck(cudaMemcpy(dv.as<uint16_t>(), v16.data(), v16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
       "upload dequantized v");

    using strata::poly::Kvq8;
    strata::poly::attend(dq.as<float>(), dk.as<uint16_t>(), dv.as<uint16_t>(), pos, rows, d_score.as<float>(), chunk,
                         d_out_a.as<float>(), dim, n_head, kv_heads, scale, nullptr);
    ck(cudaGetLastError(), "mixed: attend fp16 launch");
    strata::poly::attend(dq.as<float>(), dk.as<uint16_t>(), nullptr, pos, rows, d_score.as<float>(), chunk,
                         d_out_b.as<float>(), dim, n_head, kv_heads, scale, nullptr, 0, 0, false, nullptr, 0, 0,
                         Kvq8{}, Kvq8{dcodes.as<int8_t>(), dscales.as<uint16_t>()});
    ck(cudaGetLastError(), "mixed: attend k16v8 launch");
    ck(cudaStreamSynchronize(nullptr), "mixed decode sync");

    std::vector<float> a((size_t) rows * n_head * dim), b((size_t) rows * n_head * dim);
    ck(cudaMemcpy(a.data(), d_out_a.p, a.size() * sizeof(float), cudaMemcpyDeviceToHost), "download fp16 out");
    ck(cudaMemcpy(b.data(), d_out_b.p, b.size() * sizeof(float), cudaMemcpyDeviceToHost), "download mixed out");

    float top = 0.f;
    for (const float x : a) top = std::fmax(top, std::fabs(x));
    const float tol = 5e-3f * top + 1e-4f;
    for (size_t i = 0; i < a.size(); ++i) {
        ++values;
        if (!(std::fabs(a[i] - b[i]) <= tol)) {
            std::fprintf(stderr,
                         "poly_kv_q8_test: mixed decode: head value %zu (dim %d, keys %d): fp16=%.9g k16v8=%.9g "
                         "err=%.9g tol=%.9g\n",
                         i, dim, keys, (double) a[i], (double) b[i], (double) std::fabs(a[i] - b[i]), (double) tol);
            ++fails;
        }
    }
}

// 5. attend_batch() over a MIXED cache: the fp16 keys go to the products where they lie and only the values are
//    dequantized, into the workspace's half the engine hands out for them (kwork null, vwork the one half).  The
//    workspace rows are the same halves an fp16 value cache would hold, so the two runs put the same numbers
//    through the same products and come out the same to the last bits.
void mixed_batch(int& fails, long& values, std::mt19937& rng) {
    const int dim = 256, n_head = 8, kv_heads = 1, keys = 12, rows = 4;
    const long long pos = keys - rows;   // the last token attends the rows 0 .. keys - 1
    const float scale = 1.f / std::sqrt((float) dim);
    const int n = kv_heads * dim, nb = n / 32;
    const size_t score_floats = (size_t) 1 << 20;
    const size_t work_halves = (size_t) 2048 * n;

    std::vector<float> q((size_t) rows * n_head * dim), k((size_t) keys * n), v((size_t) keys * n);
    for (float& x : q) x = rand_between(rng, -1.f, 1.f);
    for (float& x : k) x = rand_between(rng, -1.f, 1.f);
    for (float& x : v) x = rand_between(rng, 0.5f, 1.5f);

    std::vector<uint16_t> k16((size_t) keys * n);
    for (size_t i = 0; i < k.size(); ++i) k16[i] = half_bits(__float2half(k[i]));

    Dev dq(q.size() * sizeof(float)), dk(k16.size() * sizeof(uint16_t)), dvf(v.size() * sizeof(float));
    Dev dv((size_t) keys * n * sizeof(uint16_t));
    Dev dcodes((size_t) keys * n), dscales((size_t) keys * nb * sizeof(uint16_t));
    Dev d_score(score_floats * sizeof(float));
    Dev d_work(((size_t) rows * n_head * (dim + dim / 2 + 3) + 1) * sizeof(float));
    Dev d_out_a((size_t) rows * n_head * dim * sizeof(float)), d_out_b((size_t) rows * n_head * dim * sizeof(float));
    Dev d_vwork(work_halves * sizeof(uint16_t));
    ck(cudaMemcpy(dq.as<float>(), q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "upload batch q");
    ck(cudaMemcpy(dk.as<uint16_t>(), k16.data(), k16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
       "upload batch k");
    ck(cudaMemcpy(dvf.as<float>(), v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "upload batch v");

    strata::poly::store_q8(dvf.as<float>(), dcodes.as<int8_t>(), dscales.as<uint16_t>(), n, 0, keys, nullptr);
    ck(cudaGetLastError(), "mixed batch: store_q8 launch");
    ck(cudaStreamSynchronize(nullptr), "mixed batch: store_q8 sync");
    std::vector<int8_t> codes((size_t) keys * n);
    std::vector<uint16_t> scales((size_t) keys * nb);
    ck(cudaMemcpy(codes.data(), dcodes.p, codes.size(), cudaMemcpyDeviceToHost), "download batch codes");
    ck(cudaMemcpy(scales.data(), dscales.p, scales.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "download batch scales");
    const std::vector<uint16_t> v16 = dequantized(codes, scales, keys, n);
    ck(cudaMemcpy(dv.as<uint16_t>(), v16.data(), v16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
       "upload dequantized batch v");

    void* blas = strata::poly::blas_create(nullptr);
    using strata::poly::Kvq8;
    strata::poly::attend_batch(blas, dq.as<float>(), dk.as<uint16_t>(), dv.as<uint16_t>(), pos, rows,
                               d_score.as<float>(), score_floats, d_work.as<float>(), d_out_a.as<float>(), dim,
                               n_head, kv_heads, scale, nullptr);
    ck(cudaGetLastError(), "mixed batch: attend_batch fp16 launch");
    strata::poly::attend_batch(blas, dq.as<float>(), dk.as<uint16_t>(), nullptr, pos, rows, d_score.as<float>(),
                               score_floats, d_work.as<float>(), d_out_b.as<float>(), dim, n_head, kv_heads, scale,
                               nullptr, 0, 0, false, Kvq8{}, Kvq8{dcodes.as<int8_t>(), dscales.as<uint16_t>()},
                               nullptr, d_vwork.as<uint16_t>(), work_halves);
    ck(cudaGetLastError(), "mixed batch: attend_batch k16v8 launch");
    ck(cudaStreamSynchronize(nullptr), "mixed batch sync");
    strata::poly::blas_destroy(blas);

    std::vector<float> a((size_t) rows * n_head * dim), b((size_t) rows * n_head * dim);
    ck(cudaMemcpy(a.data(), d_out_a.p, a.size() * sizeof(float), cudaMemcpyDeviceToHost), "download batch fp16 out");
    ck(cudaMemcpy(b.data(), d_out_b.p, b.size() * sizeof(float), cudaMemcpyDeviceToHost), "download batch mixed out");

    float top = 0.f;
    for (const float x : a) top = std::fmax(top, std::fabs(x));
    const float tol = 1e-5f * top + 1e-6f;   // the same halves into the same products
    for (size_t i = 0; i < a.size(); ++i) {
        ++values;
        if (!(std::fabs(a[i] - b[i]) <= tol)) {
            std::fprintf(stderr,
                         "poly_kv_q8_test: mixed batch: head value %zu (dim %d, rows %d): fp16=%.9g k16v8=%.9g "
                         "(the products read the same halves, so these should be the same bits)\n",
                         i, dim, rows, (double) a[i], (double) b[i]);
            ++fails;
        }
    }
}

}  // namespace

int main() {
    ck(cudaSetDevice(0), "cudaSetDevice");

    std::mt19937 rng(0x51u);
    int fails = 0;
    long values = 0;

    const int ns[3] = {32, 128, 256};
    const int rss[3] = {1, 3, 7};
    // many trials: 20 sweeps of (1+3+7) rows over three n = 660 random rows
    for (int t = 0; t < 20; ++t)
        for (int ri = 0; ri < 3; ++ri)
            for (int ci = 0; ci < 3; ++ci) round_trip(rss[ri], ns[ci], rng, fails, values);

    byte_layout(2, 256, rng, fails, values);
    byte_layout(3, 256, rng, fails, values);
    byte_layout(5, 256, rng, fails, values);

    ring_test(rng, fails, values);

    mixed_decode(fails, values, rng);
    mixed_decode(fails, values, rng);
    mixed_batch(fails, values, rng);
    mixed_batch(fails, values, rng);

    if (fails) {
        std::fprintf(stderr, "poly_kv_q8_test: FAIL: %d failed assertion(s)\n", fails);
        return 1;
    }
    std::printf("poly_kv_q8_test: PASS (%ld values checked)\n", values);
    return 0;
}
