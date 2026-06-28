# TurboQuant — Near-Optimal KV Cache Compression

**Source:** Google Research, ICLR 2026. arXiv:2504.19874.
**Community:** vLLM-merged (May 2026), 5+ Apple Silicon MLX implementations.

Data-oblivious vector quantization compressing LLM KV caches to ~3 bits per coordinate with near-lossless quality. No training, no calibration data.

## Core Mechanism

Two-stage design:
- **PolarQuant** (AISTATS 2026) — Hadamard rotation Gaussianizes coordinates, Lloyd-Max scalar quantization on N(0,1) precomputed codebooks
- **QJL** (arXiv:2406.03482) — 1-bit residual correction removes inner-product bias in attention

## Apple Silicon (MLX) Ecosystem

| Implementation | Scope | Key Features |
|---|---|---|
| manjunathshiva/turboquant-mlx | KV cache | Mixed K8/V3, sink protection, 4× compression |
| chiuweilun1107/turboquant-mlx | KV cache | Fused Metal kernels, layer-adaptive (FP16 critical layers), 1-4 bit |
| arozanov/turboquant-mlx | KV cache + mlx-lm fork | V-only TQ, pre-rotated Q scoring, SIMD-group reductions, disk persistence |
| rachittshah/mlx-turboquant | KV cache | Standalone PoC, fractional bits, all bit widths 2-4 |
| ediestel/turboquant-plus-mlx | KV cache + extensions | Adaptive bit allocation, temporal decay, MoE-aware |
| matt-k-wong/turboquant-mlx-full | **Weights + KV cache** | Enables Qwen2.5-32B on 16GB MacBook Air |
| vllm-metal | KV cache (Metal kernel) | Paged attention + TurboQuant, MHA/hybrid models |

## Memory Impact on Apple Silicon

| Config | Compression | Quality Impact |
|---|---|---|
| K8/V3 (default) | ~2.6× | Near-lossless (K-MSE 0.00002) |
| K5/V3 | ~3.4× | Good |
| K4/V3 | ~3.8× | Matches paper config |
| K2/V3 | ~4.9× | Noticeable quality loss |

On M5 Pro 24GB with turboquant-mlx-full:
- **Qwen2.5-32B**: standard 4-bit MLX → ~18GB (too large). With TQ weights + 3-bit KV → ~13.5GB ✓
- **Qwen2.5-27B**: ~14GB marginal → ~11.4GB ✓
- **Qwen2.5-14B**: 9.5+0.8 → 8.2+0.3 = 8.5GB, saving ~1GB for context

Sink protection: first 128 tokens kept in FP16.

## Configuration Template

```yaml
context:
  kv_cache:
    compression: turboquant_4bit_nc
    k_quant: q8_0
    v_quant: q3_0
    min_tokens_before_quant: 128
```

Constraint: compresses KV cache during inference, not the agent's prompt context. Benefit is longer generation without OOM on long-horizon tasks.

## Relevance to Meredith

1. **local_model.yaml upgrade**: enables 27B-32B models on 24GB Apple Silicon
2. **Longer local context**: KV compression extends effective context from 32K→~96K with same memory budget
3. **Vector search compression**: same algorithm replaces Product Quantization in RAG — zero calibration data needed
