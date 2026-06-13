# Trainable Mixture of Experts in CUDA

Build a trainable Mixture-of-Experts layer from scratch in CUDA, starting from
low-level matmul and activation kernels and culminating in a full forward,
backward, and training loop with load-balancing auxiliary loss. Includes sparse
routing, token dispatch, and end-to-end MoE training on the GPU.

## Key results

**1. Gradient correctness.** Every backward kernel is validated against a
reference: analytical gradients vs. finite-difference (and/or a PyTorch autograd
reference), reported as max relative error per kernel. The backward pass through
sparse top-k routing is the hard part of MoE — this figure shows it is correct.

*Figure:* analytical vs. numerical gradient scatter (y = x line), plus a
per-kernel max-relative-error bar chart.

**2. Load balancing / expert specialization.** Tokens-per-expert occupancy is
tracked over training, **with vs. without** the load-balancing auxiliary loss.
Without it, routing collapses onto a few experts; with it, the auxiliary term
(`aux_scale`) pushes the occupancy toward equipartition.

*Figure:* tokens-per-expert histogram (aux on vs. off) across training steps,
showing collapse vs. balanced utilization.

**3. Profiling & performance.** Forward+backward latency is profiled to show
where time actually goes in a sparse MoE — not just raw kernel speed.

- *Pipeline breakdown* (Nsight Systems): per-stage time split across
  router → dispatch (gather/scatter) → expert matmuls → combine. In sparse MoE
  the data movement and load imbalance often rival the expert FLOPs, since the
  per-expert matmuls are small and irregular.
- *Straggler / load-imbalance cost* (ties back to result 2): the busiest expert
  gates the whole batch, so balancing the router cuts wall-clock latency by
  removing stragglers — the auxiliary loss is both a quality regularizer and a
  performance optimization.
- *Roofline / kernel craft* (Nsight Compute): naive vs. tiled matmul vs. cuBLAS
  in GFLOP/s and % of peak, with achieved occupancy and memory throughput.

*Figures:* stacked-bar pipeline breakdown; step latency vs. occupancy skew;
matmul roofline (naive / tiled / cuBLAS).

## Layout

```
include/moe/            Public headers (declarations)
  common.cuh            CUDA_CHECK macro + shared helpers
  ops/                  Generic GPU primitives
    matmul.cuh          steps 001-004  matmul (naive / tiled / AtB / ABt)
    elementwise.cuh     steps 005-007  add_bias_row, reduce_rows_to_bias_grad, elementwise_add
    activations.cuh     steps 008-011  relu / gelu forward + backward
    softmax.cuh         steps 012-013  softmax_rows forward + backward
    topk.cuh            steps 014-016  topk_per_row, normalize_topk_gates (+ backward)
  router/
    router.cuh          steps 017-020  logits, softmax, top-k experts, gate-weight backward
  dispatch/
    dispatch.cuh        steps 021-025  count / offsets / assign slots / gather / scatter
  combine/
    combine.cuh         steps 026-028  combine expert outputs (+ backward to outputs/gates)
  expert/
    expert.cuh          steps 029-040  up/down projection forward + backward, activation
  loss/
    aux_loss.cuh        steps 041-044  dispatch fractions, mean router probs, load-balancing aux loss
    mse_loss.cuh        steps 045-046  mse loss forward + backward
  optim/
    optim.cuh           steps 047-048  zero_buffer, sgd_update_parameters
  model/
    moe.cuh             steps 049-052  moe_forward, moe_backward, training step + loop

src/                    Implementations, mirroring include/ layout
app/
  main.cu               End-to-end toy training runner (was main.c)
```

Each `.cu` under `src/` implements the kernels declared in the matching `.cuh`.
The 52 steps map one-to-one onto the functions listed above.

## Build

```
make            # builds ./moe
make clean
```
