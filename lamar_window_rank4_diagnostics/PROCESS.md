# Process record

## 1. Scope and safety

The diagnostics were isolated from the original immutable dataset and model
run. All new artifacts were written to separate output directories. Before
formal computation, the workflow inspected the data schema, model
configuration, train/dev membership, negative pool, sequence-length
invariants, software environment, GPU state, and active processes.

Calibration and test paths were explicitly forbidden. Neither diagnostic used
those splits.

## 2. Window-ablation design

The original sequence context was 101 nt with the biological center at
zero-based index 50. Shorter inputs were produced in memory:

```python
flank = (window - 1) // 2
cropped = sequence[50 - flank : 51 + flank]
```

Every crop was checked for the requested length and a central C. Original
tables were not rewritten.

The source `s1_frozen_center` configuration was used as the reference:

- mode: Frozen + Head;
- pooling: center;
- seed: 42;
- random negatives, 10 per positive per epoch;
- batch size 16 and gradient accumulation 2;
- BCE loss;
- head learning rate 1e-4;
- head dropout 0.1;
- weight decay 0.01;
- linear schedule with warmup ratio 0.03;
- 20 epochs = 7,080 optimizer steps.

Early stopping was not allowed to vary the compute budget. Each window ran to
the same fixed step count. The complete dev universe was scored once after the
final step.

The preflight generated a hash for the exact negative IDs selected in each
epoch. The final summary required the 20-hash sequence to be identical across
all four windows.

## 3. Resource gate and execution

Four CPU one-step smoke runs first validated token length, center-token
position, Frozen + Head forward/backward propagation, checkpoint writing, and
prediction writing.

Formal runs were serialized on one GPU. A resource gate required at least
9,000 MiB free memory and at most 10% utilization for three consecutive
30-second checks before the first run. Existing GPU tasks were allowed to
finish naturally and were not killed or preempted.

Execution order was 21, 41, 61, then 101 bp. The summary was created only
after all four formal summaries reported success.

## 4. Rank-4 spectrum audit

The existing `final_seed42` checkpoint contained 48 paired LoRA tensors:

- `A`: 4 x 768;
- `B`: 768 x 4;
- Q, K, V, and attention-output projections in each of 12 layers.

For every module:

```text
delta_W = (alpha / r) * B @ A
alpha = 8
r = 4
alpha / r = 2
```

The four nonzero singular values were computed efficiently without a dense
768 x 768 SVD. Thin QR decompositions give:

```text
B = Q_B R_B
A^T = Q_A R_A

nonzero_singular_values(delta_W)
  = singular_values((alpha / r) * R_B @ R_A^T)
```

The resulting core matrix is only 4 x 4 and has exactly the same nonzero
singular values.

Reported rank measures:

- numerical rank: number of singular values above
  `sigma_1 * max(m,n) * machine_epsilon`;
- entropy effective rank:
  `exp(-sum(p_i * log(p_i)))`, with `p_i = sigma_i / sum(sigma_i)`;
- stable rank: `sum(sigma_i^2) / sigma_1^2`.

Numerical rank answers whether a direction is distinguishable from numerical
zero. Effective and stable rank describe whether spectral mass is spread
across all four directions.

## 5. Interpretation boundary

The window result is a controlled single-seed Frozen + Head screen. It selects
61 bp for the next experiment but does not establish statistical
significance.

The spectrum audit is diagnostic, not a performance ablation. It shows that
all rank-4 directions are nonzero while most Q/V/O updates remain strongly
spectrally concentrated. It cannot by itself prove that rank 2 would preserve
predictive performance.
