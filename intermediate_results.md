# Intermediate Throughput Profiling Results:

(base) [biggs.s@explorer-01 WSS-Demo-Dev]$ python src/complex/experiments/collect_results.py
merged 5 rows -> /home/biggs.s/WSS-Demo-Dev/src/complex/experiments/outputs/perf/profile_all_cuda.csv
merged 5 rows -> /home/biggs.s/WSS-Demo-Dev/src/complex/experiments/outputs/perf/convergence_all_cuda.csv
merged 1 rows -> /home/biggs.s/WSS-Demo-Dev/src/complex/experiments/outputs/perf/parity_all_cuda.csv
merged 12 rows -> /home/biggs.s/WSS-Demo-Dev/src/complex/experiments/outputs/perf/lr_sweep_all_cuda.csv

## Throughput / phase attribution (cuda)

| config | params | it/s | vs dense | ms/step | fwd | div | bwd | retr | retr speedup | peak MB | ortho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100k-dense | 184,780 | 32.5 | 1.00x | 30.82 | 6.43 | 0.15 | 26.52 | 1.04 | - | 169 | 0.0e+00 |
| 100k-wss-qr | 116,236 | 8.8 | 0.27x | 114.03 | 19.10 | 4.19 | 59.90 | 42.59 | 1.2x | 261 | 4.8e-07 |
| 100k-wss-newton_schulz | 116,236 | 8.9 | 0.27x | 112.77 | 13.81 | 3.08 | 59.69 | 49.34 | 1.0x | 261 | 2.4e-07 |
| 100k-wss-none | 116,236 | 12.1 | 0.37x | 82.95 | 13.82 | 3.09 | 60.10 | 17.55 | 2.8x | 261 | 2.8e-05 |
| 100k-wss-ns_K4 | 116,236 | 11.0 | 0.34x | 90.95 | 13.75 | 3.09 | 61.47 | 25.58 | 1.9x | 261 | 2.4e-07 |

_`retr` = optimizer step = the Stiefel retraction. `retr speedup` is normalized to the slowest wss method here. `div` is the diversity/eigvalsh phase (CUDA-native on this branch). Phase sums slightly exceed ms/step (per-phase syncs)._

## LR calibration — best LR per method (Stage 0.5) (cuda)

| method | lr group | lr | final acc | ortho err | it/s | best |
| --- | --- | --- | --- | --- | --- | --- |
| dense_matched | lr_euclid | 1e-03 | 46.84% | 0.0e+00 | 15.2 | ★ |
| dense_matched | lr_euclid | 3e-03 | 45.77% | 0.0e+00 | 13.2 |  |
| dense_matched | lr_euclid | 1e-02 | 38.49% | 0.0e+00 | 14.8 |  |
| dense_matched | lr_euclid | 3e-02 | 2.46% | 0.0e+00 | 14.7 |  |
| single_rank_Jr | lr_riemann | 1e-03 | 29.06% | 1.2e-07 | 7.3 |  |
| single_rank_Jr | lr_riemann | 3e-03 | 31.63% | 1.8e-07 | 7.3 |  |
| single_rank_Jr | lr_riemann | 1e-02 | 34.71% | 1.8e-07 | 7.4 |  |
| single_rank_Jr | lr_riemann | 3e-02 | 37.13% | 2.4e-07 | 6.2 | ★ |
| wss | lr_riemann | 1e-03 | 29.75% | 2.4e-07 | 5.8 |  |
| wss | lr_riemann | 3e-03 | 32.48% | 2.4e-07 | 6.7 |  |
| wss | lr_riemann | 1e-02 | 36.37% | 2.4e-07 | 6.6 |  |
| wss | lr_riemann | 3e-02 | 36.77% | 3.6e-07 | 6.8 | ★ |

_Best LR per method: dense_matched: lr_euclid=1e-03 (46.84%) · single_rank_Jr: lr_riemann=3e-02 (37.13%) · wss: lr_riemann=3e-02 (36.77%). Pin each method to its own best LR for the Stage-1 headline (LR is per-geometry, not a shared axis); record the lr_riemann(wss) / lr_euclid(dense) ratio as a finding._

## fp32-vs-bf16 parity — trust gate (cuda)

| config | params | acc fp32 | acc bf16 | Δacc | ortho fp32 | ortho bf16 | it/s fp32 | it/s bf16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100k-wss-newton_schulz | 116,236 | 31.40% | 31.53% | +0.13% | 2.4e-07 | 2.4e-07 | 7.8 | 6.8 |

_GATE 0c: trust bf16 if |Δacc| is small (target ≤ 1.0%) and ortho stays the same order of magnitude. Otherwise run the scaling suite in fp32._

## Convergence — param-matched accuracy (cuda)

| config | params | dtype | final acc | ortho err | it/s | peak MB |
| --- | --- | --- | --- | --- | --- | --- |
| 100k-dense | 184,780 | bf16 | 48.68% | 0.0e+00 | 12.9 | 169 |
| 100k-dense_matched | 115,068 | bf16 | 48.25% | 0.0e+00 | 15.1 | 151 |
| 100k-single_rank_Jr-newton_schulz | 116,236 | bf16 | 31.35% | 1.2e-07 | 7.4 | 204 |
| 100k-wss-newton_schulz | 116,236 | bf16 | 31.53% | 2.4e-07 | 6.7 | 262 |
| 100k-wss-none | 116,236 | bf16 | 31.54% | 5.7e-04 | 8.4 | 262 |

## Scaling sweeps — depth↔width & J↔r (cuda)

_(no scaling CSVs found — run scaling.sbatch)_


# Strong early/local result: 

python -u src/complex/experiments/headline_vit.py --epochs 10 --lr_euclid 1e-2 --lr_riemann 1e-2 --dim 64 --depth 4 --J 4 --r 4 --euclidean --runs dense,single_rank_Jr,wss

WSS dominates - why? 

