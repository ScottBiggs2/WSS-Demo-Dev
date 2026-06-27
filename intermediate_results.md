# Intermediate Throughput Profiling Results:

(base) [biggs.s@explorer-02 WSS-Demo-Dev]$ python src/complex/experiments/collect_results.py
merged 5 rows -> /home/biggs.s/WSS-Demo-Dev/src/complex/experiments/outputs/perf/profile_all_cuda.csv
merged 5 rows -> /home/biggs.s/WSS-Demo-Dev/src/complex/experiments/outputs/perf/convergence_all_cuda.csv
merged 1 rows -> /home/biggs.s/WSS-Demo-Dev/src/complex/experiments/outputs/perf/parity_all_cuda.csv

## Throughput / phase attribution (cuda)

| config | params | it/s | vs dense | ms/step | fwd | div | bwd | retr | retr speedup | peak MB | ortho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100k-dense | 184,780 | 32.4 | 1.00x | 30.83 | 4.61 | 0.14 | 26.51 | 0.94 | - | 169 | 0.0e+00 |
| 100k-wss-qr | 116,236 | 23.8 | 0.74x | 41.94 | 6.53 | 1.77 | 15.57 | 18.28 | 2.7x | 310 | 6.0e-07 |
| 100k-wss-newton_schulz | 116,236 | 8.8 | 0.27x | 113.76 | 14.30 | 3.19 | 59.98 | 49.98 | 1.0x | 261 | 2.4e-07 |
| 100k-wss-none | 116,236 | 12.2 | 0.38x | 81.64 | 13.87 | 3.09 | 59.86 | 17.22 | 2.9x | 261 | 2.8e-05 |
| 100k-wss-ns_K4 | 116,236 | 11.2 | 0.35x | 88.95 | 13.61 | 3.05 | 59.70 | 25.24 | 2.0x | 261 | 2.4e-07 |

_`retr` = optimizer step = the Stiefel retraction. `retr speedup` is normalized to the slowest wss method here. `div` is the diversity/eigvalsh phase (CUDA-native on this branch). Phase sums slightly exceed ms/step (per-phase syncs)._

## fp32-vs-bf16 parity — trust gate (cuda)

| config | params | acc fp32 | acc bf16 | Δacc | ortho fp32 | ortho bf16 | it/s fp32 | it/s bf16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100k-wss-newton_schulz | 116,236 | 31.40% | 31.53% | +0.13% | 3.0e-07 | 2.4e-07 | 8.3 | 5.6 |

_GATE 0c: trust bf16 if |Δacc| is small (target ≤ 1.0%) and ortho stays the same order of magnitude. Otherwise run the scaling suite in fp32._

## Convergence — param-matched accuracy (cuda)

| config | params | dtype | final acc | ortho err | it/s | peak MB |
| --- | --- | --- | --- | --- | --- | --- |
| 100k-dense | 184,780 | bf16 | 48.68% | 0.0e+00 | 14.5 | 169 |
| 100k-dense_matched | 115,068 | bf16 | 48.25% | 0.0e+00 | 14.7 | 151 |
| 100k-single_rank_Jr-newton_schulz | 116,236 | bf16 | 31.35% | 1.2e-07 | 7.3 | 204 |
| 100k-wss-newton_schulz | 116,236 | bf16 | 31.53% | 2.4e-07 | 6.7 | 262 |
| 100k-wss-none | 116,236 | bf16 | 31.54% | 5.7e-04 | 8.6 | 262 |

## Scaling sweeps — depth↔width & J↔r (cuda)

_(no convergence CSVs found)_
