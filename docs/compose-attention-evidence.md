<!-- GENERATED on node 53 / GPU 1 (2026-11-12) by tools/compose_search.py check-decode16 / check-prefill (5 interleaved repetitions, median): the recorded decode16 and bf16 prefill split rules against the measured best -->

# composition search: paged decode (decode16) (measured top-k vs the recorded dispatch rule)

5 interleaved repetitions, top 5 by the coarse model + the rule's choice; estimator: median; tolerance 3%.

## Summary

| shape | recorded rule (measured) | measured best | rule vs best | verdict |
|---|---|---|--:|---|
| 4 8 1024 16 | `pa_decode16_bf16_hd128_p16_c` S2 11.6 us | `pa_decode16_bf16_hd128_p16_c` S2 11.6 us | +0.0% | confirmed |
| 4 8 2048 16 | `pa_decode16_bf16_hd128_p16_c` S4 15.4 us | `pa_decode16_bf16_hd128_p16_c` S4 15.4 us | +0.0% | confirmed |
| 4 8 8192 16 | `pa_decode16_bf16_hd128_p16_c` S8 28.3 us | `pa_decode16_bf16_hd128_p16_c` S8 28.3 us | +0.0% | confirmed |
| 8 8 1024 16 | `pa_decode16_bf16_hd128_p16_c` S2 11.7 us | `pa_decode16_bf16_hd128_p16_c` S2 11.7 us | +0.0% | confirmed |
| 8 8 2048 16 | `pa_decode16_bf16_hd128_p16_c` S4 19.6 us | `pa_decode16_bf16_hd128_p16_c` S4 19.6 us | +0.0% | confirmed |
| 8 8 8192 16 | `pa_decode16_bf16_hd128_p16_c` S4 41.0 us | `pa_decode16_bf16_hd128_p16_c` S4 41.0 us | +0.0% | confirmed |
| 16 8 1024 16 | `pa_decode16_bf16_hd128_p16_c` S2 15.7 us | `pa_decode16_bf16_hd128_p16_c` S2 15.7 us | +0.0% | confirmed |
| 16 8 2048 16 | `pa_decode16_bf16_hd128_p16_c` S2 24.2 us | `pa_decode16_bf16_hd128_p16_c` S2 24.2 us | +0.0% | confirmed |
| 16 8 8192 16 | `pa_decode16_bf16_hd128_p16_c` S2 93.9 us | `pa_decode16_bf16_hd128_p16_c` S4 91.9 us | +2.2% | confirmed |
| 32 8 1024 16 | `pa_decode16_bf16_hd128_p16_f` S1 20.2 us | `pa_decode16_bf16_hd128_p16_f` S1 20.2 us | +0.0% | confirmed |
| 32 8 2048 16 | `pa_decode16_bf16_hd128_p16_f` S1 35.5 us | `pa_decode16_bf16_hd128_p16_f` S1 35.5 us | +0.0% | confirmed |
| 32 8 8192 16 | `pa_decode16_bf16_hd128_p16_f` S1 164.9 us | `pa_decode16_bf16_hd128_p16_f` S1 164.9 us | +0.0% | confirmed |
| 64 8 1024 16 | `pa_decode16_bf16_hd128_p16_f` S1 33.3 us | `pa_decode16_bf16_hd128_p16_f` S1 33.3 us | +0.0% | confirmed |
| 64 8 2048 16 | `pa_decode16_bf16_hd128_p16_f` S1 84.2 us | `pa_decode16_bf16_hd128_p16_f` S1 84.2 us | +0.0% | confirmed |
| 64 8 8192 16 | `pa_decode16_bf16_hd128_p16_f` S1 294.6 us | `pa_decode16_bf16_hd128_p16_f` S1 294.6 us | +0.0% | confirmed |

**0 shape(s) where the recorded rule is not within 3% of the measured best.**

### decode16 4 8 1024 16: 3 candidates, 1 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_c` S2

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_c` | 2 | - | 64 | 5.7 | 11.59 (11.6-11.6) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 32 | 10.2 | 11.80 (11.8-11.9) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 128 | 3.8 | 12.00 (12.0-12.1) | yes |  |

### decode16 4 8 2048 16: 4 candidates, 1 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_c` S4

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 128 | 6.3 | 15.40 (15.2-15.6) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_c` | 8 | - | 256 | 5.0 | 18.24 (18.1-18.4) | yes |  |
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 32 | 20.5 | 21.00 (20.9-21.7) | yes |  |

### decode16 4 8 8192 16: 5 candidates, 2 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_c` S8

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_c` | 8 | - | 256 | 12.6 | 28.31 (28.2-28.6) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_c` | 16 | - | 512 | 15.0 | 38.22 (38.0-38.4) | yes |  |
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 32 | 81.9 | 72.26 (71.9-72.5) | yes |  |

### decode16 8 8 1024 16: 3 candidates, 1 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_c` S2

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_c` | 2 | - | 128 | 5.7 | 11.70 (11.6-11.8) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 64 | 10.2 | 11.86 (11.8-12.0) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 256 | 3.8 | 12.25 (12.2-12.4) | yes |  |

### decode16 8 8 2048 16: 4 candidates, 1 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_c` S4

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 256 | 6.3 | 19.59 (19.5-19.7) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_c` | 8 | - | 512 | 7.5 | 23.21 (23.0-23.2) | yes |  |
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 64 | 20.5 | 26.97 (26.9-27.1) | yes |  |

### decode16 8 8 8192 16: 5 candidates, 1 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_c` S4

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 256 | 21.7 | 40.98 (40.9-41.0) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_c` | 8 | - | 512 | 22.9 | 44.17 (44.1-44.2) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 16 | - | 1024 | 25.3 | 54.08 (53.8-54.3) | yes |  |
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 64 | 81.9 | 87.95 (87.7-88.3) | yes |  |

### decode16 16 8 1024 16: 3 candidates, 0 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_c` S2

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_c` | 2 | - | 256 | 5.7 | 15.71 (15.6-15.8) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 128 | 10.2 | 15.89 (15.8-16.0) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 512 | 6.3 | 18.52 (18.4-18.6) | yes |  |

### decode16 16 8 2048 16: 4 candidates, 0 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_c` S2

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_c` | 2 | - | 256 | 10.8 | 24.23 (24.1-24.3) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 512 | 11.4 | 26.24 (26.1-26.4) | yes |  |
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 128 | 20.5 | 28.31 (28.1-28.4) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 8 | - | 1024 | 12.6 | 32.65 (32.5-32.8) | yes |  |

### decode16 16 8 8192 16: 5 candidates, 0 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_c` S2

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 512 | 42.2 | 91.92 (91.4-92.7) | yes | **best** |
| `pa_decode16_bf16_hd128_p16_c` | 2 | - | 256 | 41.6 | 93.94 (93.7-94.4) | yes | rule |
| `pa_decode16_bf16_hd128_p16_c` | 8 | - | 1024 | 43.4 | 99.53 (99.0-100.7) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 16 | - | 2048 | 45.8 | 115.98 (115.8-133.4) | yes |  |
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 128 | 81.9 | 129.72 (129.5-134.9) | yes |  |

### decode16 32 8 1024 16: 3 candidates, 0 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_f` S1

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 256 | 10.2 | 20.20 (20.1-20.4) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_c` | 2 | - | 512 | 10.8 | 23.12 (23.0-23.2) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 1024 | 11.4 | 28.40 (28.3-28.5) | yes |  |

### decode16 32 8 2048 16: 4 candidates, 0 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_f` S1

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 256 | 20.5 | 35.50 (35.4-35.5) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_c` | 2 | - | 512 | 21.1 | 38.31 (38.2-38.5) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 1024 | 21.7 | 42.36 (42.0-42.6) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 8 | - | 2048 | 22.9 | 52.12 (52.0-52.6) | yes |  |

### decode16 32 8 8192 16: 5 candidates, 0 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_f` S1

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 256 | 81.9 | 164.90 (163.4-166.1) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_c` | 2 | - | 512 | 82.5 | 171.23 (170.9-172.5) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 1024 | 83.1 | 174.49 (174.1-174.8) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 8 | - | 2048 | 84.3 | 189.47 (189.2-189.9) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 16 | - | 4096 | 86.7 | 214.66 (214.2-215.4) | yes |  |

### decode16 64 8 1024 16: 3 candidates, 0 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_f` S1

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 512 | 20.5 | 33.26 (33.0-33.4) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_c` | 2 | - | 1024 | 21.1 | 38.66 (38.5-38.9) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 2048 | 21.7 | 49.15 (48.7-49.4) | yes |  |

### decode16 64 8 2048 16: 4 candidates, 0 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_f` S1

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 512 | 41.0 | 84.19 (84.0-84.5) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_c` | 2 | - | 1024 | 41.6 | 99.25 (99.1-109.6) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 2048 | 42.2 | 106.82 (106.4-108.3) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 8 | - | 4096 | 43.4 | 124.15 (123.3-124.4) | yes |  |

### decode16 64 8 8192 16: 5 candidates, 0 pruned by the model; rule = `pa_decode16_bf16_hd128_p16_f` S1

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `pa_decode16_bf16_hd128_p16_f` | 1 | - | 512 | 163.8 | 294.57 (292.4-303.7) | yes | **best** rule |
| `pa_decode16_bf16_hd128_p16_c` | 2 | - | 1024 | 164.4 | 299.10 (297.1-322.9) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 4 | - | 2048 | 165.0 | 328.69 (303.6-330.5) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 8 | - | 4096 | 166.2 | 352.99 (332.9-353.6) | yes |  |
| `pa_decode16_bf16_hd128_p16_c` | 16 | - | 8192 | 168.6 | 395.84 (372.2-397.1) | yes |  |

# composition search: bf16 hd128 prefill (measured top-k vs the recorded dispatch rule)

5 interleaved repetitions, top 5 by the coarse model + the rule's choice; estimator: median; tolerance 3%.

## Summary

| shape | recorded rule (measured) | measured best | rule vs best | verdict |
|---|---|---|--:|---|
| 1 8 4096 4096 0 | `fmha_fwd_hd128_bf16_v3_ks` S2 90.3 us | `fmha_fwd_hd128_bf16_v3_ks` S2 90.3 us | +0.0% | confirmed |
| 1 8 4096 4096 1 | `fmha_fwd_hd128_bf16_v3_causal_ks` S4 74.9 us | `fmha_fwd_hd128_bf16_v3_causal_ks` S4 74.9 us | +0.0% | confirmed |
| 1 8 2048 2048 0 | `fmha_fwd_hd128_bf16_v3_ks` S4 35.5 us | `fmha_fwd_hd128_bf16_v3_ks` S4 35.5 us | +0.0% | confirmed |
| 1 8 2048 2048 1 | `fmha_fwd_hd128_bf16_v3_causal_ks` S4 41.5 us | `fmha_fwd_hd128_bf16_v3_causal_ks` S4 41.5 us | +0.0% | confirmed |
| 2 8 1000 1300 0 | - | - | - | no measurement |
| 2 8 1000 1300 1 | `fmha_fwd_hd128_bf16_causal` S1 36.5 us | `fmha_fwd_hd128_bf16_causal` S1 36.5 us | +0.0% | confirmed |
| 1 32 8192 8192 1 | `fmha_fwd_hd128_bf16_v3_causal` S1 575.7 us | `fmha_fwd_hd128_bf16_v3_causal` S1 575.7 us | +0.0% | confirmed |
| 4 16 2048 2048 0 | `fmha_fwd_hd128_bf16_v3` S1 170.8 us | `fmha_fwd_hd128_bf16_v3` S1 170.8 us | +0.0% | confirmed |

**0 shape(s) where the recorded rule is not within 3% of the measured best.**

### prefill_bf16 1 8 4096 4096 0: 3 candidates, 1 pruned by the model; rule = `fmha_fwd_hd128_bf16_v3_ks` S2

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `fmha_fwd_hd128_bf16_v3_ks` | 2 | - | 256 | 20.0 | 90.34 (87.7-92.3) | yes | **best** rule |
| `fmha_fwd_hd128_bf16_v3_ks` | 4 | - | 512 | 24.0 | 105.20 (103.7-108.8) | yes |  |

### prefill_bf16 1 8 4096 4096 1: 5 candidates, 1 pruned by the model; rule = `fmha_fwd_hd128_bf16_v3_causal_ks` S4

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `fmha_fwd_hd128_bf16_v3_causal_ks` | 4 | - | 256 | 16.0 | 74.87 (73.0-76.8) | yes | **best** rule |
| `fmha_fwd_hd128_bf16_v3_causal_ks` | 2 | - | 128 | 20.0 | 84.94 (84.6-85.8) | yes |  |
| `fmha_fwd_hd128_bf16_v3_causal_q128` | 1 | - | 128 | 16.0 | 92.22 (91.9-92.6) | yes |  |
| `fmha_fwd_hd128_bf16_causal` | 1 | - | 256 | 16.0 | 95.68 (95.2-96.0) | yes |  |

### prefill_bf16 1 8 2048 2048 0: 3 candidates, 0 pruned by the model; rule = `fmha_fwd_hd128_bf16_v3_ks` S4

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `fmha_fwd_hd128_bf16_v3_ks` | 4 | - | 256 | 12.0 | 35.55 (35.3-36.0) | yes | **best** rule |
| `fmha_fwd_hd128_bf16_v3_ks` | 2 | - | 128 | 12.0 | 40.89 (40.8-41.2) | yes |  |
| `fmha_fwd_hd128_bf16_v3` | 1 | - | 64 | 16.0 | 55.45 (55.3-55.5) | yes |  |

### prefill_bf16 1 8 2048 2048 1: 5 candidates, 3 pruned by the model; rule = `fmha_fwd_hd128_bf16_v3_causal_ks` S4

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `fmha_fwd_hd128_bf16_v3_causal_ks` | 4 | - | 128 | 12.0 | 41.50 (41.4-41.7) | yes | **best** rule |
| `fmha_fwd_hd128_bf16_causal` | 1 | - | 128 | 8.0 | 48.70 (48.5-48.9) | yes |  |
| `fmha_fwd_hd128_bf16_v3_causal_q128` | 1 | - | 64 | 8.0 | 51.07 (50.9-51.1) | yes |  |

### prefill_bf16 2 8 1000 1300 0: 2 candidates, 0 pruned by the model; rule = `fmha_fwd_hd128_bf16_v3_ks` S4

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `fmha_fwd_hd128_bf16_v3_ks` | 2 | - | 128 | 9.5 | 34.19 (34.0-34.3) | yes | **best** |
| `fmha_fwd_hd128_bf16_v3` | 1 | - | 64 | 11.0 | 41.45 (41.4-41.6) | yes |  |

### prefill_bf16 2 8 1000 1300 1: 4 candidates, 2 pruned by the model; rule = `fmha_fwd_hd128_bf16_causal` S1

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `fmha_fwd_hd128_bf16_causal` | 1 | - | 128 | 5.5 | 36.52 (36.5-36.5) | yes | **best** rule |
| `fmha_fwd_hd128_bf16_v3_causal_q128` | 1 | - | 64 | 5.5 | 44.90 (44.9-44.9) | yes |  |

### prefill_bf16 1 32 8192 8192 1: 5 candidates, 0 pruned by the model; rule = `fmha_fwd_hd128_bf16_v3_causal` S1

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `fmha_fwd_hd128_bf16_v3_causal` | 1 | - | 512 | 128.0 | 575.72 (571.4-578.3) | yes | **best** rule |
| `fmha_fwd_hd128_bf16_v3_causal_q128` | 1 | - | 1024 | 128.0 | 645.18 (643.2-658.5) | yes |  |
| `fmha_fwd_hd128_bf16_v3_causal_ks` | 2 | - | 1024 | 132.0 | 691.78 (688.7-714.4) | yes |  |
| `fmha_fwd_hd128_bf16_v3_causal_ks` | 4 | - | 2048 | 136.0 | 802.45 (801.5-806.3) | yes |  |
| `fmha_fwd_hd128_bf16_causal` | 1 | - | 2048 | 256.0 | 848.26 (840.2-849.8) | yes |  |

### prefill_bf16 4 16 2048 2048 0: 3 candidates, 0 pruned by the model; rule = `fmha_fwd_hd128_bf16_v3` S1

| candidate | splits | tile | ctas | model us | measured us | feasible | |
|---|--:|---|--:|--:|--:|---|---|
| `fmha_fwd_hd128_bf16_v3` | 1 | - | 512 | 32.0 | 170.75 (170.0-173.0) | yes | **best** rule |
| `fmha_fwd_hd128_bf16_v3_ks` | 2 | - | 1024 | 36.0 | 213.08 (211.1-214.4) | yes |  |
| `fmha_fwd_hd128_bf16_v3_ks` | 4 | - | 2048 | 40.0 | 301.67 (300.7-302.0) | yes |  |

