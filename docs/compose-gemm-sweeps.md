<!-- GENERATED on node 53 / GPU 1 (2026-11-12) by tools/compose_search.py check-gemm (7 interleaved repetitions, median) and tools/perf_ab.py --families gemm: the sweeps behind the 256-tile fill threshold and the 64-row split model in dispatch.rs -->

# GEMM rule sweeps and gate

## Fill sweep: best time per tile (us), the measured winner and the previous rule

```
===== bf16
shape          ct256  fill |     t256     t128      t64 | winner                     | rule
1024x4096x4096    64  0.25 |        -     48.3        - | 16w16_bf16_128 S1          | 16w16_bf16_128 S1  maxspread 1.02
1280x4096x4096    80  0.31 |     72.5     57.6     62.5 | 16w16_bf16_128 S1          | 16w16_bf16_128 S1  maxspread 1.03
1536x4096x4096    96  0.38 |     73.8     62.0     68.6 | 16w16_bf16_128 S1          | 16w16_bf16_128 S1  maxspread 1.06
1792x4096x4096   112  0.44 |     75.7     66.9        - | 16w16_bf16_128 S1          | 16w16_bf16_128 S1  maxspread 1.05
2048x4096x4096   128  0.50 |     80.1     74.3        - | 16w16_bf16_128 S1          | 16w16_bf16_128 S1  maxspread 1.04
2304x4096x4096   144  0.56 |     83.7     98.4    103.4 | 16w16_bf16 S1              | 16w16_bf16_128 S1  maxspread 1.02
2560x4096x4096   160  0.62 |     86.8    100.6    108.5 | 16w16_bf16 S1              | 16w16_bf16_128 S1  maxspread 1.03
3072x4096x4096   192  0.75 |     94.8    112.1        - | 16w16_bf16 S1              | 16w16_bf16_128 S1  maxspread 1.02
3584x4096x4096   224  0.88 |    102.9    126.8    146.1 | 16w16_bf16 S1              | 16w16_bf16_128 S1  maxspread 1.02
4096x4096x4096   256  1.00 |    107.3    135.0        - | 16w16_bf16 S1              | 16w16_bf16 S1  maxspread 1.04
1024x7168x4096   112  0.44 |     77.1     66.7        - | 16w16_bf16_128 S1          | 16w16_bf16_128 S1  maxspread 1.03
1280x7168x4096   140  0.55 |     83.6     96.7    102.7 | 16w16_bf16 S1              | 16w16_bf16_128 S1  maxspread 1.03
1536x7168x4096   168  0.66 |     88.7    102.4        - | 16w16_bf16 S1              | 16w16_bf16_128 S1  maxspread 1.03
1792x7168x4096   196  0.77 |     94.7    117.5    138.6 | 16w16_bf16 S1              | 16w16_bf16_128 S1  maxspread 1.04
2048x7168x4096   224  0.88 |    101.4    126.5    147.1 | 16w16_bf16 S1              | 16w16_bf16_128 S1  maxspread 1.02
2304x7168x4096   252  0.98 |    107.6    132.1        - | 16w16_bf16 S1              | 16w16_bf16_128 S1  maxspread 1.04
2560x7168x4096   280  1.09 |    159.4    157.8    181.1 | 16w16_bf16_128 S1          | 16w16_bf16 S1  maxspread 1.02
3072x7168x4096   336  1.31 |    168.1    180.4    214.6 | 16w16_bf16 S1              | 16w16_bf16 S1  maxspread 1.02
3584x7168x4096   392  1.53 |    176.9    210.0    246.1 | 16w16_bf16 S1              | 16w16_bf16 S1  maxspread 1.01
4096x7168x4096   448  1.75 |    188.2    222.8        - | 16w16_bf16 S1              | 16w16_bf16 S1  maxspread 1.01
===== i8_rowcol
shape          ct256  fill |     t256     t128      t64 | winner                     | rule
1024x4096x4096    64  0.25 |        -     27.3        - | 8w8_i8_128 S1              | 8w8_i8_128 S1  maxspread 1.03
1280x4096x4096    80  0.31 |     42.7     32.6     35.0 | 8w8_i8_128 S1              | 8w8_i8_128 S1  maxspread 1.07
1536x4096x4096    96  0.38 |     43.6     35.8     40.8 | 8w8_i8_128 S1              | 8w8_i8_128 S1  maxspread 1.07
1792x4096x4096   112  0.44 |     45.2     39.5        - | 8w8_i8_128 S1              | 8w8_i8_128 S1  maxspread 1.07
2048x4096x4096   128  0.50 |     47.0     41.8        - | 8w8_i8_128 S1              | 8w8_i8_128 S1  maxspread 1.05
2304x4096x4096   144  0.56 |     48.9     57.5     61.3 | 8w8_i8 S1                  | 8w8_i8_128 S1  maxspread 1.03
2560x4096x4096   160  0.62 |     51.8     59.5     65.2 | 8w8_i8 S1                  | 8w8_i8_128 S1  maxspread 1.05
3072x4096x4096   192  0.75 |     56.3     65.4        - | 8w8_i8 S1                  | 8w8_i8_128 S1  maxspread 1.04
3584x4096x4096   224  0.88 |     60.4     74.4     88.5 | 8w8_i8 S1                  | 8w8_i8_128 S1  maxspread 1.06
4096x4096x4096   256  1.00 |     63.2     81.2        - | 8w8_i8 S1                  | 8w8_i8 S1  maxspread 1.03
1024x7168x4096   112  0.44 |     44.7     37.7        - | 8w8_i8_128 S1              | 8w8_i8_128 S1  maxspread 1.06
1280x7168x4096   140  0.55 |     48.6     56.7     59.8 | 8w8_i8 S1                  | 8w8_i8_128 S1  maxspread 1.03
1536x7168x4096   168  0.66 |     52.4     59.9        - | 8w8_i8 S1                  | 8w8_i8_128 S1  maxspread 1.04
1792x7168x4096   196  0.77 |     56.3     68.9     83.0 | 8w8_i8 S1                  | 8w8_i8_128 S1  maxspread 1.06
2048x7168x4096   224  0.88 |     59.5     74.4     89.4 | 8w8_i8 S1                  | 8w8_i8_128 S1  maxspread 1.05
2304x7168x4096   252  0.98 |     63.9     78.8        - | 8w8_i8 S1                  | 8w8_i8_128 S1  maxspread 1.05
2560x7168x4096   280  1.09 |     92.7     94.3    110.4 | 8w8_i8 S1                  | 8w8_i8 S1  maxspread 1.04
3072x7168x4096   336  1.31 |     99.0    106.6    129.1 | 8w8_i8 S1                  | 8w8_i8 S1  maxspread 1.04
3584x7168x4096   392  1.53 |    106.7    125.1    148.7 | 8w8_i8 S1                  | 8w8_i8 S1  maxspread 1.04
4096x7168x4096   448  1.75 |    113.6    134.0        - | 8w8_i8 S1                  | 8w8_i8 S1  maxspread 1.03
===== fp8_rowcol
shape          ct256  fill |     t256     t128      t64 | winner                     | rule
1024x4096x4096    64  0.25 |        -     27.0        - | 8w8_rowcol_fp8_128 S1      | 8w8_rowcol_fp8_128 S1  maxspread 1.05
1280x4096x4096    80  0.31 |     41.0     32.2     33.9 | 8w8_rowcol_fp8_128 S1      | 8w8_rowcol_fp8_128 S1  maxspread 1.14
1536x4096x4096    96  0.38 |     41.9     35.2     40.6 | 8w8_rowcol_fp8_128 S1      | 8w8_rowcol_fp8_128 S1  maxspread 1.11
1792x4096x4096   112  0.44 |     43.1     38.8        - | 8w8_rowcol_fp8_128 S1      | 8w8_rowcol_fp8_128 S1  maxspread 1.08
2048x4096x4096   128  0.50 |     46.1     40.3        - | 8w8_rowcol_fp8_128 S1      | 8w8_rowcol_fp8_128 S1  maxspread 1.07
2304x4096x4096   144  0.56 |     47.7     56.4     60.3 | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8_128 S1  maxspread 1.08
2560x4096x4096   160  0.62 |     51.3     58.0     63.2 | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8_128 S1  maxspread 1.05
3072x4096x4096   192  0.75 |     54.3     65.0        - | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8_128 S1  maxspread 1.04
3584x4096x4096   224  0.88 |     59.1     73.4     88.8 | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8_128 S1  maxspread 1.04
4096x4096x4096   256  1.00 |     61.9     80.6        - | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8 S1  maxspread 1.07
1024x7168x4096   112  0.44 |     43.5     37.7        - | 8w8_rowcol_fp8_128 S1      | 8w8_rowcol_fp8_128 S1  maxspread 1.08
1280x7168x4096   140  0.55 |     47.4     55.6     59.6 | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8_128 S1  maxspread 1.05
1536x7168x4096   168  0.66 |     51.8     59.9        - | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8_128 S1  maxspread 1.06
1792x7168x4096   196  0.77 |     54.9     68.3     81.7 | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8_128 S1  maxspread 1.04
2048x7168x4096   224  0.88 |     58.3     73.2     88.5 | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8_128 S1  maxspread 1.09
2304x7168x4096   252  0.98 |     62.1     78.4        - | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8_128 S1  maxspread 1.04
2560x7168x4096   280  1.09 |     90.5     91.8    110.1 | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8 S1  maxspread 1.03
3072x7168x4096   336  1.31 |     97.1    105.6    129.0 | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8 S1  maxspread 1.05
3584x7168x4096   392  1.53 |    104.6    123.3    147.5 | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8 S1  maxspread 1.03
4096x7168x4096   448  1.75 |    112.0    132.6        - | 8w8_rowcol_fp8 S1          | 8w8_rowcol_fp8 S1  maxspread 1.03
```

## 64-row form by split (us), i8 and bf16 (c64 = ceil(M/64) ceil(N/128) CTAs; the 128-tile best for comparison)

```
===== i8_rowcol
shape           c64 |      S1      S2      S4      S8 | 128-tile best
64x4096x4096     32 |    11.6    10.2     9.4    10.6 | 17.2
128x4096x4096    64 |    11.8    11.8    11.8    15.7 | 17.4
192x4096x4096    96 |    12.0    13.3    16.2       - | 17.6
256x4096x4096   128 |    12.3    15.0    17.4       - | 17.8
384x4096x4096   192 |    12.9       -       -       - | 18.8
512x4096x4096   256 |    14.0       -       -       - | 22.2
64x7168x4096     56 |    11.8    10.9    11.9    14.5 | 18.0
128x7168x4096   112 |    12.4    12.7    16.1       - | 20.4
192x7168x4096   168 |    13.1    18.9       -       - | 22.8
256x7168x4096   224 |    14.0       -       -       - | 23.4
384x7168x4096   336 |    21.3       -       -       - | 24.2
512x7168x4096   448 |    23.7       -       -       - | 25.9
64x4096x8192     32 |       -       -    11.9    12.3 | 40.3
128x4096x8192    64 |       -    15.5    13.7    17.4 | 41.9
192x4096x8192    96 |    21.1    16.6    18.6    21.7 | 40.9
256x4096x8192   128 |    21.8    18.3    20.7       - | 41.9
384x4096x8192   192 |    23.9    27.8    28.7       - | 42.5
512x4096x8192   256 |    27.0    31.4       -       - | 43.1
64x7168x8192     56 |       -    16.8    16.0    18.5 | 40.3
128x7168x8192   112 |    21.5    18.0    21.4       - | 41.5
192x7168x8192   168 |    23.1    27.1    28.5       - | 41.5
256x7168x8192   224 |    25.2    30.0       -       - | 42.0
384x7168x8192   336 |    40.0       -       -       - | 43.7
512x7168x8192   448 |    46.2       -       -       - | 47.5
===== bf16
shape           c64 |      S1      S2      S4      S8 | 128-tile best
64x4096x4096     32 |       -       -    11.9    12.3 | 15.2
128x4096x4096    64 |       -    15.3    13.8    17.1 | 17.3
192x4096x4096    96 |    20.4    16.4    18.6       - | 20.6
256x4096x4096   128 |    21.1    17.8    19.6       - | 21.8
384x4096x4096   192 |    22.8    26.6    28.1       - | 27.9
512x4096x4096   256 |    24.9    30.4       -       - | 31.1
64x7168x4096     56 |       -    16.1    15.9    18.8 | 20.6
128x7168x4096   112 |    20.8    18.0    21.0       - | 22.5
192x7168x4096   168 |    21.9    26.2    27.4       - | 27.8
256x7168x4096   224 |    23.8    28.5       -       - | 30.2
384x7168x4096   336 |    39.1       -       -       - | 41.2
512x7168x4096   448 |    43.5       -       -       - | 45.7
64x4096x8192     32 |       -       -    17.0    17.5 | 20.9
128x4096x8192    64 |       -       -    18.7    22.3 | 23.1
192x4096x8192    96 |       -    26.8    27.3    29.6 | 30.0
256x4096x8192   128 |       -    30.4    30.1       - | 32.4
384x4096x8192   192 |    44.1    46.1    44.8       - | 42.8
512x4096x8192   256 |    50.1    51.4       -       - | 50.2
64x7168x8192     56 |       -       -    23.5    26.4 | 29.6
128x7168x8192   112 |       -    28.8    31.7       - | 32.2
192x7168x8192   168 |    42.3    47.5    43.8       - | 44.2
256x7168x8192   224 |    47.4    49.6       -       - | 46.5
384x7168x8192   336 |    75.5    73.6       -       - | 67.4
512x7168x8192   448 |    83.2       -       -       - | 76.6
```

## Gate: perf_ab gemm family, old vs new rules (IMPROVED / FAIL cases)

Families:

| family | cases | geomean new/old | allowed drop | verdict |
|---|--:|--:|--:|---|
| gemm | 79 | 1.042 | 1.0% | OK |



```
| gemm/bf16/1280x7168x4096 | 782.00 | 914.70 | 1.170 | 2.0% | IMPROVED | - | - | - |
| gemm/bf16/128x7168x8192 | 469.10 | 519.40 | 1.107 | 6.4% | IMPROVED | - | - | - |
| gemm/bf16/1792x8192x4096 | 958.80 | 1195 | 1.247 | 2.6% | IMPROVED | - | - | - |
| gemm/bf16/192x4096x4096 | 349.50 | 393.70 | 1.126 | 2.0% | IMPROVED | - | - | - |
| gemm/bf16/2048x7168x7168 | 1029 | 1286 | 1.250 | 2.0% | IMPROVED | - | - | - |
| gemm/bf16/320x8192x6144 | 548.40 | 581.80 | 1.061 | 3.9% | IMPROVED | - | - | - |
| gemm/bf16/768x4096x8192 | 692.40 | 720.80 | 1.041 | 3.8% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/1280x7168x4096 | 1345 | 1566 | 1.165 | 4.1% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/128x7168x8192 | 701.00 | 836.90 | 1.194 | 4.0% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/160x4096x8192 | 611.50 | 664.30 | 1.086 | 4.0% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/1792x8192x4096 | 1626 | 2054 | 1.263 | 3.2% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/192x4096x4096 | 411.70 | 548.90 | 1.333 | 2.9% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/2048x7168x7168 | 1773 | 2244 | 1.266 | 4.2% | IMPROVED | - | - | - |
| gemm/i8_rowcol/1280x7168x4096 | 1319 | 1525 | 1.156 | 2.4% | IMPROVED | - | - | - |
| gemm/i8_rowcol/128x7168x8192 | 695.30 | 816.00 | 1.174 | 2.0% | IMPROVED | - | - | - |
| gemm/i8_rowcol/160x4096x8192 | 588.60 | 667.10 | 1.133 | 2.5% | IMPROVED | - | - | - |
| gemm/i8_rowcol/1792x8192x4096 | 1655 | 2033 | 1.228 | 4.2% | IMPROVED | - | - | - |
| gemm/i8_rowcol/192x4096x4096 | 403.10 | 535.30 | 1.328 | 2.0% | IMPROVED | - | - | - |
| gemm/i8_rowcol/2048x7168x7168 | 1770 | 2202 | 1.244 | 2.3% | IMPROVED | - | - | - |
```

## Split-K S = 8 at small M (node 53 / GPU 1, 2026-11-12, 7 interleaved repetitions, median us; N = 4096, 64-row form)

`tools/compose_search.py check-gemm --kinds bf16,i8_rowcol,fp8_rowcol --ms 1,4,16,32,64,128 --ns 4096,7168,8192 --ks 4096,8192,14336` with the split candidates S = 2, 4, 8, 16. At N = 4096 the grid of the 64-row form is c = 32 CTAs, so S = 4 fills 128 of the 256 CUs and S = 8 fills them all:

| kind | M | K | S4 | S8 | S16 |
|---|--:|--:|--:|--:|--:|
| bf16 | 1 | 4096 | 9.2 | 7.4 | 8.5 |
| bf16 | 16 | 4096 | 9.4 | 8.0 | 9.8 |
| bf16 | 32 | 4096 | 10.2 | 9.3 | 11.3 |
| bf16 | 64 | 4096 | 11.8 | 12.4 | 15.4 |
| bf16 | 1 | 8192 | 14.3 | 14.5 | 14.5 |
| bf16 | 1 | 14336 | 20.8 | 20.3 | 20.8 |
| fp8 per-token | 1 | 4096 | 6.7 | 6.0 | 6.9 |
| fp8 per-token | 16 | 4096 | 7.0 | 6.6 | 7.9 |
| fp8 per-token | 32 | 4096 | 7.7 | 7.8 | 9.8 |
| fp8 per-token | 64 | 4096 | 9.2 | 10.5 | 14.6 |
| fp8 per-token | 1 | 8192 | 8.8 | 7.2 | 8.4 |
| fp8 per-token | 16 | 8192 | 9.1 | 7.9 | 9.7 |
| fp8 per-token | 32 | 8192 | 10.1 | 9.2 | 11.2 |
| fp8 per-token | 64 | 8192 | 11.6 | 12.3 | 15.3 |
| fp8 per-token | 1 / 4 / 16 / 32 | 14336 | 12.8 / 13.0 / 13.2 / 13.9 | 12.7 / 12.8 / 13.1 / 13.9 | - |
| int8 per-token | 1 | 8192 | 9.1 | 7.4 | 8.7 |
| int8 per-token | 32 | 14336 | 13.8 | 13.9 | - |

The refit of the time model on this wider data (K up to 14336, S up to 16) fits worse (12 % mean error) and alone would regress M = 64 and N >= 7168 shapes; the rule keeps the fitted model and adds the measured S8 override for M <= 16.