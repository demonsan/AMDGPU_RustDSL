# One-hot ISA probes: `v_smfmac_f32_16x16x128_fp8_fp8` and `ds_read_b64_tr_b8` (gfx950)

Prerequisite for the FMHA row-sum restructure (two accumulated smfmacs). Everything here is **measured** on MI355X (GPU 4) with one-hot inputs;
the recorded maps are committed and replayed by a unit test. Code: kernels `probe_smfmac_gen` / `probe_lds_tr_b8_gen`
(`crates/amdgpu-kernels/src/probes.rs`), host checker `tests/gpu/isa_probe_host.cpp`, re-record with `tools/isa_probes.sh [--update]`, recorded maps
`crates/amdgpu-isa/data/{smfmac_16x16x128_fp8,lds_tr_b8}.map`, layouts / semantics in `crates/amdgpu-isa/src/lib.rs`
(`MmaAtom::smfmac_16x16x128_fp8`, `smfmac`, `tr_b8`; provenance `Probe`), replay test `crates/amdgpu-isa/tests/probe_maps.rs`.

## 1. `v_smfmac_f32_16x16x128_fp8_fp8 D(4), A(4), B(8), idx` (A sparse, 2:4)

### Method

The probe kernel loads every lane's A (4 dwords), B (8 dwords) and index dword from global memory, issues ONE smfmac with a zero accumulator and stores D, so the
host chooses every register bit (E4M3 1.0 = 0x38). One probe = one wave.

1. **A bytes -> D rows.** A one-hot at each (lane, byte) (1024 probes), B all ones, idx `0x44444444`: the set of D (lane, dword) that become 1.
2. **B bytes -> D columns.** B one-hot at each (lane, register, byte) (2048 probes), A all ones, both complementary index words (`0x44444444`, `0xeeeeeeee`).
3. **A byte <-> B byte (which k).** For an index word and an A one-hot, B holds ones on the elements whose element index has bit j set (j = 0..10, 11 probes) plus one
   all-ones probe: D(row of A, column n) tells, per column, the index of the B element that has the same k as the A byte. Run for 14 index words: the six uniform
   nibbles `0x4 0x8 0xc 0x9 0xd 0xe` (kept positions (0,1) (0,2) (0,3) (1,2) (1,3) (2,3); `0x44444444` is the vendor row-sum word, `0xeeeeeeee` its complement) and
   eight "walking" words (all nibbles `0x4`, nibble j = `0xe`), and for three random per-lane index words.

Names (m, n, k) are class labels defined by the probes: m = the lane % 16 of the A bytes of a row class, n = the lane % 16 of the B bytes of a column class, k = the
B element's `32 (lane / 16) + 4 register + byte` (the 16 B elements matched to one A byte, one per column, always carry the same label: 0 disagreements).

### Results (all checks: 0 violations)

* All ones, `0x44444444`: every D element = 64 (64 selected products).
* **D:** lane l, dword r = (row m = 4 (l/16) + r, column n = l % 16). Exactly 16 row classes and 16 column classes; each A byte feeds the 16 D elements of its row,
  each B byte the 16 D elements of its column (one product each).
* **A (compressed tile 16 x 64):** lane l (row m = l % 16, lane group g = l / 16) holds stored values s = 0..15 (register s / 4, byte s % 4), compressed column
  c = 16 g + s. The row only depends on the lane % 16.
* **B (dense 128 x 16):** lane l (column n = l % 16, group h = l / 16) holds k = 32 h + 4 r + y (register r = 0..7, byte y).
* **Index semantics.** Stored value s of A lane l uses the 2-bit field at bits `2 s .. 2 s + 1` of **its own lane's** index dword (lanes never read each other's word:
  3 random per-lane variants, 0 violations). With q = s / 2 (the pair of kept entries of one group of 4) and g = l / 16:

  ```
  k(l, s, idx) = 32 h + 4 r + ((idx >> 2 s) & 3),   (h, r) = (2 (g & 1) + q / 4,  4 (g >> 1) + q % 4)
               = 64 (g & 1) + 16 (g >> 1) + 32 (q / 4) + 4 (q % 4) + ((idx >> 2 s) & 3)
  ```

  So a pair (2 q, 2 q + 1) contracts with **one B dword** (lane group h, register r) and the two index fields pick which BYTES of that dword they multiply; the
  hardware accepts any two fields (all six ascending (p0, p1) pairs were measured; nibble = p0 | p1 << 2; descending or equal pairs were not probed).
  The A lane group g selects the register half and the K half: g = 0: B groups {0, 1}, registers 0..3; g = 1: groups {2, 3}, registers 0..3; g = 2: groups {0, 1},
  registers 4..7; g = 3: groups {2, 3}, registers 4..7.
* **Two accumulated smfmacs cover all of K.** `0x44444444` keeps bytes (0, 1) of every B dword, `0xeeeeeeee` bytes (2, 3): with both, every one of the 128 k of every
  row is contracted exactly once (test `the_two_vendor_index_words_cover_all_128_k_exactly_once`). With an all-ones A the pair gives the 128-key row sum.
* **What it means for a row-sum restructure.** For `row sum of P` (A = ones, B = the P fragment) any index words whose kept positions are complementary
  (e.g. `0x44444444` then `0xeeeeeeee`, or `0x88888888` (0,2) then `0xdddddddd` (1,3)) are equivalent; the A operand only has to be non-zero in the stored values
  that should count (a "mask" on A bytes zeroes the corresponding k's: byte s of lane l masks `k_of(l, s, idx)`).

### Layouts in `amdgpu-isa`

| operand | TV layout (lane, value) -> element | convention |
|---|---|---|
| A (compressed) | `((16,4),16) : ((1,256),16)` | `m + 16 c`, c = 16 g + s |
| B | `((16,4),32) : ((1,512),16)` | `n + 16 k` |
| C | `((16,4),4) : ((16,4),1)` | `m + 16 n` |

## 2. `ds_read_b64_tr_b8 v[0:1], vaddr`

### Method

The kernel copies 4096 host-chosen bytes into LDS, then every lane reads the 8-byte word at the address the host gives it. Two passes fill the LDS with
`addr & 255` and `(addr >> 8) & 255` so every received byte identifies its source address uniquely; the (source lane, byte offset) of each received byte follows from the lane
addresses. 45 address sets: consecutive qwords, stride 16, an 8x8 column-major block, 3072-based, `l ^ 5`, and 40 random permutations of 8-byte blocks.

### Result (0 differences between all 45 address sets, 0 unmatched)

Destination lane d, byte j (register j / 4, byte j % 4) receives **byte d % 8 of the word read by lane 16 (d / 16) + 2 j + (d % 16) / 8**. Inside every 16-lane group
the eight words of the even lanes are transposed 8 x 8 bytes into destination lanes 0..7, the eight words of the odd lanes into destination lanes 8..15. The map is
independent of the addresses. As a TV layout over the source tile "64 lanes x 8 bytes" (element 8 s' + h): `((8,2,4),8) : ((1,8,128),16)`
(`tr_b8::dest_tv`, `tr_b8::source`).

## 3. Reproduce

```
HIP_VISIBLE_DEVICES=4 bash tools/isa_probes.sh            # re-runs the probes, compares with the committed maps
cargo test -p amdgpu-isa                                   # replays the committed maps against the atoms
```

Not covered: the immediate-offset forms of `ds_read_b64_tr_b8`, the scaled / other-format smfmac variants, bank-conflict behaviour of the transposing read.