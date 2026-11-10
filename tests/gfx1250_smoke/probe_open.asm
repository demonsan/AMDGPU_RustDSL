# probes (assembler acceptance only, not hardware semantics): cluster_load_async_to_lds_b8 and an s_set_vgpr_msb immediate above 0xff
# are both accepted by upstream MC tests (gfx1250_asm_vflat.s, gfx1250_asm_sopp.s @274ab686); recorded for the rocm-10.1 assembler.
.text
cluster_load_async_to_lds_b8 v1, v[2:3], off offset:-64
s_set_vgpr_msb 0x100
