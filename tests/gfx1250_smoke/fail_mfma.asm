# EXPECT-FAIL: no MFMA on gfx1250 (the matrix instructions are WMMA / SWMMAC)
.text
v_mfma_f32_32x32x16_bf16 v[0:15], v[16:19], v[20:23], v[0:15]
