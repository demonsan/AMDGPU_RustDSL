use amdgpu_dsl::{FnBuilder, Function};
pub trait FusedSlots {
    type Cx;
    fn prelude(&mut self, b: &mut FnBuilder) -> Self::Cx;
    fn w1_image(&mut self, b: &mut FnBuilder, cx: &mut Self::Cx);
    fn w2_image(&mut self, b: &mut FnBuilder, cx: &mut Self::Cx);
    fn x_first(&mut self, b: &mut FnBuilder, cx: &mut Self::Cx);
    fn loop_open(&mut self, b: &mut FnBuilder, cx: &mut Self::Cx);
    fn gemm1_act(&mut self, b: &mut FnBuilder, cx: &mut Self::Cx);
    fn x_next(&mut self, b: &mut FnBuilder, cx: &mut Self::Cx);
    fn gemm2(&mut self, b: &mut FnBuilder, cx: &mut Self::Cx);
    fn loop_close(&mut self, b: &mut FnBuilder, cx: &mut Self::Cx);
}
pub fn build_fused<S: FusedSlots>(name: &str, slots: &mut S) -> Function {
    let mut fb = FnBuilder::new(name, 256);
    let b = &mut fb;
    let mut cx = slots.prelude(b);
    slots.w1_image(b, &mut cx);
    slots.w2_image(b, &mut cx);
    slots.x_first(b, &mut cx);
    slots.loop_open(b, &mut cx);
    slots.gemm1_act(b, &mut cx);
    slots.x_next(b, &mut cx);
    slots.gemm2(b, &mut cx);
    slots.loop_close(b, &mut cx);
    fb.finish()
}