//! Stage instances (docs/fusion-stage-library-design.md §1, §7): a stage is a compile-time, per-shape instantiation of one library primitive that
//! RETURNS emission units plus its boundary contract; the kernel composes stage instances and decides the orchestration (the weave order, the
//! scheduler regions) at the call site. Emitting a stage instance opens a `Node::Stage` region with the contract (checked by GV-STAGE), weaves the
//! units (declaration order unless the kernel gives one) inside a weave region of the stage's name, records the live-out / scalar hand-overs
//! produced at emission time, and closes the region.
//!
//! The registry ([`registry`]) lists the stage kinds the kernels in this crate family instantiate (key, primitive, parameters): the op graph and the
//! skeleton generator (`opgraph`, P3) refer to stages by key.

use crate::weave::{Units, Weaver};
use amdgpu_dsl::{FnBuilder, StageContract, StageScalar, StageVal};

/// The library primitives (docs/fusion-stage-library-design.md §1.1; `X`, the cross-device exchange, is out of scope).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prim {
    /// global <-> LDS <-> register moves (incl. index-driven gather / scatter)
    M,
    /// MMA main loop
    A,
    /// elementwise map (activations, affine, packing, epilogue chains)
    E,
    /// reductions
    R,
    /// online reductions with a rescaled consumer (softmax)
    O,
    /// quantize (amax -> scale -> convert / pack)
    Q,
    /// select / sort
    S,
    /// scan / recurrence
    P,
}

/// Hand-overs known only once the stage's units have been emitted (the IR values exist then).
pub type Handover<'a> = Box<dyn FnOnce(&mut FnBuilder) -> (Vec<StageVal>, Vec<StageScalar>) + 'a>;

/// One stage instance.
pub struct StageInst<'a> {
    /// registry key of the stage kind
    pub key: &'static str,
    pub prim: Prim,
    /// instance name (unique in the kernel; the Node::Stage / weave region name and the orchestration slot prefix)
    pub name: String,
    /// instantiation parameters (for reports and the op graph)
    pub params: Vec<(String, String)>,
    pub contract: StageContract,
    pub units: Units,
    /// live-out / scalar-out sets, evaluated after the units were emitted
    pub handover: Option<Handover<'a>>,
    /// open a `Node::Stage` region (false: units only, e.g. the unfused forms of a kernel keep their plain program)
    pub region: bool,
    /// emitted BEFORE the region opens, at the caller's scope: bindings shared with other instances (e.g. the per-slot LDS offsets that
    /// `SlotIdx` caches for every consumer of a slot) so they are not hand-overs between stages
    pub prebind: Option<Box<dyn FnOnce(&mut FnBuilder) + 'a>>,
}

impl<'a> StageInst<'a> {
    pub fn new(key: &'static str, prim: Prim, name: impl Into<String>, contract: StageContract, units: Units) -> StageInst<'a> {
        StageInst { key, prim, name: name.into(), params: Vec::new(), contract, units, handover: None, region: true, prebind: None }
    }
    pub fn param(mut self, k: &str, v: impl ToString) -> StageInst<'a> {
        self.params.push((k.to_string(), v.to_string()));
        self
    }
    pub fn handover(mut self, f: impl FnOnce(&mut FnBuilder) -> (Vec<StageVal>, Vec<StageScalar>) + 'a) -> StageInst<'a> {
        self.handover = Some(Box::new(f));
        self
    }
    pub fn prebind(mut self, f: impl FnOnce(&mut FnBuilder) + 'a) -> StageInst<'a> {
        self.prebind = Some(Box::new(f));
        self
    }
    pub fn plain(mut self, plain: bool) -> StageInst<'a> {
        self.region = !plain;
        self
    }
    /// Emits the stage in the units' declaration order.
    pub fn emit(self, b: &mut FnBuilder) {
        self.emit_with(b, |w| w.seq());
    }
    /// Emits the stage with the kernel's orchestration: `weave` receives the stage's weaver (labelled with the stage name) and returns it with an
    /// order / strategy.
    pub fn emit_with(self, b: &mut FnBuilder, weave: impl FnOnce(Weaver) -> Weaver) {
        let StageInst { name, contract, units, handover, region, prebind, .. } = self;
        if let Some(p) = prebind {
            p(b);
        }
        if region {
            b.stage_begin(&name, contract);
        }
        weave(Weaver::new(units).phase(name.clone())).emit(b);
        if let Some(h) = handover {
            let (vals, scalars) = h(b);
            if region {
                for v in vals {
                    b.stage_live_out(v);
                }
                for s in scalars {
                    b.stage_scalar_out(s);
                }
            }
        }
        if region {
            b.stage_end();
        }
    }
}

/// Several stage instances woven together into ONE phase (their units interleave under the kernel's order): the instances are emitted plain (a
/// `Node::Stage` region cannot interleave with another); returns the phase's weaver (label `phase`) and the instances' (key, name, params) for
/// reports. Unit ids are the instances' ids shifted by the units of the instances before them (declaration order = instance order).
pub fn compose<'a>(phase: &str, insts: Vec<StageInst<'a>>) -> (Weaver, Vec<(&'static str, String, Vec<(String, String)>)>) {
    let mut u = Units::new();
    let mut info = Vec::new();
    for inst in insts {
        assert!(inst.handover.is_none(), "compose: stage {} has a hand-over (emit it on its own)", inst.name);
        info.push((inst.key, inst.name, inst.params));
        u.append(inst.units, &[]);
    }
    (Weaver::new(u).phase(phase.to_string()), info)
}

/// Several stage instances merged into ONE stage instance (one `Node::Stage` region named `name` with `contract`): their units interleave under
/// the kernel's weave order. The members' keys and parameters are recorded as parameters of the merged instance.
pub fn merge<'a>(name: &str, contract: StageContract, insts: Vec<StageInst<'a>>) -> StageInst<'a> {
    let mut u = Units::new();
    let mut params = Vec::new();
    for inst in insts {
        assert!(inst.handover.is_none() && inst.prebind.is_none(), "merge: stage {} has a hand-over / prebind (attach them to the merged instance)", inst.name);
        params.push((inst.name.clone(), inst.key.to_string()));
        params.extend(inst.params.into_iter().map(|(k, v)| (format!("{}.{k}", inst.name), v)));
        u.append(inst.units, &[]);
    }
    let mut m = StageInst::new("merged", Prim::A, name, contract, u);
    m.params = params;
    m
}

/// One stage kind of the registry.
#[derive(Clone, Copy, Debug)]
pub struct StageDesc {
    pub key: &'static str,
    pub prim: Prim,
    /// (parameter, meaning)
    pub params: &'static [(&'static str, &'static str)],
    pub doc: &'static str,
}
