//! Lane identities and wave-group roles.

use amdgpu_dsl::ev;
use crate::shared::SlotIdx;
use amdgpu_dsl::{FnBuilder, Op, Opnd, ValueId, I32};

/// The per-lane ids every component derives its addressing from (emitted exactly like `fmha_t_build`: `tid & 63`, `tid >> 6`, `lane & 31`, `lane >> 5`).
#[derive(Clone, Copy, Debug)]
pub struct Lane {
    /// workitem id in the workgroup
    pub tid: ValueId,
    /// lane in the wave (0..63)
    pub lane: ValueId,
    /// wave in the workgroup
    pub wave: ValueId,
    /// lane & 31: the column (query) index of a 32x32 MFMA accumulator
    pub j: ValueId,
    /// lane >> 5: the lane half (0/1)
    pub half: ValueId,
}

impl Lane {
    pub fn new(b: &mut FnBuilder) -> Lane {
        let tid = b.let_(Op::WorkitemId);
        Lane::from_tid(b, tid)
    }
    /// From an already-read workitem id.
    pub fn from_tid(b: &mut FnBuilder, tid: ValueId) -> Lane {
        let lane = b.bind_i32(ev(tid) & 63);
        let wave = b.bind_i32(ev(tid).lshr(6));
        let j = b.bind_i32(ev(lane) & 31);
        let half = b.bind_i32(ev(lane).lshr(5));
        Lane { tid, lane, wave, j, half }
    }
    pub fn tid(&self) -> I32 {
        ev(self.tid)
    }
    pub fn lane(&self) -> I32 {
        ev(self.lane)
    }
    pub fn wave(&self) -> I32 {
        ev(self.wave)
    }
    pub fn j(&self) -> I32 {
        ev(self.j)
    }
    pub fn half(&self) -> I32 {
        ev(self.half)
    }
}

/// A wave group of a role-split kernel. The tag is a compile-time value: role-dependent code (who stages which tile, where the phase-offset barriers go) is
/// selected while building, inside the `begin_if(is_a)` / `else_` branches that `Roles::each` emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// waves 0 .. group-1 (the `is_a` side)
    A,
    /// waves group .. 2*group-1
    B,
    /// a single group: every wave (no role split)
    All,
}

/// The role split of the workgroup (`_vs128`: two groups of 4 waves; group A = waves 0-3).
#[derive(Clone, Copy, Debug)]
pub struct Roles {
    /// wave < group (I1)
    pub is_a: ValueId,
    /// wave index inside its group (wave & (group-1))
    pub wave_in_group: ValueId,
    pub group: u32,
}

impl Roles {
    /// Two groups of `group` waves (a power of two): `is_a = wave < group`, `wave_in_group = wave & (group - 1)`.
    pub fn split(b: &mut FnBuilder, lane: &Lane, group: u32) -> Roles {
        assert!(group.is_power_of_two(), "Roles::split: the group size must be a power of two");
        let wave_in_group = b.bind_i32(lane.wave() & (group as i32 - 1));
        let is_a = b.bind_cond(lane.wave().lt(group as i32));
        Roles { is_a, wave_in_group, group }
    }
    /// One group made of every wave (no split): `is_a` is unused and `wave_in_group` is the wave.
    pub fn single(lane: &Lane, group: u32) -> Roles {
        Roles { is_a: lane.wave, wave_in_group: lane.wave, group }
    }
    /// Emits `if is_a { body(A) } else { body(B) }`.
    pub fn each(&self, b: &mut FnBuilder, mut body: impl FnMut(&mut FnBuilder, Role)) {
        b.begin_if(Opnd::Val(self.is_a));
        body(b, Role::A);
        b.else_();
        body(b, Role::B);
        b.end_if();
    }
    /// The workgroup-wide wave id of this role's wave (`wave_in_group` for A, `wave_in_group + group` for B): the key of per-wave private regions.
    pub fn wave_id(&self, b: &mut FnBuilder, role: Role) -> ValueId {
        match role {
            Role::A | Role::All => self.wave_in_group,
            Role::B => b.bind_i32(ev(self.wave_in_group) + self.group as i32),
        }
    }
}

/// Two code copies selected once by a workgroup-uniform condition (e.g. the short/long sequence recipes): `if cond { body(true) } else { body(false) }`.
pub fn versioned(b: &mut FnBuilder, cond: ValueId, mut body: impl FnMut(&mut FnBuilder, bool)) {
    b.begin_if(Opnd::Val(cond));
    body(b, true);
    b.else_();
    body(b, false);
    b.end_if();
}

/// One iteration of the pipeline as the body sees it (the generation / slot bookkeeping every pipeline template hands to its body; L3 data, used by the L3 helpers (`reduce`) and the L2 pipeline driver).
#[derive(Clone, Debug)]
pub struct TileIt {
    /// tile counter (0 in the prologue, the loop variable inside the loop and in the tail)
    pub it: Opnd,
    /// the tile being computed: slot `idx`, generation `it`
    pub cur: SlotIdx,
    /// tile t+1: slot / generation (`_vs128`: staged in this iteration; `_t128`: staged in the previous one)
    pub next: SlotIdx,
    /// tile t+2 (`OneWindow` only)
    pub n2: Option<SlotIdx>,
    /// the masked copy of the last (partial) tile
    pub masked: bool,
    /// a tile is staged in this iteration (false in the peeled tail)
    pub staged: bool,
    /// rows per tile
    pub tile_rows: i32,
}

impl TileIt {
    /// First row (key) of the current tile: `it * tile_rows`.
    pub fn key0(&self) -> I32 {
        amdgpu_dsl::ox(self.it) << (self.tile_rows.trailing_zeros() as i32)
    }
}
