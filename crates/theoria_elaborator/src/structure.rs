//! Construction of kernel inductive triples from elaborated `Structure`
//! declarations.
//!
//! This module concentrates the de Bruijn arithmetic — the part of the
//! elaborator most likely to hide a bug — in one auditable file. Every
//! index below is derived from first principles and cross-checked
//! against the `Point2D` and `Wrapper` examples in this header.
//!
//! ## Notation
//!
//! Given a structure with:
//!
//! * `k` parameters `P₁, ..., Pₖ`, each elaborated in the context of the
//!   parameters before it,
//! * `m` fields with types `F₁, ..., Fₘ`, each elaborated in the parameter
//!   context `[p₁, ..., pₖ]` (so `pᵢ = Var(k - i)`, 1-indexed `i`),
//! * `nⱼ ∈ {0, 1}` indicating whether field `j` (1-indexed) is directly
//!   recursive (its type is exactly `T p⃗`),
//! * `Oⱼ = Σ_{j'≤j} n_{j'}` the cumulative recursive-field count,
//!
//! the recursor `T.rec.{v}` has type
//!
//! ```text
//! Π {p₁:P₁} ... {pₖ:Pₖ}.
//!   Π {motive : T p⃗ -> Sort v}.
//!     Π (minor : Π (f₁:F₁) [ih₁ : motive f₁] ... (fₘ:Fₘ) [ihₘ : motive fₘ].
//!                   motive (T.mk p⃗ f₁ ... fₘ)).
//!       Π (major : T p⃗).
//!         motive major
//! ```
//!
//! Parameters and the motive are implicit (matching `List.rec`); the
//! minor, the major, and every field/IH binder inside the minor are
//! explicit.
//!
//! The single ι-reduction rule fires when the major premise is
//! `T.mk p⃗ f₁ ... fₘ`. Its right-hand side,
//!
//! ```text
//! minor f₁ [T.rec p⃗ motive minor f₁] ... fₘ [T.rec p⃗ motive minor fₘ],
//! ```
//!
//! lives in the rule context `[p₁, ..., pₖ, motive, minor, f₁, ..., fₘ]`
//! (parameters outermost, fields innermost — the layout the kernel's
//! NbE ι-reduction builds).
//!
//! ## Worked example: `Point2D`
//!
//! `k = 0`, `m = 2`, no recursive fields. Recursor type:
//!
//! ```text
//! Π {motive : Point2D -> Sort v}.
//!   Π (minor : Π (x:Nat) (y:Nat). motive (Point2D.mk x y)).
//!     Π (major : Point2D).
//!       motive major
//! ```
//!
//! Inside the minor's codomain the context is `[motive, x, y]`:
//! `motive = Var(2)`, `x = Var(1)`, `y = Var(0)`.
//!
//! ## Worked example: `Wrapper`
//!
//! Fields `value : Nat` (non-recursive) and `inner : Wrapper`
//! (recursive): `m = 2`, `O₂ = 1`. Recursor type:
//!
//! ```text
//! Π {motive : Wrapper -> Sort v}.
//!   Π (minor : Π (value:Nat) (inner:Wrapper) (ih : motive inner).
//!                motive (Wrapper.mk value inner)).
//!     Π (major : Wrapper).
//!       motive major
//! ```
//!
//! Inside the minor's codomain the context is
//! `[motive, value, inner, ih]`: `motive = Var(3)`, `value = Var(2)`,
//! `inner = Var(1)`, `ih = Var(0)`. At the IH's domain the context is
//! `[motive, value, inner]`: `motive = Var(2)`, `inner = Var(0)`.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use theoria_kernel::expr::{BinderInfo, DeBruijnIndex, Expr};
use theoria_kernel::level::{Level, UniverseParamId};
use theoria_kernel::name::NameId;
use theoria_kernel::{ConstructorVal, InductiveVal, RecursorRule, RecursorVal};

/// The anonymous-binder name, matching the elaborator's convention for
/// binders with no surface name.
pub(crate) const ANON: NameId = NameId(u32::MAX);

/// Everything needed to build a structure's kernel triple, as produced
/// by [`elaborate_structure`](crate::elaborate::Elaborator::elaborate_structure).
pub(crate) struct StructureShape {
    /// The structure's own `NameId` (`T`).
    pub name: NameId,
    /// The constructor's `NameId` (by convention `T.mk`).
    pub ctor_name: NameId,
    /// The recursor's `NameId` (by convention `T.rec`).
    pub rec_name: NameId,
    /// Parameter binder names, in declaration order.
    pub param_names: Vec<NameId>,
    /// Parameter types; `param_types[i]` lives in the context of the
    /// parameters before it (depth `i`).
    pub param_types: Vec<Expr>,
    /// Field binder names, in declaration order.
    pub field_names: Vec<NameId>,
    /// Field types; each lives in the parameter context `[p₁, ..., pₖ]`.
    pub field_types: Vec<Expr>,
    /// Whether each field is directly recursive (`T p⃗`).
    pub is_recursive: Vec<bool>,
    /// Binder names for the recursor's motive, minor, and major premises.
    pub motive_name: NameId,
    pub minor_name: NameId,
    pub major_name: NameId,
    /// Binder name for induction hypotheses.
    pub ih_name: NameId,
    /// The recursor's universe parameter.
    pub rec_universe: UniverseParamId,
}

impl StructureShape {
    /// `k` — number of parameters.
    fn k(&self) -> usize {
        self.param_types.len()
    }

    /// `m` — number of fields.
    fn m(&self) -> usize {
        self.field_types.len()
    }

    /// `Oⱼ` for 1-indexed `j`: cumulative recursive-field count through
    /// field `j` (inclusive). `cumulative_recursive(0) == 0`.
    fn cumulative_recursive(&self, j: usize) -> usize {
        self.is_recursive[..j].iter().filter(|b| **b).count()
    }

    /// `Oₘ` — total recursive fields.
    fn total_recursive(&self) -> usize {
        self.cumulative_recursive(self.m())
    }
}

/// Build the inductive's type: `Π {p₁:P₁} ... {pₖ:Pₖ}. Sort 1`.
pub(crate) fn inductive_type(shape: &StructureShape) -> Expr {
    let mut ty = Expr::Sort(Level::Zero.succ());
    for i in (0..shape.k()).rev() {
        ty = Expr::Pi(
            BinderInfo::Implicit,
            shape.param_names[i],
            Box::new(shape.param_types[i].clone()),
            Box::new(ty),
        );
    }
    ty
}

/// Build the constructor's type:
/// `Π {p₁:P₁} ... {pₖ:Pₖ} (f₁:F₁) ... (fₘ:Fₘ). T p⃗`.
///
/// Field `Fⱼ` (1-indexed) is written under the `j - 1` earlier fields,
/// so it shifts by `j - 1` out of the parameter context. The codomain
/// `T p⃗` is written under all `m` fields: parameter `pᵢ` (1-indexed)
/// sits at `m + (k - i)`.
pub(crate) fn constructor_type(shape: &StructureShape) -> Expr {
    let k = shape.k();
    let m = shape.m();
    let mut body = Expr::Const(shape.name, Vec::new());
    for i in 1..=k {
        body = Expr::app(body, Expr::Var(DeBruijnIndex((m + (k - i)) as u32)));
    }
    for i in (1..=m).rev() {
        body = Expr::Pi(
            BinderInfo::Default,
            shape.field_names[i - 1],
            Box::new(shift(&shape.field_types[i - 1], i - 1)),
            Box::new(body),
        );
    }
    for i in (0..k).rev() {
        body = Expr::Pi(
            BinderInfo::Implicit,
            shape.param_names[i],
            Box::new(shape.param_types[i].clone()),
            Box::new(body),
        );
    }
    body
}

/// Build the recursor's type. See the module header for the layout.
///
/// All field indices below are 1-indexed (`j = 1..=m`); `Oⱼ` is
/// [`StructureShape::cumulative_recursive`].
pub(crate) fn recursor_type(shape: &StructureShape) -> Expr {
    let v = shape.rec_universe;
    let k = shape.k();
    let m = shape.m();
    let total_ih = shape.total_recursive();

    // --- Minor codomain: `motive (T.mk p⃗ f⃗)`. ---
    //
    // Written under all `m + Oₘ` minor-internal binders (motive and minor
    // are *not* in scope inside the minor's own type):
    // * `motive = Var(m + Oₘ)`,
    // * field `fⱼ = Var((m - j) + (Oₘ - Oⱼ₋₁))` (binders strictly after
    //   it: later fields plus later IHs *including its own IH*),
    // * parameter `pᵢ = Var(m + Oₘ + 1 + (k - i))`.
    let mut mk_app = Expr::Const(shape.ctor_name, Vec::new());
    for i in 1..=k {
        mk_app = Expr::app(
            mk_app,
            Expr::Var(DeBruijnIndex((m + total_ih + 1 + (k - i)) as u32)),
        );
    }
    for j in 1..=m {
        let after = (m - j) + (total_ih - shape.cumulative_recursive(j - 1));
        mk_app = Expr::app(mk_app, Expr::Var(DeBruijnIndex(after as u32)));
    }
    let mut minor_ty = Expr::app(Expr::Var(DeBruijnIndex((m + total_ih) as u32)), mk_app);

    // --- Minor Pi chain, built inside-out (rightmost binder first). ---
    //
    // Field `fⱼ`'s domain is written under the motive binder plus the
    // `(j - 1) + Oⱼ₋₁` earlier minor binders: `Fⱼ` shifts by
    // `1 + (j - 1) + Oⱼ₋₁` out of the parameter context.
    //
    // The IH domain `motive fⱼ` is written just under `fⱼ`: `fⱼ = Var(0)`
    // and `motive = Var(1 + (j - 1) + Oⱼ₋₁)` (the earlier minor binders
    // plus the motive itself — the `minor` binder is not in scope here).
    for j in (1..=m).rev() {
        let before = (j - 1) + shape.cumulative_recursive(j - 1);
        if shape.is_recursive[j - 1] {
            let ih_ty = Expr::app(
                Expr::Var(DeBruijnIndex((1 + before) as u32)),
                Expr::Var(DeBruijnIndex(0)),
            );
            minor_ty = Expr::Pi(
                BinderInfo::Default,
                shape.ih_name,
                Box::new(ih_ty),
                Box::new(minor_ty),
            );
        }
        minor_ty = Expr::Pi(
            BinderInfo::Default,
            shape.field_names[j - 1],
            Box::new(shift(&shape.field_types[j - 1], 1 + before)),
            Box::new(minor_ty),
        );
    }

    // --- Motive type: `Π (_ : T p⃗). Sort v`, at depth `k`. ---
    let mut motive_dom = Expr::Const(shape.name, Vec::new());
    for i in 1..=k {
        motive_dom = Expr::app(motive_dom, Expr::Var(DeBruijnIndex((k - i) as u32)));
    }
    let motive_ty = Expr::Pi(
        BinderInfo::Default,
        ANON,
        Box::new(motive_dom),
        Box::new(Expr::Sort(Level::param(v))),
    );

    // --- Major premise: `Π (major : T p⃗). motive major`, at depth
    // --- `k + 2` (params, motive, minor). `T p⃗` shifts by 2; the
    // --- codomain is `App(Var(2), Var(0))`.
    let mut major_dom = Expr::Const(shape.name, Vec::new());
    for i in 1..=k {
        major_dom = Expr::app(major_dom, Expr::Var(DeBruijnIndex((2 + (k - i)) as u32)));
    }
    let major_ty = Expr::Pi(
        BinderInfo::Default,
        shape.major_name,
        Box::new(major_dom),
        Box::new(Expr::app(
            Expr::Var(DeBruijnIndex(2)),
            Expr::Var(DeBruijnIndex(0)),
        )),
    );

    // --- Assemble: params, motive (implicit), minor, major. ---
    let mut acc = major_ty;
    acc = Expr::Pi(
        BinderInfo::Default,
        shape.minor_name,
        Box::new(minor_ty),
        Box::new(acc),
    );
    acc = Expr::Pi(
        BinderInfo::Implicit,
        shape.motive_name,
        Box::new(motive_ty),
        Box::new(acc),
    );
    for i in (0..k).rev() {
        acc = Expr::Pi(
            BinderInfo::Implicit,
            shape.param_names[i],
            Box::new(shape.param_types[i].clone()),
            Box::new(acc),
        );
    }
    acc
}

/// Build the recursor's single ι-reduction rule.
///
/// Rule context (parameters outermost, fields innermost):
/// field `fⱼ` (1-indexed) `= Var(m - j)`, `minor = Var(m)`,
/// `motive = Var(m + 1)`, parameter `pᵢ` (1-indexed)
/// `= Var(m + 2 + (k - i))`.
///
/// RHS `= minor f₁ [ih₁] ... fₘ [ihₘ]` with
/// `ihⱼ = T.rec.{v} p⃗ motive minor fⱼ`.
pub(crate) fn recursor_rule(shape: &StructureShape) -> RecursorRule {
    let k = shape.k();
    let m = shape.m();
    let minor = Expr::Var(DeBruijnIndex(m as u32));
    let mut args: Vec<Expr> = Vec::with_capacity(m + shape.total_recursive());
    for j in 1..=m {
        let f_j = Expr::Var(DeBruijnIndex((m - j) as u32));
        args.push(f_j.clone());
        if shape.is_recursive[j - 1] {
            let mut call: Vec<Expr> = Vec::with_capacity(k + 3);
            for i in 1..=k {
                call.push(Expr::Var(DeBruijnIndex((m + 2 + (k - i)) as u32)));
            }
            call.push(Expr::Var(DeBruijnIndex((m + 1) as u32)));
            call.push(Expr::Var(DeBruijnIndex(m as u32)));
            call.push(f_j);
            let head = Expr::Const(
                shape.rec_name,
                alloc::vec![Level::param(shape.rec_universe)],
            );
            args.push(head.apps(call));
        }
    }
    RecursorRule {
        constructor: shape.ctor_name,
        num_fields: m as u32,
        rhs: Rc::new(minor.apps(args)),
    }
}

/// Build the type of the `i`-th projection (0-indexed):
/// `Π {p⃗} (s : T p⃗). Fᵢ`, with `Fᵢ` shifted by 1 under the receiver.
pub(crate) fn projection_type(shape: &StructureShape, i: usize) -> Expr {
    let k = shape.k();
    let mut receiver_dom = Expr::Const(shape.name, Vec::new());
    for p in 1..=k {
        receiver_dom = Expr::app(receiver_dom, Expr::Var(DeBruijnIndex((k - p) as u32)));
    }
    let mut acc = Expr::Pi(
        BinderInfo::Default,
        ANON,
        Box::new(receiver_dom),
        Box::new(shift(&shape.field_types[i], 1)),
    );
    for p in (0..k).rev() {
        acc = Expr::Pi(
            BinderInfo::Implicit,
            shape.param_names[p],
            Box::new(shape.param_types[p].clone()),
            Box::new(acc),
        );
    }
    acc
}

/// Build the body of the `i`-th projection (0-indexed):
/// `λ {p⃗} (s : T p⃗). T.rec.{1} p⃗ motive minor s`.
///
/// The motive is `λ (_ : T p⃗). Fᵢ` (constant in the scrutinee, as in
/// delivery 8). The minor mirrors the recursor's minor-type chain as a
/// lambda chain returning `fᵢ`; its IH domains use the reduced form
/// `Fᵢ` rather than `motive fⱼ` because the kernel infers every lambda
/// domain and an application headed by a lambda is not inferable.
///
/// The minor term sits at depth `k + 1` (params + receiver), so its free
/// parameter references carry a base offset of 1: field `fⱼ`'s
/// (1-indexed) domain is `Fⱼ` shifted by `1 + (j - 1) + Oⱼ₋₁`, IH `ihⱼ`'s
/// domain is `Fᵢ` shifted by `1 + j + Oⱼ₋₁ = j + Oⱼ`, and the body is
/// `fᵢ = Var((m - i') + (Oₘ - Oᵢ'₋₁))` with `i' = i + 1` (binders
/// strictly after it, including its own IH).
pub(crate) fn projection_body(shape: &StructureShape, i: usize) -> Expr {
    let k = shape.k();
    let m = shape.m();
    let field_1indexed = i + 1;

    // Minor term, built inside-out. `depth` counts minor-internal
    // binders bound so far (fields + IHs strictly inside the current
    // position).
    let after_i = (m - field_1indexed) + (shape.total_recursive() - shape.cumulative_recursive(i));
    let mut minor = Expr::Var(DeBruijnIndex(after_i as u32));
    for j in (1..=m).rev() {
        let before = (j - 1) + shape.cumulative_recursive(j - 1);
        if shape.is_recursive[j - 1] {
            // IH domain: reduced `motive fⱼ` = `Fᵢ`, shifted from the
            // parameter context to the current scope (outer `k + 1`
            // binders plus the `j + Oⱼ₋₁` minor binders bound so far,
            // including `fⱼ` itself).
            let ih_ty = shift(
                &shape.field_types[i],
                1 + j + shape.cumulative_recursive(j - 1),
            );
            minor = Expr::Lam(
                BinderInfo::Default,
                shape.ih_name,
                Box::new(ih_ty),
                Box::new(minor),
            );
        }
        let field_ty = shift(&shape.field_types[j - 1], 1 + before);
        minor = Expr::Lam(
            BinderInfo::Default,
            shape.field_names[j - 1],
            Box::new(field_ty),
            Box::new(minor),
        );
    }

    // Motive term at depth `k + 1`: domain `T p⃗` with parameters shifted
    // by 1, body `Fᵢ` shifted by 2 (motive binder + base offset).
    let mut motive_dom = Expr::Const(shape.name, Vec::new());
    for p in 1..=k {
        motive_dom = Expr::app(motive_dom, Expr::Var(DeBruijnIndex((1 + (k - p)) as u32)));
    }
    let motive = Expr::Lam(
        BinderInfo::Default,
        ANON,
        Box::new(motive_dom),
        Box::new(shift(&shape.field_types[i], 2)),
    );

    // `T.rec.{1} p⃗ motive minor s` at depth `k + 1`: `s = Var(0)`,
    // parameters at `1 + (k - p)`.
    let mut call: Vec<Expr> = Vec::with_capacity(k + 3);
    for p in 1..=k {
        call.push(Expr::Var(DeBruijnIndex((1 + (k - p)) as u32)));
    }
    // NOTE: motive and minor are *terms* applied here, so they are
    // spliced in directly (their internal indices were built for depth
    // `k + 1`, which is exactly where they are used).
    call.push(motive);
    call.push(minor);
    call.push(Expr::Var(DeBruijnIndex(0)));
    let head = Expr::Const(shape.rec_name, alloc::vec![Level::Zero.succ()]);
    let mut body = head.apps(call);

    // Wrap the receiver binder (domain `T p⃗` at depth `k`).
    let mut receiver_dom = Expr::Const(shape.name, Vec::new());
    for p in 1..=k {
        receiver_dom = Expr::app(receiver_dom, Expr::Var(DeBruijnIndex((k - p) as u32)));
    }
    body = Expr::Lam(
        BinderInfo::Default,
        ANON,
        Box::new(receiver_dom),
        Box::new(body),
    );

    // Wrap the parameter binders (reverse order, no shifting: each
    // domain was elaborated in the context of the parameters before it).
    for p in (0..k).rev() {
        body = Expr::Lam(
            BinderInfo::Implicit,
            shape.param_names[p],
            Box::new(shape.param_types[p].clone()),
            Box::new(body),
        );
    }
    body
}

/// Assemble the kernel triple. The caller inserts it with
/// [`GlobalEnv::add_inductive`], which runs the positivity gate.
pub(crate) fn build_triple(
    shape: &StructureShape,
) -> (InductiveVal, Vec<ConstructorVal>, RecursorVal) {
    use theoria_kernel::ConstantVal;
    let k = shape.k();
    let m = shape.m();
    let is_recursive = shape.is_recursive.iter().any(|b| *b);
    let ind = InductiveVal {
        base: ConstantVal {
            name: shape.name,
            universe_params: Vec::new(),
            ty: Rc::new(inductive_type(shape)),
        },
        num_params: k as u32,
        num_indices: 0,
        all: alloc::vec![shape.name],
        constructors: alloc::vec![shape.ctor_name],
        is_recursive,
        is_nested: false,
        is_unsafe: false,
    };
    let ctor = ConstructorVal {
        base: ConstantVal {
            name: shape.ctor_name,
            universe_params: Vec::new(),
            ty: Rc::new(constructor_type(shape)),
        },
        inductive: shape.name,
        index: 0,
        num_params: k as u32,
        num_fields: m as u32,
        is_unsafe: false,
    };
    let rec = RecursorVal {
        base: ConstantVal {
            name: shape.rec_name,
            universe_params: alloc::vec![shape.rec_universe],
            ty: Rc::new(recursor_type(shape)),
        },
        all: alloc::vec![shape.name],
        num_params: k as u32,
        num_indices: 0,
        num_motives: 1,
        num_minors: 1,
        rules: alloc::vec![recursor_rule(shape)],
        is_k: false,
        is_unsafe: false,
    };
    (ind, alloc::vec![ctor], rec)
}

// ---------------------------------------------------------------------------
// Recursion classification
// ---------------------------------------------------------------------------

/// `true` iff `ty` is exactly `T` applied to the `k` parameter variables
/// (`T p₁ ... pₖ` with `pᵢ = Var(k - i)`), i.e. a directly recursive
/// field type.
pub(crate) fn is_direct_recursion(ty: &Expr, t: NameId, k: usize) -> bool {
    let mut cur = ty;
    let mut args: Vec<&Expr> = Vec::new();
    while let Expr::App(f, a) = cur {
        args.push(a);
        cur = f;
    }
    let Expr::Const(head, _) = cur else {
        return false;
    };
    if *head != t || args.len() != k {
        return false;
    }
    // `args` collected outside-in; `args[0]` is the last argument.
    for (pos, a) in args.iter().enumerate() {
        // Last argument (pos 0) is `pₖ = Var(0)`; first (pos k-1) is
        // `p₁ = Var(k-1)`.
        let Expr::Var(idx) = a else {
            return false;
        };
        if idx.0 as usize != pos {
            return false;
        }
    }
    true
}

/// `true` iff `T` occurs anywhere in `ty`.
pub(crate) fn mentions(ty: &Expr, t: NameId) -> bool {
    match ty {
        Expr::Sort(_) | Expr::Var(_) | Expr::Lit(_) => false,
        Expr::Const(name, _) => *name == t,
        Expr::App(f, a) => mentions(f, t) || mentions(a, t),
        Expr::Lam(_, _, d, b) | Expr::Pi(_, _, d, b) => mentions(d, t) || mentions(b, t),
        Expr::Let(_, ty, val, body) => mentions(ty, t) || mentions(val, t) || mentions(body, t),
    }
}

/// `true` iff `T` occurs in the domain of any `Pi` inside `ty` (a
/// negative position).
pub(crate) fn occurs_in_pi_domain(ty: &Expr, t: NameId) -> bool {
    match ty {
        Expr::Sort(_) | Expr::Var(_) | Expr::Lit(_) | Expr::Const(..) => false,
        Expr::App(f, a) => occurs_in_pi_domain(f, t) || occurs_in_pi_domain(a, t),
        Expr::Lam(_, _, d, b) => occurs_in_pi_domain(d, t) || occurs_in_pi_domain(b, t),
        Expr::Pi(_, _, d, b) => mentions(d, t) || occurs_in_pi_domain(b, t),
        Expr::Let(_, ty, val, body) => {
            occurs_in_pi_domain(ty, t)
                || occurs_in_pi_domain(val, t)
                || occurs_in_pi_domain(body, t)
        }
    }
}

/// Instantiate parameter variables in a field type with actual terms:
/// replace `Var(cutoff + (k - 1 - p))` (parameter `p`, 0-indexed) with
/// `args[p]` lifted under `cutoff` binders. Variables below `cutoff`
/// are bound inside the expression and untouched; any other free
/// variable is left as-is (field types are elaborated at top level, so
/// only parameter variables occur free).
pub(crate) fn instantiate_params(ty: &Expr, args: &[Expr]) -> Expr {
    instantiate_rec(ty, args, 0)
}

fn instantiate_rec(ty: &Expr, args: &[Expr], cutoff: u32) -> Expr {
    let k = args.len() as u32;
    match ty {
        Expr::Sort(l) => Expr::Sort(l.clone()),
        Expr::Lit(l) => Expr::Lit(l.clone()),
        Expr::Const(name, lvls) => Expr::Const(*name, lvls.clone()),
        Expr::Var(i) => {
            if i.0 < cutoff {
                Expr::Var(*i)
            } else {
                let rel = i.0 - cutoff;
                if rel < k {
                    // Parameter `p` (0-indexed) is `Var(k - 1 - p)`; invert.
                    let p = (k - 1 - rel) as usize;
                    shift(&args[p], cutoff as usize)
                } else {
                    Expr::Var(*i)
                }
            }
        }
        Expr::App(f, a) => Expr::app(
            instantiate_rec(f, args, cutoff),
            instantiate_rec(a, args, cutoff),
        ),
        Expr::Lam(info, name, d, b) => Expr::Lam(
            *info,
            *name,
            Box::new(instantiate_rec(d, args, cutoff)),
            Box::new(instantiate_rec(b, args, cutoff + 1)),
        ),
        Expr::Pi(info, name, d, b) => Expr::Pi(
            *info,
            *name,
            Box::new(instantiate_rec(d, args, cutoff)),
            Box::new(instantiate_rec(b, args, cutoff + 1)),
        ),
        Expr::Let(name, t, v, b) => Expr::Let(
            *name,
            Box::new(instantiate_rec(t, args, cutoff)),
            Box::new(instantiate_rec(v, args, cutoff)),
            Box::new(instantiate_rec(b, args, cutoff + 1)),
        ),
    }
}

// ---------------------------------------------------------------------------
// Shifting
// ---------------------------------------------------------------------------

/// Shift every free de Bruijn variable in `e` up by `by` (variables
/// bound inside `e` are untouched). Lifts an expression from one binder
/// depth to a deeper one; shifting a closed term is a no-op.
pub(crate) fn shift(e: &Expr, by: usize) -> Expr {
    if by == 0 {
        return e.clone();
    }
    shift_rec(e, by as u32, 0)
}

fn shift_rec(e: &Expr, by: u32, cutoff: u32) -> Expr {
    match e {
        Expr::Sort(l) => Expr::Sort(l.clone()),
        Expr::Var(i) => {
            if i.0 >= cutoff {
                Expr::Var(DeBruijnIndex(i.0 + by))
            } else {
                Expr::Var(*i)
            }
        }
        Expr::Const(name, lvls) => Expr::Const(*name, lvls.clone()),
        Expr::App(f, a) => Expr::app(shift_rec(f, by, cutoff), shift_rec(a, by, cutoff)),
        Expr::Lam(info, name, d, b) => Expr::Lam(
            *info,
            *name,
            Box::new(shift_rec(d, by, cutoff)),
            Box::new(shift_rec(b, by, cutoff + 1)),
        ),
        Expr::Pi(info, name, d, b) => Expr::Pi(
            *info,
            *name,
            Box::new(shift_rec(d, by, cutoff)),
            Box::new(shift_rec(b, by, cutoff + 1)),
        ),
        Expr::Let(name, t, v, b) => Expr::Let(
            *name,
            Box::new(shift_rec(t, by, cutoff)),
            Box::new(shift_rec(v, by, cutoff)),
            Box::new(shift_rec(b, by, cutoff + 1)),
        ),
        Expr::Lit(l) => Expr::Lit(l.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use theoria_kernel::name::NameTable;

    fn point_shape(names: &mut NameTable) -> StructureShape {
        let nat = names.intern("Nat");
        StructureShape {
            name: names.intern("Point2D"),
            ctor_name: names.intern("Point2D.mk"),
            rec_name: names.intern("Point2D.rec"),
            param_names: Vec::new(),
            param_types: Vec::new(),
            field_names: alloc::vec![names.intern("x"), names.intern("y")],
            field_types: alloc::vec![Expr::Const(nat, Vec::new()), Expr::Const(nat, Vec::new()),],
            is_recursive: alloc::vec![false, false],
            motive_name: names.intern("motive"),
            minor_name: names.intern("minor"),
            major_name: names.intern("major"),
            ih_name: names.intern("ih"),
            rec_universe: UniverseParamId::fresh(),
        }
    }

    #[test]
    fn point_recursor_type_has_expected_shape() {
        // Pins the exact de Bruijn layout: any index slip changes these
        // assertions (the kernel check in the elaborator tests would also
        // fail, but this names the expected term directly).
        let mut names = NameTable::new();
        let shape = point_shape(&mut names);
        let nat = names.intern("Nat");
        let nat_ty = Expr::Const(nat, Vec::new());
        let t_ty = Expr::Const(shape.name, Vec::new());

        let Expr::Pi(BinderInfo::Implicit, _, motive_ty, rest) = recursor_type(&shape) else {
            panic!("recursor type must start with the implicit motive binder");
        };
        // Motive type: `Π (_ : Point2D). Sort v`.
        let Expr::Pi(BinderInfo::Default, _, motive_dom, motive_cod) = motive_ty.as_ref() else {
            panic!("motive type must be a Pi, got {motive_ty:?}");
        };
        assert_eq!(**motive_dom, t_ty);
        assert!(matches!(**motive_cod, Expr::Sort(_)));

        // Minor premise.
        let Expr::Pi(BinderInfo::Default, _, minor_ty, rest2) = rest.as_ref() else {
            panic!("expected the minor premise, got {rest:?}");
        };
        // Minor chain: `Π (x:Nat) (y:Nat). motive (mk x y)` with
        // `motive = Var(2)`, `x = Var(1)`, `y = Var(0)`.
        let Expr::Pi(BinderInfo::Default, _, x_dom, rest3) = minor_ty.as_ref() else {
            panic!("expected the x binder, got {minor_ty:?}");
        };
        assert_eq!(**x_dom, nat_ty);
        let Expr::Pi(BinderInfo::Default, _, y_dom, minor_cod) = rest3.as_ref() else {
            panic!("expected the y binder, got {rest3:?}");
        };
        assert_eq!(**y_dom, nat_ty);
        let mk_app = Expr::app(
            Expr::app(
                Expr::Const(shape.ctor_name, Vec::new()),
                Expr::Var(DeBruijnIndex(1)),
            ),
            Expr::Var(DeBruijnIndex(0)),
        );
        assert_eq!(**minor_cod, Expr::app(Expr::Var(DeBruijnIndex(2)), mk_app));

        // Major premise: `Π (major : Point2D). motive major`.
        let Expr::Pi(BinderInfo::Default, _, major_dom, major_cod) = rest2.as_ref() else {
            panic!("expected the major premise, got {rest2:?}");
        };
        assert_eq!(**major_dom, t_ty);
        assert_eq!(
            **major_cod,
            Expr::app(Expr::Var(DeBruijnIndex(2)), Expr::Var(DeBruijnIndex(0)))
        );
    }

    #[test]
    fn point_recursor_rule_applies_minor_to_fields() {
        let mut names = NameTable::new();
        let shape = point_shape(&mut names);
        let rule = recursor_rule(&shape);
        assert_eq!(rule.constructor, shape.ctor_name);
        assert_eq!(rule.num_fields, 2);
        // `minor x y` with `minor = Var(2)`, `x = Var(1)`, `y = Var(0)`.
        assert_eq!(
            *rule.rhs,
            Expr::app(
                Expr::app(Expr::Var(DeBruijnIndex(2)), Expr::Var(DeBruijnIndex(1))),
                Expr::Var(DeBruijnIndex(0)),
            )
        );
    }

    #[test]
    fn direct_recursion_matches_applied_params() {
        let mut names = NameTable::new();
        let stream = names.intern("Stream");
        let a = Expr::Var(DeBruijnIndex(0));
        // `Stream A` with one parameter: direct recursion.
        assert!(is_direct_recursion(
            &Expr::app(Expr::Const(stream, Vec::new()), a),
            stream,
            1
        ));
        // Bare `Stream` (missing the argument): not direct.
        assert!(!is_direct_recursion(
            &Expr::Const(stream, Vec::new()),
            stream,
            1
        ));
        // `Stream Nat`: instantiated, not direct.
        let nat = Expr::Const(names.intern("Nat"), Vec::new());
        assert!(!is_direct_recursion(
            &Expr::app(Expr::Const(stream, Vec::new()), nat),
            stream,
            1
        ));
    }
}
