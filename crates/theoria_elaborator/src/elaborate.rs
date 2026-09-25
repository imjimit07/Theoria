//! Surface-to-kernel elaboration.
//!
//! The elaborator resolves surface identifiers (locals first, then
//! globals), desugars `Nat` literals into successor chains, assembles
//! `Function` declarations into kernel `Pi`/`Lam` chains wrapped in a
//! [`ConstantInfo::Definition`],
//! and hands each declaration to the kernel for checking before
//! insertion.
//!
//! The kernel is the authority on types: the elaborator never duplicates
//! a typing rule. Anything the elaborator produces that is ill-typed
//! comes back as [`ElaborateErrorKind::Kernel`].

use crate::error::{ElaborateError, ElaborateErrorKind};
use crate::scope::{LocalBinding, Scope};
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use theoria_kernel::Expr as KernelExpr;
use theoria_kernel::env::{ConstantInfo, ConstantVal, GlobalEnv, LocalDecl};
use theoria_kernel::infer::{TypeChecker, TypedContext};
use theoria_kernel::nbe::{Env as NbeEnv, Nbe};
use theoria_kernel::{
    BinderInfo, DbLevel, DeBruijnIndex, DefinitionVal, EnvError, InductiveVal, Level, NameId,
    NameTable, TerminationObligation, TheoremVal, Transparency, TransparencyMode, UniverseParamId,
    Value,
};
use theoria_syntax::{self, Module, Span};
use theoria_units::Exponents;
use theoria_units::UnitError;
use theoria_units::{UnitRegistry, parse_unit};

/// Extract a literal natural number from a surface expression.
///
/// Used by `unit_of_surface` to recover the exponent of `^` from the
/// surface syntax before the exponent has been elaborated.
///
/// Returns `None` for anything that is not a parenthesised-or-bare `Nat`
/// literal. Non-literal exponents make the enclosing unit expression
/// unclassifiable; the caller decides whether that is an error or a
/// silent `Bare`.
fn surface_nat_literal(e: &theoria_syntax::Expr) -> Option<i32> {
    match e {
        theoria_syntax::Expr::Nat { value, .. } => i32::try_from(*value).ok(),
        theoria_syntax::Expr::Paren { inner, .. } => surface_nat_literal(inner),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// The result of elaborating a module.
///
/// `Clone` is intentionally absent: neither [`GlobalEnv`] nor
/// [`NameTable`] is cloneable.
#[derive(Debug)]
pub struct ElaboratedModule {
    /// The parsed `Module` declaration, if any.
    pub module_name: Option<theoria_syntax::Path>,
    /// The environment after all functions have been elaborated and
    /// inserted. Contains the prelude plus every user-defined function.
    pub env: GlobalEnv,
    /// The name table used to intern identifiers. Shared with the prelude.
    pub names: NameTable,
    /// One entry per user-defined structure, in declaration order.
    pub structures: Vec<ElaboratedStructure>,
    /// One entry per user-defined function, in declaration order.
    pub functions: Vec<ElaboratedFunction>,
    /// One entry per user-defined theorem, in declaration order.
    pub theorems: Vec<ElaboratedTheorem>,
}

/// Summary of one elaborated structure.
#[derive(Clone, Debug)]
pub struct ElaboratedStructure {
    /// The identifier the structure was given in the source.
    pub source_name: String,
    /// The kernel `NameId` of the inductive type former.
    pub kernel_name: NameId,
    /// Number of parameters.
    pub num_params: usize,
    /// Number of fields.
    pub num_fields: usize,
    /// Span of the entire `Structure` declaration.
    pub span: Span,
}

/// Summary of one elaborated function.
#[derive(Clone, Debug)]
pub struct ElaboratedFunction {
    /// The identifier the function was given in the source.
    pub source_name: String,
    /// The kernel `NameId` it received in the environment.
    pub kernel_name: NameId,
    /// Number of parameters.
    pub arity: usize,
    /// The span of the entire `Function` declaration.
    pub span: Span,
}

/// Summary of one elaborated theorem.
#[derive(Clone, Debug)]
pub struct ElaboratedTheorem {
    /// The theorem's source name.
    pub source_name: String,
    /// The kernel `NameId` the theorem was registered under.
    pub kernel_name: NameId,
    /// Number of `Given:` binders.
    pub num_given: usize,
    /// Number of `Assume:` hypotheses.
    pub num_assume: usize,
    /// Span of the entire `Theorem` declaration.
    pub span: Span,
}

/// Elaborate a module against the standard prelude.
///
/// On success, returns the elaborated module and the augmented
/// environment.
///
/// On failure, returns one or more errors. This delivery returns at most
/// one error (fail-fast); the `Vec` return type commits the interface to
/// multi-error recovery in a later delivery without a breaking change.
pub fn elaborate_module(module: &Module) -> Result<ElaboratedModule, Vec<ElaborateError>> {
    let prelude = theoria_kernel::prelude::build_prelude();
    elaborate_module_with_env(module, prelude.env, prelude.names)
}

/// Elaborate a module against a caller-supplied environment.
///
/// Useful for tests that want to start from a custom prelude (e.g. one
/// without quotients, or with a mock inductive), and for the interactive
/// REPL that will carry an environment across input lines.
pub fn elaborate_module_with_env(
    module: &Module,
    env: GlobalEnv,
    names: NameTable,
) -> Result<ElaboratedModule, Vec<ElaborateError>> {
    let mut elb = Elaborator::new(env, names);
    for import in &module.imports {
        elb.elaborate_import(import).map_err(|e| alloc::vec![e])?;
    }
    // Pre-flight: verify that every structure's generated names are free
    // before elaborating any of them. This catches collisions with the
    // prelude or with earlier declarations in the same module before any
    // declaration is inserted, so a failed module leaves no partial
    // state in the environment. The check also rejects two structures in
    // the same module that share a name.
    {
        let mut seen: alloc::collections::BTreeSet<NameId> = alloc::collections::BTreeSet::new();
        for s in &module.structures {
            let id = elb.names.intern(&s.name);
            if !seen.insert(id) {
                return Err(alloc::vec![ElaborateError::new(
                    s.name_span,
                    ElaborateErrorKind::DuplicateStructure,
                    format!("`{}` is declared twice in this module", s.name),
                )]);
            }
            elb.check_structure_names_free(s)
                .map_err(|e| alloc::vec![e])?;
        }
    }
    // Structures precede functions: a function may project out of a
    // structure defined earlier in the file, but never vice versa.
    // (Forward references are rejected with `UnknownIdentifier`,
    // matching delivery 2's policy.)
    let mut structures = Vec::new();
    for s in &module.structures {
        structures.push(elb.elaborate_structure(s).map_err(|e| alloc::vec![e])?);
    }
    let mut functions = Vec::new();
    for f in &module.functions {
        functions.push(elb.elaborate_function(f).map_err(|e| alloc::vec![e])?);
    }
    // Theorems come after functions and structures: a theorem may
    // reference any function or structure declared earlier in the same
    // module, but a function cannot reference a theorem (forward
    // references are rejected per delivery 2's policy).
    let mut theorems = Vec::new();
    for t in &module.theorems {
        theorems.push(elb.elaborate_theorem(t).map_err(|e| alloc::vec![e])?);
    }
    Ok(ElaboratedModule {
        module_name: module.decl.as_ref().map(|d| d.path.clone()),
        env: elb.env,
        names: elb.names,
        structures,
        functions,
        theorems,
    })
}

// ---------------------------------------------------------------------------
// Elaborator
// ---------------------------------------------------------------------------

/// The elaboration state: the environment under construction and the
/// shared name table.
struct Elaborator {
    env: GlobalEnv,
    names: NameTable,
    /// The inductive being defined, if any. While set, an identifier
    /// resolving to this name elaborates to the type former (used for
    /// direct-recursion field types; the inductive is not yet in the
    /// environment).
    self_type: Option<NameId>,
    /// User-defined structures by inductive `NameId`, for field-access
    /// resolution.
    struct_records: BTreeMap<NameId, StructureRecord>,
    /// Tracks the kernel constants backing the unit system, and interns
    /// canonical unit constants on demand. Installed once at
    /// construction.
    units: UnitRegistry,
}

/// One elaborated structure, as remembered for field access.
#[derive(Clone, Debug)]
struct StructureRecord {
    /// The structure's source name (for diagnostics).
    source_name: String,
    /// Number of parameters.
    num_params: usize,
    /// One entry per field, in declaration order.
    fields: Vec<StructureField>,
}

/// One field of an elaborated structure.
#[derive(Clone, Debug)]
struct StructureField {
    /// The field's source name.
    name: String,
    /// The projection function's kernel `NameId` (`T.field`).
    proj: NameId,
    /// The field's type in the parameter context `[p₁, ..., pₖ]`.
    ty: KernelExpr,
}

impl Elaborator {
    /// Fresh elaboration state over an existing environment.
    ///
    /// Installs the unit system's kernel declarations (`Unit`,
    /// `Quantity`) if they are not already present. This is idempotent
    /// across calls with the same environment, so an interactive session
    /// that reuses a `GlobalEnv` will not hit a duplicate-declaration
    /// error.
    fn new(env: GlobalEnv, names: NameTable) -> Self {
        let mut elb = Elaborator {
            env,
            names,
            self_type: None,
            struct_records: BTreeMap::new(),
            units: UnitRegistry::new(),
        };
        elb.units
            .install(&mut elb.env, &mut elb.names)
            .expect("unit declarations failed to install");
        elb
    }
}

/// A resolved inductive family, as seen by `match` elaboration.
struct InductiveInfo {
    /// The inductive's kernel name.
    name: NameId,
    /// Number of constructors.
    num_ctors: usize,
    /// Constructor names, in declaration order.
    ctors: Vec<NameId>,
    /// Declared field count per constructor.
    fields: BTreeMap<NameId, usize>,
    /// The dotted source name (e.g. `"Nat"`), for building `"Nat.rec"`.
    source_name: String,
    /// Span of the scrutinee (used for exhaustiveness errors).
    span: Span,
}

impl InductiveInfo {
    /// Build the view from a kernel inductive declaration.
    fn from(v: &InductiveVal, env: &GlobalEnv, names: &NameTable, span: Span) -> Self {
        let mut fields = BTreeMap::new();
        for ctor in &v.constructors {
            let n = env
                .find(*ctor)
                .and_then(|info| match &*info {
                    ConstantInfo::Constructor(c) if c.inductive == v.base.name => {
                        Some(c.num_fields as usize)
                    }
                    _ => None,
                })
                .unwrap_or(0);
            fields.insert(*ctor, n);
        }
        InductiveInfo {
            name: v.base.name,
            num_ctors: v.constructors.len(),
            ctors: v.constructors.clone(),
            fields,
            source_name: names.resolve(v.base.name).to_string(),
            span,
        }
    }

    /// Declared field count of a constructor (0 if unknown).
    fn field_arity(&self, ctor: NameId) -> usize {
        self.fields.get(&ctor).copied().unwrap_or(0)
    }
}

/// One validated `match` arm: the constructor it covers (`None` for a
/// catch-all) plus the source pattern and body.
struct ValidatedArm<'a> {
    /// The covered constructor, or `None` for a catch-all.
    ctor: Option<NameId>,
    /// The source pattern (for bound-variable extraction).
    pattern: &'a theoria_syntax::Pattern,
    /// The arm's body.
    body: &'a theoria_syntax::Expr,
}

/// The unit-carrying status of a surface expression.
///
/// The elaborator uses this to enforce dimensional correctness at the
/// source level. Two classes are recognised:
///
/// * [`Bare`](UnitClass::Bare) — a plain value, a `Nat`, a `Bool`, or
///   anything else that is not a quantity. Bare values coerce freely to
///   any unit for the purposes of `+`, `-`, and the comparisons (so
///   `r + 1` where `r : Quantity(m)` is accepted).
/// * [`Quantity`](UnitClass::Quantity) — a value whose declared type is
///   `Quantity(u)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnitClass {
    /// A plain, non-quantity value.
    Bare,
    /// A value of type `Quantity(u)`.
    Quantity(Exponents),
}

impl UnitClass {
    /// The exponent vector, if this is a quantity.
    #[must_use]
    fn exponents(self) -> Option<Exponents> {
        match self {
            UnitClass::Bare => None,
            UnitClass::Quantity(e) => Some(e),
        }
    }
}

/// Peel `App(f, a)` applications down to `(head, argument count)`.
fn app_head_count(e: &KernelExpr) -> (&KernelExpr, usize) {
    let mut cur = e;
    let mut nargs = 0usize;
    while let KernelExpr::App(f, _) = cur {
        cur = f.as_ref();
        nargs += 1;
    }
    (cur, nargs)
}

/// Shift every free de Bruijn variable in `expr` up by `amount`.
/// Variables bound inside `expr` (indices below `cutoff`) are left
/// untouched; `cutoff` grows by one under each binder.
///
/// Needed when embedding an already-elaborated term under additional
/// binders it was not elaborated under: a `match`-arm body under the
/// recursor minor's induction-hypothesis lambdas, or the motive's
/// codomain under the motive binder. Shifting a closed term is a
/// no-op, so callers need no closedness pre-check.
fn shift_vars(expr: &KernelExpr, amount: u32, cutoff: u32) -> KernelExpr {
    match expr {
        KernelExpr::Sort(_) | KernelExpr::Const(..) | KernelExpr::Lit(_) => expr.clone(),
        KernelExpr::Var(i) => {
            if i.0 >= cutoff {
                KernelExpr::Var(DeBruijnIndex(i.0 + amount))
            } else {
                expr.clone()
            }
        }
        KernelExpr::App(f, a) => KernelExpr::App(
            Box::new(shift_vars(f, amount, cutoff)),
            Box::new(shift_vars(a, amount, cutoff)),
        ),
        KernelExpr::Lam(bi, n, d, b) => KernelExpr::Lam(
            *bi,
            *n,
            Box::new(shift_vars(d, amount, cutoff)),
            Box::new(shift_vars(b, amount, cutoff + 1)),
        ),
        KernelExpr::Pi(bi, n, d, b) => KernelExpr::Pi(
            *bi,
            *n,
            Box::new(shift_vars(d, amount, cutoff)),
            Box::new(shift_vars(b, amount, cutoff + 1)),
        ),
        KernelExpr::Let(n, t, v, b) => KernelExpr::Let(
            *n,
            Box::new(shift_vars(t, amount, cutoff)),
            Box::new(shift_vars(v, amount, cutoff)),
            Box::new(shift_vars(b, amount, cutoff + 1)),
        ),
    }
}

/// Place a catch-all arm body (which binds the whole scrutinee value
/// as `Var(0)`) under a constructor's `net_extra` additional binders,
/// replacing the bound value with `repl` (the reconstructed
/// constructor application, built at the final depth).
///
/// `net_extra` is the final binder count minus one: positive when
/// wrapping under extra lambdas (e.g. `Nat.succ`: 2 binders, so +1),
/// negative when binders disappear (e.g. `Nat.zero`: 0 binders, so
/// −1, which degenerates to ordinary substitution).
fn graft_catchall_body(body: &KernelExpr, net_extra: i32, repl: &KernelExpr) -> KernelExpr {
    fn go(expr: &KernelExpr, net_extra: i32, repl: &KernelExpr, cutoff: u32) -> KernelExpr {
        match expr {
            KernelExpr::Sort(_) | KernelExpr::Const(..) | KernelExpr::Lit(_) => expr.clone(),
            KernelExpr::Var(i) => match i.0.cmp(&cutoff) {
                core::cmp::Ordering::Less => expr.clone(),
                core::cmp::Ordering::Equal => shift_vars(repl, cutoff, 0),
                core::cmp::Ordering::Greater => {
                    // Same outer level at the final depth: source
                    // index `i.0 - cutoff` becomes
                    // `i.0 - cutoff + net_extra`, re-based under
                    // `cutoff` entered binders. Non-negative: here
                    // `i.0 >= cutoff + 1` and `net_extra >= -1`.
                    KernelExpr::Var(DeBruijnIndex((i.0 as i32 + net_extra) as u32))
                }
            },
            KernelExpr::App(f, a) => KernelExpr::App(
                Box::new(go(f, net_extra, repl, cutoff)),
                Box::new(go(a, net_extra, repl, cutoff)),
            ),
            KernelExpr::Lam(bi, n, d, b) => KernelExpr::Lam(
                *bi,
                *n,
                Box::new(go(d, net_extra, repl, cutoff)),
                Box::new(go(b, net_extra, repl, cutoff + 1)),
            ),
            KernelExpr::Pi(bi, n, d, b) => KernelExpr::Pi(
                *bi,
                *n,
                Box::new(go(d, net_extra, repl, cutoff)),
                Box::new(go(b, net_extra, repl, cutoff + 1)),
            ),
            KernelExpr::Let(n, t, v, b) => KernelExpr::Let(
                *n,
                Box::new(go(t, net_extra, repl, cutoff)),
                Box::new(go(v, net_extra, repl, cutoff)),
                Box::new(go(b, net_extra, repl, cutoff + 1)),
            ),
        }
    }
    go(body, net_extra, repl, 0)
}

impl Elaborator {
    // -- imports ------------------------------------------------------

    fn elaborate_import(
        &mut self,
        import: &theoria_syntax::ImportDecl,
    ) -> Result<(), ElaborateError> {
        let mut dotted = String::new();
        for (i, seg) in import.path.segments.iter().enumerate() {
            if i > 0 {
                dotted.push('.');
            }
            dotted.push_str(&seg.text);
        }
        // The name list, if present, is discarded: delivery 2 has a
        // single implicit namespace.
        if dotted != "Standard.Prelude" {
            return Err(ElaborateError::new(
                import.span,
                ElaborateErrorKind::UnknownModule,
                format!("unknown module `{dotted}` (only Standard.Prelude is supported)"),
            ));
        }
        Ok(())
    }

    // -- functions ----------------------------------------------------

    fn elaborate_function(
        &mut self,
        f: &theoria_syntax::FunctionDef,
    ) -> Result<ElaboratedFunction, ElaborateError> {
        let err =
            |kind: ElaborateErrorKind, message: String| ElaborateError::new(f.span, kind, message);

        let kernel_name = self.names.intern(&f.name);
        if self.env.find(kernel_name).is_some() {
            // Delivery 2 has no namespace or shadowing story; a
            // redefinition is rejected rather than silently shadowed.
            return Err(err(
                ElaborateErrorKind::NotImplemented,
                format!("`{}` is already defined", f.name),
            ));
        }

        let Some(return_ty) = &f.return_ty else {
            return Err(err(
                ElaborateErrorKind::NotImplemented,
                "functions require an explicit return type".to_string(),
            ));
        };

        let mut scope = Scope::new();
        let mut domains: Vec<(NameId, KernelExpr)> = Vec::new();
        for param in &f.params {
            self.reject_global_shadow(&param.name, param.name_span)?;
            let dom = self.elaborate_expr(&param.ty, &mut scope)?;
            let unit = self.classify_type(&dom).exponents();
            let nid = self.names.intern(&param.name);
            let binding = match unit {
                Some(u) => LocalBinding::with_unit(param.name.clone(), Rc::new(dom.clone()), u),
                None => LocalBinding::bare(param.name.clone(), Rc::new(dom.clone())),
            };
            scope.push(binding);
            domains.push((nid, dom));
        }
        let codomain = self.elaborate_expr(return_ty, &mut scope)?;
        let body_e = self.elaborate_body(&f.body, &mut scope, &codomain, f.span)?;

        let mut ty = codomain;
        for (nid, dom) in domains.iter().rev() {
            ty = KernelExpr::Pi(
                BinderInfo::Default,
                *nid,
                Box::new(dom.clone()),
                Box::new(ty),
            );
        }
        let mut body = body_e;
        for (nid, dom) in domains.into_iter().rev() {
            body = KernelExpr::Lam(BinderInfo::Default, nid, Box::new(dom), Box::new(body));
        }

        let info = ConstantInfo::Definition(DefinitionVal {
            base: ConstantVal {
                name: kernel_name,
                universe_params: Vec::new(),
                ty: Rc::new(ty),
            },
            body: Rc::new(body),
            transparency: Transparency::Semireducible,
            termination: TerminationObligation::none(),
        });

        {
            let checker = TypeChecker::new(&self.env);
            checker.check_declaration(&info).map_err(|error| {
                let message = format!("kernel rejected `{}`: {error}", f.name);
                ElaborateError::new(f.span, ElaborateErrorKind::Kernel(error), message)
            })?;
        }
        self.env.add(info).map_err(|e| {
            err(
                ElaborateErrorKind::NotImplemented,
                format!("cannot register `{}`: {e}", f.name),
            )
        })?;

        Ok(ElaboratedFunction {
            source_name: f.name.clone(),
            kernel_name,
            arity: f.params.len(),
            span: f.span,
        })
    }

    /// Elaborate a `Theorem` declaration into a kernel `TheoremVal`.
    ///
    /// Sections become binders in order:
    ///
    /// 1. Every `Given:` parameter contributes a `Pi` (in the type) and
    ///    a `Lam` (in the body).
    /// 2. Every `Assume:` hypothesis contributes a `Pi` and a `Lam` the
    ///    same way.
    /// 3. `Show:` is the innermost codomain.
    /// 4. `Proof:` is the innermost body.
    ///
    /// The theorem is registered as `ConstantInfo::Theorem` with
    /// `Transparency::Irreducible`, following the kernel's convention
    /// that proof bodies are never unfolded during ordinary type
    /// checking.
    ///
    /// ## Known limitation
    ///
    /// The `Show:` goal is not checked to be a proposition (`Sort 0`).
    /// A `Theorem` whose goal is a `Nat` and whose proof is a `Nat`
    /// will register as a definition-shaped theorem; the kernel accepts
    /// it because `check_declaration` treats theorems identically to
    /// definitions. Enforcing `Prop`-ness requires the elaborator to
    /// run the kernel's `infer` on `Show` with the `Given`/`Assume`
    /// binders in scope, which is a small follow-up.
    fn elaborate_theorem(
        &mut self,
        t: &theoria_syntax::TheoremDef,
    ) -> Result<ElaboratedTheorem, ElaborateError> {
        let err =
            |kind: ElaborateErrorKind, message: String| ElaborateError::new(t.span, kind, message);

        let kernel_name = self.names.intern(&t.name);
        if self.env.find(kernel_name).is_some() {
            return Err(err(
                ElaborateErrorKind::NotImplemented,
                format!("`{}` is already defined", t.name),
            ));
        }

        // Build the theorem's scope: Given binders first, then Assume
        // hypotheses. Both are ordinary local bindings from the kernel's
        // perspective; only their surface sections differ.
        let mut scope = Scope::new();
        let mut binders: Vec<(NameId, KernelExpr, BinderInfo)> = Vec::new();

        for param in &t.given {
            self.reject_global_shadow(&param.name, param.name_span)?;
            let dom = self.elaborate_expr(&param.ty, &mut scope)?;
            let unit = self.classify_type(&dom).exponents();
            let nid = self.names.intern(&param.name);
            let binding = match unit {
                Some(u) => LocalBinding::with_unit(param.name.clone(), Rc::new(dom.clone()), u),
                None => LocalBinding::bare(param.name.clone(), Rc::new(dom.clone())),
            };
            scope.push(binding);
            binders.push((nid, dom, BinderInfo::Default));
        }

        for hyp in &t.assume {
            self.reject_global_shadow(&hyp.name, hyp.name_span)?;
            let ty = self.elaborate_expr(&hyp.ty, &mut scope)?;
            let nid = self.names.intern(&hyp.name);
            // A hypothesis is a proof obligation: its type must be a
            // proposition. The check runs with every prior `Given` and
            // `Assume` binder in scope, so a hypothesis may depend on
            // an earlier one.
            self.check_local_binding_is_prop(&binders, nid, hyp.name_span, &ty, "hypothesis")?;
            scope.push(LocalBinding::bare(hyp.name.clone(), Rc::new(ty.clone())));
            binders.push((nid, ty, BinderInfo::Default));
        }

        // Elaborate the goal, then verify it is a proposition before
        // elaborating the proof. Running the check first means the
        // user sees "goal is not a proposition" rather than a
        // confusing downstream kernel error.
        let show_k = self.elaborate_expr(&t.show, &mut scope)?;
        self.check_show_is_prop(&binders, &show_k, t.span)?;

        let proof_k = match &t.proof {
            theoria_syntax::ProofBlock::Term(e) => {
                self.elaborate_expr_checked(e, &mut scope, &show_k)?
            }
            theoria_syntax::ProofBlock::Steps(steps) => {
                self.elaborate_proof_steps(steps, &mut scope, &show_k, &binders)?
            }
        };

        // Wrap in Pi (type) and Lam (body), innermost = last binder.
        let mut ty = show_k;
        for (nid, dom, info) in binders.iter().rev() {
            ty = KernelExpr::Pi(*info, *nid, Box::new(dom.clone()), Box::new(ty));
        }
        let mut body = proof_k;
        for (nid, dom, info) in binders.iter().rev() {
            body = KernelExpr::Lam(*info, *nid, Box::new(dom.clone()), Box::new(body));
        }

        let info = ConstantInfo::Theorem(TheoremVal {
            base: ConstantVal {
                name: kernel_name,
                universe_params: Vec::new(),
                ty: Rc::new(ty),
            },
            body: Rc::new(body),
            transparency: Transparency::Irreducible,
        });

        // The kernel's `check_declaration` verifies both that the type
        // is a `Sort` and that the body has that type. Because the
        // theorem is closed (all `Given`/`Assume` binders are accounted
        // for by the outer `Pi`/`Lam`), the check runs in an empty
        // local context.
        {
            let checker = TypeChecker::new(&self.env);
            checker.check_declaration(&info).map_err(|error| {
                let message = format!("kernel rejected theorem `{}`: {error}", t.name);
                ElaborateError::new(t.span, ElaborateErrorKind::Kernel(error), message)
            })?;
        }
        self.env.add(info).map_err(|e| {
            err(
                ElaborateErrorKind::NotImplemented,
                format!("cannot register theorem `{}`: {e}", t.name),
            )
        })?;

        Ok(ElaboratedTheorem {
            source_name: t.name.clone(),
            kernel_name,
            num_given: t.given.len(),
            num_assume: t.assume.len(),
            span: t.span,
        })
    }

    /// Elaborate a numbered proof-step block into a kernel `Let`-chain
    /// ending in the `Exact` term.
    ///
    /// Steps are processed in order. Each `Let` and `Have` extends the
    /// local scope with a new binding; `Exact` provides the final proof
    /// term and must be the last step. The result is a nested
    /// `Expr::Let` sequence:
    ///
    /// ```text
    /// let n₁ : T₁ = v₁ in
    /// let n₂ : T₂ = v₂ in
    /// ... in
    /// exact_term
    /// ```
    ///
    /// # Errors
    ///
    /// * [`ElaborateErrorKind::InvalidProofStep`] if `Exact` is missing,
    ///   if any step follows `Exact`, or if a step's binding name is
    ///   already in scope.
    /// * [`ElaborateErrorKind::Kernel`] if elaboration of any step's
    ///   type or term fails.
    /// * Propagates errors from `elaborate_expr` and
    ///   `elaborate_expr_checked`.
    ///
    /// Elaborate a numbered proof-step block into a kernel `Let`-chain
    ///   ending in the final proof term.
    ///
    /// Steps are processed in order. Each `Let` and `Have` extends the
    /// local scope with a new binding; the final step must supply the
    /// theorem's proof (`Exact` or `From`). The result is a nested
    /// `Expr::Let` sequence.
    ///
    /// `initial_binders` carries the theorem's `Given` and `Assume`
    /// binders so that inference and Prop-ness checks can be run inside
    /// `elaborate_proof_steps`. Each `Let` and `Have` step appends to a
    /// local copy; the caller's vector is not modified.
    ///
    /// # Errors
    ///
    /// * [`ElaborateErrorKind::InvalidProofStep`] for a missing final
    ///   proof, a step after the final proof, or a name collision with
    ///   an enclosing binder or an earlier step.
    /// * [`ElaborateErrorKind::Kernel`] for elaboration failures.
    /// * Propagated errors from `elaborate_expr` and
    ///   `elaborate_expr_checked`.
    fn elaborate_proof_steps(
        &mut self,
        steps: &[theoria_syntax::ProofStep],
        scope: &mut Scope,
        goal: &KernelExpr,
        initial_binders: &[(NameId, KernelExpr, BinderInfo)],
    ) -> Result<KernelExpr, ElaborateError> {
        let mut binders: Vec<(NameId, KernelExpr, BinderInfo)> = initial_binders.to_vec();
        let mut bindings: Vec<(NameId, KernelExpr, KernelExpr)> = Vec::new();
        let mut proof: Option<KernelExpr> = None;

        for step in steps {
            if proof.is_some() {
                return Err(ElaborateError::new(
                    step.span,
                    ElaborateErrorKind::InvalidProofStep { step: step.number },
                    format!(
                        "step {} follows the final proof step; nothing may appear after it",
                        step.number
                    ),
                ));
            }
            match &step.kind {
                theoria_syntax::ProofStepKind::Let {
                    name,
                    name_span,
                    ty,
                    value,
                } => {
                    self.reject_global_shadow(name, *name_span)?;
                    if scope.lookup(name).is_some() {
                        return Err(ElaborateError::new(
                            *name_span,
                            ElaborateErrorKind::InvalidProofStep { step: step.number },
                            format!("`{name}` is already bound in this proof"),
                        ));
                    }
                    let (ty_k, value_k) = match ty {
                        Some(ty_expr) => {
                            let t = self.elaborate_expr(ty_expr, scope)?;
                            let v = self.elaborate_expr_checked(value, scope, &t)?;
                            (t, v)
                        }
                        None => {
                            // Infer the value's type in the current
                            // local context.
                            let v = self.elaborate_expr(value, scope)?;
                            let t = self.infer_type_in_context(&binders, &v, step.span)?;
                            (t, v)
                        }
                    };
                    let unit = self.classify_type(&ty_k).exponents();
                    let nid = self.names.intern(name);
                    let binding = match unit {
                        Some(u) => LocalBinding::with_unit(name.clone(), Rc::new(ty_k.clone()), u),
                        None => LocalBinding::bare(name.clone(), Rc::new(ty_k.clone())),
                    };
                    scope.push(binding);
                    binders.push((nid, ty_k.clone(), BinderInfo::Default));
                    bindings.push((nid, ty_k, value_k));
                }
                theoria_syntax::ProofStepKind::Have {
                    name,
                    name_span,
                    ty,
                    proof: proof_expr,
                } => {
                    self.reject_global_shadow(name, *name_span)?;
                    if scope.lookup(name).is_some() {
                        return Err(ElaborateError::new(
                            *name_span,
                            ElaborateErrorKind::InvalidProofStep { step: step.number },
                            format!("`{name}` is already bound in this proof"),
                        ));
                    }
                    let ty_k = self.elaborate_expr(ty, scope)?;
                    // A `Have` lemma's type is a proof obligation: it
                    // must be a proposition.
                    let nid = self.names.intern(name);
                    self.check_local_binding_is_prop(&binders, nid, *name_span, &ty_k, "lemma")?;
                    let proof_k = self.elaborate_expr_checked(proof_expr, scope, &ty_k)?;
                    let unit = self.classify_type(&ty_k).exponents();
                    let binding = match unit {
                        Some(u) => LocalBinding::with_unit(name.clone(), Rc::new(ty_k.clone()), u),
                        None => LocalBinding::bare(name.clone(), Rc::new(ty_k.clone())),
                    };
                    scope.push(binding);
                    binders.push((nid, ty_k.clone(), BinderInfo::Default));
                    bindings.push((nid, ty_k, proof_k));
                }
                theoria_syntax::ProofStepKind::Exact { term } => {
                    proof = Some(self.elaborate_expr_checked(term, scope, goal)?);
                }
                theoria_syntax::ProofStepKind::From { name, .. } => {
                    // `From h` is `Exact h` in term form. Synthesize
                    // the equivalent identifier expression and reuse
                    // the `Exact` path.
                    let ident = theoria_syntax::Expr::Ident {
                        text: name.clone(),
                        span: step.span,
                    };
                    proof = Some(self.elaborate_expr_checked(&ident, scope, goal)?);
                }
            }
        }

        let proof = proof.ok_or_else(|| {
            let last = steps.last().expect("parser guarantees at least one step");
            ElaborateError::new(
                last.span,
                ElaborateErrorKind::InvalidProofStep { step: last.number },
                "the `Proof:` block does not end in a proof step".to_string(),
            )
        })?;

        for _ in &bindings {
            let _ = scope.pop();
        }

        // Wrap `proof` in a nested `Let` chain, innermost = last
        // binding.
        let mut acc = proof;
        for (nid, ty_k, value_k) in bindings.into_iter().rev() {
            acc = KernelExpr::Let(nid, Box::new(ty_k), Box::new(value_k), Box::new(acc));
        }
        Ok(acc)
    }

    // -- structures ---------------------------------------------------

    /// Elaborate a `Structure` declaration into a kernel inductive triple
    /// plus one projection definition per field.
    ///
    /// Pipeline: validate names → elaborate parameters (each must be a
    /// sort) → elaborate field types (the structure's own name resolves
    /// to the type former for direct recursion) → classify recursion →
    /// build the triple → insert via `add_inductive` (positivity gate) →
    /// kernel-check every generated type → generate, check, and insert
    /// projections.
    ///
    /// Post-insertion kernel checks cannot roll the environment back on
    /// failure (the kernel offers no snapshots); a failure aborts the
    /// whole module, so the leftover entries are never observed.
    fn elaborate_structure(
        &mut self,
        s: &theoria_syntax::StructureDef,
    ) -> Result<ElaboratedStructure, ElaborateError> {
        use crate::structure as st;

        if s.fields.is_empty() {
            return Err(ElaborateError::new(
                s.name_span,
                ElaborateErrorKind::EmptyStructure,
                format!("structure `{}` must declare at least one field", s.name),
            ));
        }

        // Intern the generated names.
        let t_name = self.names.intern(&s.name);
        if self.env.find(t_name).is_some() {
            return Err(ElaborateError::new(
                s.name_span,
                ElaborateErrorKind::DuplicateStructure,
                format!("`{}` is already defined", s.name),
            ));
        }
        let ctor_name = self.names.intern(&format!("{}.mk", s.name));
        let rec_name = self.names.intern(&format!("{}.rec", s.name));
        let motive_name = self.names.intern("motive");
        let minor_name = self.names.intern("minor");
        let major_name = self.names.intern("major");
        let ih_name = self.names.intern("ih");

        // Duplicate fields.
        {
            let mut seen = BTreeMap::new();
            for f in &s.fields {
                if seen.insert(f.name.clone(), f.name_span).is_some() {
                    return Err(ElaborateError::new(
                        f.name_span,
                        ElaborateErrorKind::DuplicateField,
                        format!("duplicate field `{}` in structure `{}`", f.name, s.name),
                    ));
                }
            }
        }

        // 1. Parameters, left to right. Each parameter's type must
        //    evaluate to a sort (delivery 9 is universe-monomorphic;
        //    `n : Nat` needs index support).
        let mut scope = Scope::new();
        let mut param_names = Vec::with_capacity(s.params.len());
        let mut param_types = Vec::with_capacity(s.params.len());
        {
            let mut nenv = NbeEnv::empty();
            for (depth, p) in s.params.iter().enumerate() {
                self.reject_global_shadow(&p.name, p.name_span)?;
                let dom = self.elaborate_expr(&p.ty, &mut scope)?;
                let sort = {
                    let nbe = Nbe::new(&self.env, TransparencyMode::Semireducible);
                    nbe.eval(&nenv, &dom)
                };
                if !matches!(&*sort, Value::Sort(_)) {
                    return Err(ElaborateError::new(
                        p.name_span,
                        ElaborateErrorKind::NonSortParameter,
                        format!(
                            "parameter `{}` of structure `{}` must have a sort type; \
                             value parameters need index support, not yet implemented",
                            p.name, s.name,
                        ),
                    ));
                }
                let nid = self.names.intern(&p.name);
                nenv = nenv.extend(Value::fresh_var(DbLevel(depth as u32)));
                scope.push(LocalBinding::bare(p.name.clone(), Rc::new(dom.clone())));
                param_names.push(nid);
                param_types.push(dom);
            }
        }
        let k = param_types.len();

        // 2. Field types, in the parameter scope. The structure's own
        //    name resolves to the type former (direct recursion). A
        //    field mentioning an earlier field fails name resolution
        //    (fields are never in scope) with `UnknownIdentifier`.
        self.self_type = Some(t_name);
        let mut field_names = Vec::with_capacity(s.fields.len());
        let mut field_types = Vec::with_capacity(s.fields.len());
        for f in &s.fields {
            let ty = self.elaborate_expr(&f.ty, &mut scope)?;
            field_names.push(self.names.intern(&f.name));
            field_types.push(ty);
        }
        self.self_type = None;
        // The parameter scope is discarded; indices from here on are
        // absolute (top-level) de Bruijn indices.
        debug_assert_eq!(scope.len(), k);

        // 3. Classify recursion per field.
        let mut is_recursive = Vec::with_capacity(s.fields.len());
        for (f, ty) in s.fields.iter().zip(field_types.iter()) {
            if st::is_direct_recursion(ty, t_name, k) {
                is_recursive.push(true);
            } else if !st::mentions(ty, t_name) {
                is_recursive.push(false);
            } else if st::occurs_in_pi_domain(ty, t_name) {
                return Err(ElaborateError::new(
                    f.name_span,
                    ElaborateErrorKind::NonPositiveField,
                    format!(
                        "field `{}` of structure `{}` mentions `{}` in a negative position",
                        f.name, s.name, s.name,
                    ),
                ));
            } else {
                return Err(ElaborateError::new(
                    f.name_span,
                    ElaborateErrorKind::NestedRecursion,
                    format!(
                        "field `{}` of structure `{}` has non-direct recursion; \
                         only `{}` applied to its parameters is supported in this delivery",
                        f.name, s.name, s.name,
                    ),
                ));
            }
        }

        // 4-7. Build and insert the triple (positivity gate fires here).
        let shape = st::StructureShape {
            name: t_name,
            ctor_name,
            rec_name,
            param_names,
            param_types,
            field_names,
            field_types,
            is_recursive,
            motive_name,
            minor_name,
            major_name,
            ih_name,
            rec_universe: UniverseParamId::fresh(),
        };
        let (ind, ctors, rec) = st::build_triple(&shape);
        self.env
            .add_inductive(ind, ctors, rec)
            .map_err(|e| match e {
                EnvError::DuplicateDeclaration(_) => ElaborateError::new(
                    s.name_span,
                    ElaborateErrorKind::DuplicateStructure,
                    format!("`{}` is already defined: {e}", s.name),
                ),
                EnvError::NonPositiveInductive { .. } => ElaborateError::new(
                    s.name_span,
                    ElaborateErrorKind::NonPositiveField,
                    format!(
                        "structure `{}` failed the kernel positivity check: {e}",
                        s.name
                    ),
                ),
                other => ElaborateError::new(
                    s.name_span,
                    ElaborateErrorKind::Kernel(theoria_kernel::KernelError::Internal(format!(
                        "{other:?}"
                    ))),
                    format!("kernel rejected structure `{}`: {other}", s.name),
                ),
            })?;

        // 8. Kernel-check every generated type now that the names
        //    resolve. This verifies the de Bruijn arithmetic above.
        {
            let checker = TypeChecker::new(&self.env);
            for info in [
                ConstantInfo::Inductive(
                    self.env
                        .find(t_name)
                        .and_then(|i| match &*i {
                            ConstantInfo::Inductive(v) => Some(v.clone()),
                            _ => None,
                        })
                        .expect("inductive just inserted"),
                ),
                ConstantInfo::Constructor(
                    self.env
                        .find(ctor_name)
                        .and_then(|i| match &*i {
                            ConstantInfo::Constructor(v) => Some(v.clone()),
                            _ => None,
                        })
                        .expect("constructor just inserted"),
                ),
                ConstantInfo::Recursor(
                    self.env
                        .find(rec_name)
                        .and_then(|i| match &*i {
                            ConstantInfo::Recursor(v) => Some(v.clone()),
                            _ => None,
                        })
                        .expect("recursor just inserted"),
                ),
            ] {
                checker.check_declaration(&info).map_err(|error| {
                    let message = format!("kernel rejected structure `{}`: {error}", s.name);
                    ElaborateError::new(s.name_span, ElaborateErrorKind::Kernel(error), message)
                })?;
            }
        }

        // 9. Projections: `T.fᵢ : Π {p⃗} (s : T p⃗). Fᵢ`.
        let mut record_fields = Vec::with_capacity(shape.field_types.len());
        for (i, f) in s.fields.iter().enumerate() {
            let proj_name = self.names.intern(&format!("{}.{}", s.name, f.name));
            let proj_ty = st::projection_type(&shape, i);
            let proj_body = st::projection_body(&shape, i);
            let info = ConstantInfo::Definition(DefinitionVal {
                base: ConstantVal {
                    name: proj_name,
                    universe_params: Vec::new(),
                    ty: Rc::new(proj_ty),
                },
                body: Rc::new(proj_body),
                transparency: Transparency::Semireducible,
                termination: TerminationObligation::none(),
            });
            {
                let checker = TypeChecker::new(&self.env);
                checker.check_declaration(&info).map_err(|error| {
                    let message = format!(
                        "kernel rejected projection `{}.{}`: {error}",
                        s.name, f.name
                    );
                    ElaborateError::new(f.name_span, ElaborateErrorKind::Kernel(error), message)
                })?;
            }
            self.env.add(info).map_err(|e| {
                ElaborateError::new(
                    f.name_span,
                    ElaborateErrorKind::DuplicateStructure,
                    format!("cannot register `{}.{}`: {e}", s.name, f.name),
                )
            })?;
            record_fields.push(StructureField {
                name: f.name.clone(),
                proj: proj_name,
                ty: shape.field_types[i].clone(),
            });
        }

        // 10. Record the structure for field access.
        self.struct_records.insert(
            t_name,
            StructureRecord {
                source_name: s.name.clone(),
                num_params: k,
                fields: record_fields,
            },
        );

        Ok(ElaboratedStructure {
            source_name: s.name.clone(),
            kernel_name: t_name,
            num_params: k,
            num_fields: s.fields.len(),
            span: s.span,
        })
    }

    /// Reject a structure whose own name, constructor name, recursor
    /// name, or any projection name collides with an existing
    /// declaration in the environment.
    ///
    /// Called in the module pre-flight so that a colliding structure is
    /// rejected before any declaration is inserted. `elaborate_structure`
    /// performs the same check again as a defence in depth; the
    /// duplicate work is negligible.
    fn check_structure_names_free(
        &mut self,
        s: &theoria_syntax::StructureDef,
    ) -> Result<(), ElaborateError> {
        let mut checks: Vec<(String, Span, &'static str)> = vec![
            (s.name.clone(), s.name_span, "structure"),
            (format!("{}.mk", s.name), s.name_span, "constructor"),
            (format!("{}.rec", s.name), s.name_span, "recursor"),
        ];
        for f in &s.fields {
            checks.push((format!("{}.{}", s.name, f.name), f.name_span, "projection"));
        }
        for (text, span, what) in checks {
            let id = self.names.intern(&text);
            if self.env.find(id).is_some() {
                return Err(ElaborateError::new(
                    span,
                    ElaborateErrorKind::DuplicateStructure,
                    format!("`{text}` ({what}) is already defined"),
                ));
            }
        }
        Ok(())
    }

    // -- expressions --------------------------------------------------

    /// Elaborate a function body: zero or more `let`s followed by exactly
    /// one `return`.
    ///
    /// Each `let` extends `scope` (reusing delivery 5's shadow check) and
    /// is rebuilt as a kernel `Let`, innermost last. The `return`
    /// expression is elaborated in checked mode against the declared
    /// return type, which is the single hook that makes `if` work.
    /// `let`-bindings are popped before returning; the caller discards
    /// the scope afterwards, so pop order does not matter.
    fn elaborate_body(
        &mut self,
        stmts: &[theoria_syntax::Stmt],
        scope: &mut Scope,
        return_ty: &KernelExpr,
        func_span: Span,
    ) -> Result<KernelExpr, ElaborateError> {
        let Some((last, rest)) = stmts.split_last() else {
            return Err(ElaborateError::new(
                func_span,
                ElaborateErrorKind::BodyNotReturning,
                "function body must end in a `return` statement".to_string(),
            ));
        };
        let theoria_syntax::Stmt::Return { value, .. } = last else {
            return Err(ElaborateError::new(
                func_span,
                ElaborateErrorKind::BodyNotReturning,
                "function body must end in a `return` statement".to_string(),
            ));
        };

        let mut bindings: Vec<(String, KernelExpr, KernelExpr)> = Vec::new();
        for stmt in rest {
            match stmt {
                theoria_syntax::Stmt::Let {
                    name,
                    name_span,
                    ty,
                    value,
                    ..
                } => {
                    self.reject_global_shadow(name, *name_span)?;
                    let ty_k = self.elaborate_expr(ty, scope)?;
                    let v_k = self.elaborate_expr(value, scope)?;
                    let unit = self.classify_type(&ty_k).exponents();
                    let binding = match unit {
                        Some(u) => LocalBinding::with_unit(name.clone(), Rc::new(ty_k.clone()), u),
                        None => LocalBinding::bare(name.clone(), Rc::new(ty_k.clone())),
                    };
                    scope.push(binding);
                    bindings.push((name.clone(), ty_k, v_k));
                }
                theoria_syntax::Stmt::Return { .. } => {
                    return Err(ElaborateError::new(
                        stmt.span(),
                        ElaborateErrorKind::BodyNotReturning,
                        "`return` must be the last statement in a function body".to_string(),
                    ));
                }
            }
        }

        let body_k = self.elaborate_expr_checked(value, scope, return_ty)?;

        // Source-level unit check: the body's unit must unify with the
        // declared return type's unit. The kernel's type check would
        // accept a bare `Nat.add` here regardless, because `Quantity(u)`
        // is defeq to `Nat`; this is the elaborator enforcing the
        // dimensional discipline.
        let body_unit = self.unit_of_surface(scope, value)?;
        let declared_unit = self.classify_type(return_ty);
        self.unify_classes(body_unit, declared_unit, func_span)?;

        for _ in &bindings {
            let _ = scope.pop();
        }

        let mut acc = body_k;
        for (name, ty_k, v_k) in bindings.into_iter().rev() {
            let nid = self.names.intern(&name);
            acc = KernelExpr::Let(nid, Box::new(ty_k), Box::new(v_k), Box::new(acc));
        }
        Ok(acc)
    }

    /// Reject a parameter name that shadows a global constant.
    ///
    /// Called immediately before a parameter is pushed onto the scope, in
    /// both `Function` and lambda elaboration. The check is against the
    /// `NameTable`-interned form of the parameter's source name; the
    /// interned id is looked up in the global environment.
    fn reject_global_shadow(
        &mut self,
        source_name: &str,
        span: Span,
    ) -> Result<(), ElaborateError> {
        let id = self.names.intern(source_name);
        if self.env.find(id).is_some() {
            let message = format!(
                "parameter `{source_name}` shadows the global constant \
                 `{source_name}`; rename the parameter to resolve the \
                 ambiguity"
            );
            return Err(ElaborateError::new(
                span,
                ElaborateErrorKind::ShadowedGlobal {
                    name: source_name.into(),
                    shadowed: source_name.into(),
                },
                message,
            ));
        }
        Ok(())
    }

    fn elaborate_expr(
        &mut self,
        expr: &theoria_syntax::Expr,
        scope: &mut Scope,
    ) -> Result<KernelExpr, ElaborateError> {
        use theoria_syntax::Expr as E;
        match expr {
            E::Ident { text, span } => self.resolve_ident(text, *span, scope),
            E::Nat { value, span } => self.elaborate_nat(*value, *span),
            E::Float { span, .. } => Err(ElaborateError::new(
                *span,
                ElaborateErrorKind::NotImplemented,
                "float literals are not supported: the kernel has no float type".to_string(),
            )),
            E::Str { span, .. } => Err(ElaborateError::new(
                *span,
                ElaborateErrorKind::NotImplemented,
                "string literals are not supported: the kernel has no string type".to_string(),
            )),
            E::Field {
                obj,
                field,
                field_span,
                span,
            } => self.elaborate_field(obj, field, *field_span, *span, scope),
            E::Index { span, .. } => Err(ElaborateError::new(
                *span,
                ElaborateErrorKind::NotImplemented,
                "indexing is not supported in this delivery".to_string(),
            )),
            E::Call { func, args, span } => {
                // The `Quantity` type former is handled specially: its
                // single argument is a unit expression, not an ordinary
                // term. Without this intercept, resolution would find
                // the installed `Quantity` constant, try to apply it,
                // and fail on the universe-arity check (the fragment
                // never writes `Quantity.{u}` explicitly).
                if let E::Ident { text, .. } = func.as_ref() {
                    if text == "Quantity" {
                        return self.elaborate_quantity(args, *span);
                    }
                }
                let mut e = self.elaborate_expr(func, scope)?;
                for a in args {
                    let av = self.elaborate_expr(a, scope)?;
                    e = KernelExpr::app(e, av);
                }
                Ok(e)
            }
            E::BinOp { lhs, op, rhs, span } => self.elaborate_binop(lhs, *op, rhs, *span, scope),
            E::UnOp { span, .. } => Err(ElaborateError::new(
                *span,
                ElaborateErrorKind::NotImplemented,
                "unary minus is not supported: the prelude has no negation yet".to_string(),
            )),
            E::Paren { inner, .. } => self.elaborate_expr(inner, scope),
            E::Lam { params, body, .. } => self.elaborate_lam(params, body, scope),
            E::If(if_expr) => Err(ElaborateError::new(
                if_expr.span,
                ElaborateErrorKind::NotImplemented,
                "`if` requires an expected type; wrap it in a `return` or use `Bool.rec` directly"
                    .to_string(),
            )),
            E::Match(m) => Err(ElaborateError::new(
                m.span,
                ElaborateErrorKind::NotImplemented,
                "`match` requires an expected type; wrap it in a `return`, an `if` branch, or a `match` arm"
                    .to_string(),
            )),
        }
    }

    /// Elaborate `expr` in checked mode: an expected kernel type is
    /// supplied by the caller, and constructs that need a hint (notably
    /// `if`) use it.
    ///
    /// For every CST variant except `Expr::If`, this delegates to
    /// [`elaborate_expr`](Elaborator::elaborate_expr) and returns its
    /// result. The kernel's top-level `check_declaration` does the actual
    /// type checking later; checked mode here only exists to thread a
    /// hint down to the `if` rule.
    fn elaborate_expr_checked(
        &mut self,
        expr: &theoria_syntax::Expr,
        scope: &mut Scope,
        expected: &KernelExpr,
    ) -> Result<KernelExpr, ElaborateError> {
        if let theoria_syntax::Expr::If(if_expr) = expr {
            return self.elaborate_if(
                &if_expr.cond,
                &if_expr.then_branch,
                &if_expr.else_branch,
                if_expr.span,
                scope,
                expected,
            );
        }
        if let theoria_syntax::Expr::Match(m) = expr {
            return self.elaborate_match(m, scope, expected);
        }
        self.elaborate_expr(expr, scope)
    }

    /// Elaborate `if cond then a else b` against `expected` as a
    /// `Bool.rec` application.
    ///
    /// The condition is elaborated in infer mode (the kernel confirms it
    /// is `Bool`-typed when the surrounding term is checked); both
    /// branches are elaborated in checked mode against `expected`, so
    /// nested `if`s work. The motive is `λ _ : Bool. expected` at
    /// universe `Sort 1`, matching the fragment's `Nat`/`Bool` types.
    fn elaborate_if(
        &mut self,
        cond: &theoria_syntax::Expr,
        then_branch: &theoria_syntax::Expr,
        else_branch: &theoria_syntax::Expr,
        span: Span,
        scope: &mut Scope,
        expected: &KernelExpr,
    ) -> Result<KernelExpr, ElaborateError> {
        let cond_k = self.elaborate_expr(cond, scope)?;
        let then_k = self.elaborate_expr_checked(then_branch, scope, expected)?;
        let else_k = self.elaborate_expr_checked(else_branch, scope, expected)?;

        let bool_name = self.names.intern("Bool");
        let Some(bool_info) = self.env.find(bool_name) else {
            return Err(ElaborateError::not_implemented(
                span,
                "`if` requires Bool and Bool.rec in the environment",
            ));
        };
        let _ = bool_info;
        let rec_name = self.names.intern("Bool.rec");
        if self.env.find(rec_name).is_none() {
            return Err(ElaborateError::not_implemented(
                span,
                "`if` requires Bool and Bool.rec in the environment",
            ));
        }

        let bool_ty = KernelExpr::Const(bool_name, Vec::new());
        let motive = KernelExpr::Lam(
            BinderInfo::Default,
            NameId(u32::MAX),
            Box::new(bool_ty),
            Box::new(expected.clone()),
        );
        Ok(KernelExpr::Const(rec_name, alloc::vec![Level::Zero.succ()])
            .apps([motive, then_k, else_k, cond_k]))
    }

    /// Elaborate `Quantity(u)` where `u` is a unit expression.
    ///
    /// The argument is parsed into an exponent vector, the canonical
    /// kernel constant `U_<m>_<s>_<kg>_<a>_<k>_<mol>_<cd>` is interned on
    /// demand, and the call elaborates to `Quantity.{1} c` where
    /// `c` is that constant.
    fn elaborate_quantity(
        &mut self,
        args: &[theoria_syntax::Expr],
        span: Span,
    ) -> Result<KernelExpr, ElaborateError> {
        if args.len() != 1 {
            return Err(ElaborateError::new(
                span,
                ElaborateErrorKind::NotImplemented,
                format!(
                    "`Quantity` takes exactly one unit argument, got {}",
                    args.len()
                ),
            ));
        }
        let exp = parse_unit(&args[0]).map_err(|e| {
            ElaborateError::new(
                span,
                ElaborateErrorKind::UnitError(e.clone()),
                format!("unit expression: {e}"),
            )
        })?;
        let unit_id = self
            .units
            .ensure(exp, &mut self.env, &mut self.names)
            .map_err(|e| {
                ElaborateError::new(
                    span,
                    ElaborateErrorKind::UnitError(e.clone()),
                    format!("unit registration: {e}"),
                )
            })?;
        let quantity_id = self.units.quantity();
        let quantity_expr = KernelExpr::Const(quantity_id, Vec::new());
        let unit_expr = KernelExpr::Const(unit_id, Vec::new());
        Ok(KernelExpr::app(quantity_expr, unit_expr))
    }

    // -- unit classification ------------------------------------------

    /// Classify a kernel type as bare or quantity-carrying.
    ///
    /// Recognises the canonical shape produced by
    /// [`Elaborator::elaborate_quantity`]: an application of the
    /// registered `Quantity` constant to a canonical `U_...` constant.
    /// Anything else is `Bare`.
    fn classify_type(&self, ty: &KernelExpr) -> UnitClass {
        let KernelExpr::App(f, a) = ty else {
            return UnitClass::Bare;
        };
        let KernelExpr::Const(qid, _) = f.as_ref() else {
            return UnitClass::Bare;
        };
        if *qid != self.units.quantity() {
            return UnitClass::Bare;
        }
        let KernelExpr::Const(uid, _) = a.as_ref() else {
            return UnitClass::Bare;
        };
        match self.units.exponents_of(*uid) {
            Some(e) => UnitClass::Quantity(e),
            None => UnitClass::Bare,
        }
    }

    /// Classify the return type of a function type by peeling its
    /// `Π`-binders and classifying the final codomain.
    ///
    /// This ignores binder dependencies; for the fragment's quantities,
    /// the return type is `Quantity(u)` with a closed `u`, so no
    /// substitution is required. A dependent return type whose codomain
    /// references an earlier binder is classified as `Bare`.
    fn classify_return_type(&self, ty: &KernelExpr) -> UnitClass {
        let mut cur = ty;
        while let KernelExpr::Pi(_, _, _, cod) = cur {
            cur = cod;
        }
        self.classify_type(cur)
    }

    /// A fresh `Nbe` engine over the current environment.
    ///
    /// `Nbe` is cheap to construct (it is a pair of a `&dyn ConstEnv`
    /// and a transparency mode), so building one on demand is preferred
    /// to caching it and threading a borrow through every method.
    fn nbe_engine(&self) -> Nbe<'_> {
        Nbe::new(&self.env, TransparencyMode::Semireducible)
    }

    /// A fresh `TypeChecker` over the current environment.
    fn type_checker(&self) -> TypeChecker<'_> {
        TypeChecker::new(&self.env)
    }

    /// Build a kernel `TypedContext` from a stack of binders, in
    /// declaration order.
    ///
    /// Each binder is a triple `(name, syntactic type, binder info)`. The
    /// syntactic type is expressed in the context *before* its own binder
    /// was pushed, matching the elaborator's `Scope` convention. The
    /// semantic type stored in the resulting `TypedContext` is the
    /// binder's type evaluated in the running `Nbe` environment, exactly
    /// what `TypeChecker::infer` returns for a variable.
    ///
    /// Returns both the `TypedContext` and the final `NbeEnv`; callers
    /// that only need the context can discard the env.
    fn build_typed_context(
        &self,
        binders: &[(NameId, KernelExpr, BinderInfo)],
    ) -> (TypedContext, NbeEnv) {
        let nbe = self.nbe_engine();
        let mut ctx = TypedContext::empty();
        let mut nenv = NbeEnv::empty();
        for (name, ty, info) in binders.iter() {
            let ty_v = nbe.eval(&nenv, ty);
            // All entries in `binders` are `Given` or `Assume` binders:
            // neither is a `Let`, so the `push_binder` shape is correct.
            // Its internal fresh-var allocation matches the one we
            // allocate separately for `nenv`.
            ctx.push_binder(LocalDecl::binder(*name, *info, Rc::new(ty.clone())), ty_v);
            nenv = nenv.extend(Value::fresh_var(DbLevel(nenv.len() as u32)));
        }
        (ctx, nenv)
    }

    /// Infer the sort of `e` in the local context described by `binders`.
    ///
    /// Returns:
    ///
    /// * `Ok(())` if the inferred sort is `Sort 0` (a proposition).
    /// * `Err(Some(level))` if the inferred sort is a `Sort` at a level
    ///   other than zero.
    /// * `Err(None)` if inference fails or the result is not a `Sort`.
    fn sort_of_in_context(
        &self,
        binders: &[(NameId, KernelExpr, BinderInfo)],
        e: &KernelExpr,
    ) -> Result<(), Option<Level>> {
        let (ctx, _env) = self.build_typed_context(binders);
        let inferred = match self.type_checker().infer(&ctx, e) {
            Ok(v) => v,
            Err(_) => return Err(None),
        };
        match &*inferred {
            Value::Sort(l) if l.normalize().is_zero() => Ok(()),
            Value::Sort(l) => Err(Some(l.normalize())),
            _ => Err(None),
        }
    }

    /// Check that a theorem's `Show:` goal is a proposition.
    ///
    /// Runs `sort_of_in_context` and constructs the appropriate error.
    fn check_show_is_prop(
        &self,
        binders: &[(NameId, KernelExpr, BinderInfo)],
        show_k: &KernelExpr,
        span: Span,
    ) -> Result<(), ElaborateError> {
        match self.sort_of_in_context(binders, show_k) {
            Ok(()) => Ok(()),
            Err(Some(level)) => Err(ElaborateError::new(
                span,
                ElaborateErrorKind::TheoremGoalNotAProposition {
                    level: Some(level.clone()),
                },
                format!(
                    "the theorem goal must be a proposition (`Sort 0`), but `Show:` lives in `{level}`"
                ),
            )),
            Err(None) => Err(ElaborateError::new(
                span,
                ElaborateErrorKind::TheoremGoalNotAProposition { level: None },
                "the theorem goal must be a proposition, but `Show:` is not a type".to_string(),
            )),
        }
    }

    /// Check that an `Assume:` hypothesis is a proposition.
    ///
    /// Same rule as `check_show_is_prop`, but the error names the
    /// hypothesis.
    fn check_local_binding_is_prop(
        &self,
        binders: &[(NameId, KernelExpr, BinderInfo)],
        name: NameId,
        name_span: Span,
        ty: &KernelExpr,
        kind: &str,
    ) -> Result<(), ElaborateError> {
        match self.sort_of_in_context(binders, ty) {
            Ok(()) => Ok(()),
            Err(Some(level)) => Err(ElaborateError::new(
                name_span,
                ElaborateErrorKind::LocalBindingNotAProposition {
                    name,
                    level: Some(level.clone()),
                },
                format!(
                    "{kind} `{}` must be a proposition (`Sort 0`), but its type lives in `{level}`",
                    self.names.resolve(name),
                ),
            )),
            Err(None) => Err(ElaborateError::new(
                name_span,
                ElaborateErrorKind::LocalBindingNotAProposition { name, level: None },
                format!(
                    "{kind} `{}` must be a proposition, but its type is not a type",
                    self.names.resolve(name),
                ),
            )),
        }
    }

    /// Infer the type of an elaborated value in a local context, and
    /// read it back as a syntactic expression.
    ///
    /// Used by `Let`-without-annotation inference in proof steps. The
    /// local context is built from `binders` by treating every entry as
    /// a λ-binder; this is correct for `Given` and `Assume` entries but
    /// not for `Let` entries introduced by earlier steps, whose
    /// *values* would be inlined by delivery 19's `push_let`.
    ///
    /// The practical consequence: an inferred type works when the
    /// value's type does not depend on a prior `Let` binding's value
    /// (the common case — `Let n = 0`, `Let p = Eq(Nat, 0, 0)`, etc.).
    /// When the value's type does depend on a prior `Let`, the inferred
    /// type contains a fresh neutral in place of that value, and
    /// downstream defeq checks may fail. Full support is deferred to
    /// the follow-up that carries a `TypedContext` alongside the
    /// elaborator's `Scope`.
    fn infer_type_in_context(
        &self,
        binders: &[(NameId, KernelExpr, BinderInfo)],
        value: &KernelExpr,
        span: Span,
    ) -> Result<KernelExpr, ElaborateError> {
        let (ctx, _env) = self.build_typed_context(binders);
        let inferred = self.type_checker().infer(&ctx, value).map_err(|e| {
            ElaborateError::new(
                span,
                ElaborateErrorKind::Kernel(e.clone()),
                format!("cannot infer the type of this `Let` value: {e}"),
            )
        })?;
        Ok(self.nbe_engine().quote(binders.len(), &inferred))
    }

    /// Determine the unit class of a surface expression.
    ///
    /// Recursive over the surface tree. For the fragment, this covers:
    ///
    /// * `Ident` — looked up in the scope first, then the environment.
    /// * `Nat` / `Float` / `Str` — `Bare`.
    /// * `Paren` — recurses.
    /// * `UnOp` — recurses on the operand.
    /// * `BinOp` — combines operand classes by the operator's rule.
    /// * `Field` — qualified-name lookup against the environment.
    /// * `Call` — looks up the callee's signature, classifies the return
    ///   type.
    /// * `If` / `Match` — unifies the branch classes.
    ///
    /// Anything else (lambdas, index expressions) is `Bare`.
    fn unit_of_surface(
        &mut self,
        scope: &Scope,
        e: &theoria_syntax::Expr,
    ) -> Result<UnitClass, ElaborateError> {
        use theoria_syntax::Expr as E;
        match e {
            E::Ident { text, .. } => {
                if let Some(idx) = scope.lookup(text) {
                    let b = scope.lookup_index(idx).expect("lookup just succeeded");
                    return Ok(match b.unit {
                        Some(u) => UnitClass::Quantity(u),
                        None => UnitClass::Bare,
                    });
                }
                let id = self.names.intern(text);
                match self.env.find(id) {
                    Some(info) => Ok(self.classify_return_type(info.ty())),
                    None => Ok(UnitClass::Bare),
                }
            }
            E::Nat { .. } | E::Float { .. } | E::Str { .. } => Ok(UnitClass::Bare),
            E::Paren { inner, .. } => self.unit_of_surface(scope, inner),
            E::UnOp { operand, .. } => self.unit_of_surface(scope, operand),
            E::BinOp { lhs, op, rhs, span } => {
                // `^` is special: the RHS must be a literal exponent,
                // not a value with a unit. Handling it here (rather than
                // in `combine_binop_units`) lets the classifier raise the
                // base's unit to the exponent and produce the correct
                // result class, instead of falling back to `Bare`.
                if let theoria_syntax::BinOp::Pow = op {
                    let base = self.unit_of_surface(scope, lhs)?;
                    match surface_nat_literal(rhs) {
                        Some(n) => {
                            return Ok(match base {
                                UnitClass::Bare => UnitClass::Bare,
                                UnitClass::Quantity(u) => UnitClass::Quantity(u.pow(n)),
                            });
                        }
                        None => {
                            // For a bare base like `2 ^ (3 ^ 2)`, a
                            // non-literal exponent is fine: the result
                            // stays bare. Only a quantity base requires
                            // a literal to compute the resulting unit.
                            if matches!(base, UnitClass::Bare) {
                                return Ok(UnitClass::Bare);
                            }
                            return Err(ElaborateError::new(
                                *span,
                                ElaborateErrorKind::UnitError(UnitError::NonLiteralExponent),
                                "exponent of `^` must be a literal natural number".to_string(),
                            ));
                        }
                    }
                }
                let l = self.unit_of_surface(scope, lhs)?;
                let r = self.unit_of_surface(scope, rhs)?;
                self.combine_binop_units(l, r, *op, *span)
            }
            E::Field { .. } => {
                // Flatten to a qualified name and look it up.
                let mut segments: Vec<String> = Vec::new();
                let mut cur = e;
                loop {
                    match cur {
                        E::Field { obj, field, .. } => {
                            segments.push(field.clone());
                            cur = obj.as_ref();
                        }
                        E::Ident { text, .. } => {
                            segments.push(text.clone());
                            break;
                        }
                        _ => return Ok(UnitClass::Bare),
                    }
                }
                segments.reverse();
                let qualified = segments.join(".");
                let id = self.names.intern(&qualified);
                match self.env.find(id) {
                    Some(info) => Ok(self.classify_return_type(info.ty())),
                    None => Ok(UnitClass::Bare),
                }
            }
            E::Call { func, .. } => {
                let func_ty = self.type_of_callee(scope, func);
                match func_ty {
                    Some(ty) => Ok(self.classify_return_type(&ty)),
                    None => Ok(UnitClass::Bare),
                }
            }
            E::If(if_expr) => {
                let t = self.unit_of_surface(scope, &if_expr.then_branch)?;
                let f = self.unit_of_surface(scope, &if_expr.else_branch)?;
                self.unify_classes(t, f, if_expr.span)
            }
            E::Match(m) => {
                let mut acc: Option<UnitClass> = None;
                for arm in &m.arms {
                    let u = self.unit_of_surface(scope, &arm.body)?;
                    acc = Some(match acc {
                        None => u,
                        Some(prev) => self.unify_classes(prev, u, arm.span)?,
                    });
                }
                Ok(acc.unwrap_or(UnitClass::Bare))
            }
            _ => Ok(UnitClass::Bare),
        }
    }

    /// Look up the elaborated type of a callee expression, if resolvable.
    fn type_of_callee(&mut self, scope: &Scope, func: &theoria_syntax::Expr) -> Option<KernelExpr> {
        use theoria_syntax::Expr as E;
        match func {
            E::Ident { text, .. } => {
                if scope.lookup(text).is_some() {
                    // A local variable: its type is a variable type, not
                    // a callable signature we can peel.
                    return None;
                }
                let id = self.names.intern(text);
                self.env.find(id).map(|i| (**i.ty()).clone())
            }
            E::Field { .. } => {
                let mut segments: Vec<String> = Vec::new();
                let mut cur = func;
                loop {
                    match cur {
                        E::Field { obj, field, .. } => {
                            segments.push(field.clone());
                            cur = obj.as_ref();
                        }
                        E::Ident { text, .. } => {
                            segments.push(text.clone());
                            break;
                        }
                        _ => return None,
                    }
                }
                segments.reverse();
                let qualified = segments.join(".");
                let id = self.names.intern(&qualified);
                self.env.find(id).map(|i| (**i.ty()).clone())
            }
            _ => None,
        }
    }

    /// Unify two unit classes.
    ///
    /// * `Bare` unifies with anything; the result is the more specific
    ///   side.
    /// * Two `Quantity`s unify iff their exponents are equal.
    fn unify_classes(
        &self,
        a: UnitClass,
        b: UnitClass,
        span: Span,
    ) -> Result<UnitClass, ElaborateError> {
        match (a, b) {
            (UnitClass::Bare, UnitClass::Bare) => Ok(UnitClass::Bare),
            (UnitClass::Bare, UnitClass::Quantity(u))
            | (UnitClass::Quantity(u), UnitClass::Bare) => Ok(UnitClass::Quantity(u)),
            (UnitClass::Quantity(u1), UnitClass::Quantity(u2)) => {
                if u1 == u2 {
                    Ok(UnitClass::Quantity(u1))
                } else {
                    Err(ElaborateError::new(
                        span,
                        ElaborateErrorKind::UnitError(UnitError::UnitMismatch { lhs: u1, rhs: u2 }),
                        format!("incompatible units `{}` and `{}`", u1.render(), u2.render()),
                    ))
                }
            }
        }
    }

    /// Combine operand unit classes according to a binary operator.
    fn combine_binop_units(
        &self,
        l: UnitClass,
        r: UnitClass,
        op: theoria_syntax::BinOp,
        span: Span,
    ) -> Result<UnitClass, ElaborateError> {
        use theoria_syntax::BinOp as B;
        match op {
            // Additive: same unit, or one side bare (coerced).
            B::Add | B::Sub => self.unify_classes(l, r, span),
            // Multiplicative: units multiply; bare is dimensionless.
            B::Mul => Ok(match (l, r) {
                (UnitClass::Bare, UnitClass::Bare) => UnitClass::Bare,
                (UnitClass::Quantity(u), UnitClass::Bare)
                | (UnitClass::Bare, UnitClass::Quantity(u)) => UnitClass::Quantity(u),
                (UnitClass::Quantity(u1), UnitClass::Quantity(u2)) => {
                    UnitClass::Quantity(u1.mul(u2))
                }
            }),
            // Divisive: units divide; bare is dimensionless.
            B::Div => Ok(match (l, r) {
                (UnitClass::Bare, UnitClass::Bare) => UnitClass::Bare,
                (UnitClass::Quantity(u), UnitClass::Bare) => UnitClass::Quantity(u),
                (UnitClass::Bare, UnitClass::Quantity(u)) => {
                    UnitClass::Quantity(Exponents::DIMENSIONLESS.div(u))
                }
                (UnitClass::Quantity(u1), UnitClass::Quantity(u2)) => {
                    UnitClass::Quantity(u1.div(u2))
                }
            }),
            // `^` is intercepted by `unit_of_surface` before reaching
            // here, because the exponent must be inspected in the
            // surface syntax. Reaching this arm would mean a caller
            // dispatched `Pow` incorrectly; reject defensively.
            B::Pow => Err(ElaborateError::new(
                span,
                ElaborateErrorKind::NotImplemented,
                "internal: `Pow` reached `combine_binop_units`".to_string(),
            )),
            // Comparisons: operands must unify; the result is `Bool`.
            B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge => {
                let _ = self.unify_classes(l, r, span)?;
                Ok(UnitClass::Bare)
            }
            // `->` is a type constructor, not a value.
            B::Arrow => Ok(UnitClass::Bare),
        }
    }

    fn resolve_ident(
        &mut self,
        text: &str,
        span: Span,
        scope: &Scope,
    ) -> Result<KernelExpr, ElaborateError> {
        // 1. Local scope (innermost wins).
        if let Some(idx) = scope.lookup(text) {
            return Ok(KernelExpr::Var(idx));
        }
        // 1b. The inductive being defined. Field types may mention the
        // structure itself for direct recursion; it is not yet in the
        // environment, so it resolves to the bare type former here.
        // (Parameter types are elaborated before this is set, so they
        // cannot mention the structure.)
        let id = self.names.intern(text);
        if Some(id) == self.self_type {
            return Ok(KernelExpr::Const(id, Vec::new()));
        }
        // 1c. The `Type` universe (surface syntax for `Sort 1`).
        // Parameters range over types, so structures need to name the
        // universe they live in.
        if text == "Type" {
            return Ok(KernelExpr::Sort(Level::Zero.succ()));
        }
        // 2. Global environment.
        let info = self.env.find(id).ok_or_else(|| {
            ElaborateError::new(
                span,
                ElaborateErrorKind::UnknownIdentifier,
                format!("unknown identifier `{text}`"),
            )
        })?;
        // 3. Supported constants get default universe levels; anything
        //    else is fenced with the constant's name in the message.
        let levels =
            crate::default_levels::default_levels(id, &info, &self.names).ok_or_else(|| {
                ElaborateError::new(
                    span,
                    ElaborateErrorKind::UnsupportedConstant,
                    format!("`{text}` is not supported by the elaborator in this delivery"),
                )
            })?;
        Ok(KernelExpr::Const(id, levels))
    }

    fn elaborate_nat(&mut self, value: u64, span: Span) -> Result<KernelExpr, ElaborateError> {
        let zero = self.names.intern("Nat.zero");
        let succ = self.names.intern("Nat.succ");
        if self.env.find(zero).is_none() || self.env.find(succ).is_none() {
            return Err(ElaborateError::new(
                span,
                ElaborateErrorKind::UnknownIdentifier,
                "Nat literals require Nat.zero and Nat.succ in scope".to_string(),
            ));
        }
        let mut e = KernelExpr::Const(zero, Vec::new());
        for _ in 0..value {
            e = KernelExpr::app(KernelExpr::Const(succ, Vec::new()), e);
        }
        Ok(e)
    }

    fn elaborate_binop(
        &mut self,
        lhs: &theoria_syntax::Expr,
        op: theoria_syntax::BinOp,
        rhs: &theoria_syntax::Expr,
        span: Span,
        scope: &mut Scope,
    ) -> Result<KernelExpr, ElaborateError> {
        use theoria_syntax::BinOp as B;
        // Compute both operands' unit classes up front and unify/combine
        // them; the check runs before any kernel term is emitted.
        match op {
            B::Add | B::Sub | B::Mul | B::Div | B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge => {
                let l_unit = self.unit_of_surface(scope, lhs)?;
                let r_unit = self.unit_of_surface(scope, rhs)?;
                let _ = self.combine_binop_units(l_unit, r_unit, op, span)?;
            }
            B::Pow => {
                // The base's unit and the exponent's class are checked
                // separately: the base may be a quantity or bare, the
                // exponent must be bare.
                let base_unit = self.unit_of_surface(scope, lhs)?;
                let exp_unit = self.unit_of_surface(scope, rhs)?;
                if let UnitClass::Quantity(_) = exp_unit {
                    return Err(ElaborateError::new(
                        span,
                        ElaborateErrorKind::UnitError(UnitError::UnitMismatch {
                            lhs: Exponents::DIMENSIONLESS,
                            rhs: exp_unit.exponents().unwrap(),
                        }),
                        "exponent of `^` must be a plain Nat, not a quantity".to_string(),
                    ));
                }
                // The base's unit is raised to the exponent literal;
                // `Nat.pow`'s own type does not constrain units, so the
                // result's class is derived by the caller's
                // `unit_of_surface` from the surface syntax. Nothing
                // further to check here.
                let _ = base_unit;
            }
            B::Arrow => {}
        }
        match op {
            B::Add | B::Mul | B::Sub | B::Pow => {
                let name = match op {
                    B::Add => "Nat.add",
                    B::Mul => "Nat.mul",
                    B::Sub => "Nat.sub",
                    B::Pow => "Nat.pow",
                    _ => unreachable!(),
                };
                self.elab_nat_binop(span, name, lhs, rhs, scope)
            }
            B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge => {
                let name = match op {
                    B::Eq => "Nat.beq",
                    B::Ne => "Nat.bne",
                    B::Lt => "Nat.blt",
                    B::Le => "Nat.ble",
                    B::Gt => "Nat.bgt",
                    B::Ge => "Nat.bge",
                    _ => unreachable!(),
                };
                self.elab_nat_binop(span, name, lhs, rhs, scope)
            }
            B::Div => Err(ElaborateError::not_implemented(
                span,
                "division is not supported: Nat has no total division yet",
            )),
            B::Arrow => {
                let dom = self.elaborate_expr(lhs, scope)?;
                let cod = self.elaborate_expr(rhs, scope)?;
                Ok(KernelExpr::Pi(
                    BinderInfo::Default,
                    NameId(u32::MAX),
                    Box::new(dom),
                    Box::new(cod),
                ))
            }
        }
    }

    /// Elaborate a `Nat`-typed binary operator, hinting both operands
    /// with `Nat`.
    ///
    /// The unit check has already run in `elaborate_binop`; this function
    /// performs the actual elaboration and emits the kernel application.
    fn elab_nat_binop(
        &mut self,
        span: Span,
        kernel_name: &str,
        lhs: &theoria_syntax::Expr,
        rhs: &theoria_syntax::Expr,
        scope: &mut Scope,
    ) -> Result<KernelExpr, ElaborateError> {
        let op_id = self.names.intern(kernel_name);
        if self.env.find(op_id).is_none() {
            return Err(ElaborateError::not_implemented(
                span,
                format!("`{kernel_name}` is not in the environment"),
            ));
        }
        let nat = self.names.intern("Nat");
        let nat_ty = KernelExpr::Const(nat, Vec::new());
        // Elaborate both operands. The expected type is `Nat` for the
        // kernel's purposes, which is defeq to any `Quantity(u)`.
        let lhs_k = self.elaborate_expr_checked(lhs, scope, &nat_ty)?;
        let rhs_k = self.elaborate_expr_checked(rhs, scope, &nat_ty)?;
        let op = KernelExpr::Const(op_id, Vec::new());
        Ok(KernelExpr::app(KernelExpr::app(op, lhs_k), rhs_k))
    }

    /// Elaborate `obj.field` by qualified-name resolution first.
    ///
    /// 1. **Flatten.** Walk `obj` down through nested `Field` and `Ident`
    ///    nodes, collecting the head identifier and the trailing field
    ///    names. If any intermediate node is not a `Field` or an `Ident`
    ///    (e.g. a `Call` or `Paren`), this is genuine projection on a
    ///    computed value, which requires records → `NotImplemented`.
    /// 2. **Join.** Concatenate the segments with `"."` into a
    ///    qualified-name string.
    /// 3. **Intern and look up.** If the qualified name resolves globally,
    ///    emit `Const` with default levels. This step always wins when it
    ///    succeeds — locals never shadow qualified names (matching Lean 4:
    ///    a parameter named `Nat` does not block `Nat.rec`).
    /// 4. **Structure projection.** If the qualified name is unknown but
    ///    the head resolves locally, resolve the receiver's structure and
    ///    emit the projection application. Chained access (`p.a.b`)
    ///    elaborates inside-out, tracking each intermediate type
    ///    structurally.
    /// 5. **Otherwise** report `UnknownIdentifier` naming the full path.
    fn elaborate_field(
        &mut self,
        obj: &theoria_syntax::Expr,
        field: &str,
        _field_span: Span,
        span: Span,
        scope: &Scope,
    ) -> Result<KernelExpr, ElaborateError> {
        let mut segments: Vec<String> = Vec::new();
        let mut cur = obj;
        loop {
            match cur {
                theoria_syntax::Expr::Field {
                    obj: inner,
                    field: f,
                    ..
                } => {
                    segments.push(f.clone());
                    cur = inner;
                }
                theoria_syntax::Expr::Ident { text, .. } => {
                    segments.push(text.clone());
                    break;
                }
                _ => {
                    return Err(ElaborateError::new(
                        span,
                        ElaborateErrorKind::CannotInferFieldReceiver,
                        "field access on a computed value needs type inference, not yet implemented"
                            .to_string(),
                    ));
                }
            }
        }
        segments.reverse();
        segments.push(field.to_string());
        let head = segments[0].clone();
        let mut qualified = String::new();
        for (i, seg) in segments.iter().enumerate() {
            if i > 0 {
                qualified.push('.');
            }
            qualified.push_str(seg);
        }

        let id = self.names.intern(&qualified);
        if let Some(info) = self.env.find(id) {
            let levels =
                crate::default_levels::default_levels(id, &info, &self.names).ok_or_else(|| {
                    ElaborateError::new(
                        span,
                        ElaborateErrorKind::UnsupportedConstant,
                        format!(
                            "`{qualified}` is not supported by the elaborator in this delivery"
                        ),
                    )
                })?;
            return Ok(KernelExpr::Const(id, levels));
        }
        if scope.lookup(&head).is_some() {
            return self.elaborate_projection(obj, field, span, scope);
        }
        Err(ElaborateError::new(
            span,
            ElaborateErrorKind::UnknownIdentifier,
            format!("unknown identifier `{qualified}`"),
        ))
    }

    /// Elaborate `obj.field` as a structure projection application.
    ///
    /// The receiver must be a local or a chain of projections rooted at
    /// a local; anything else is [`ElaborateErrorKind::CannotInferFieldReceiver`].
    /// Its type must be a structure application with a matching field,
    /// else [`ElaborateErrorKind::UnknownField`].
    fn elaborate_projection(
        &mut self,
        obj: &theoria_syntax::Expr,
        field: &str,
        span: Span,
        scope: &Scope,
    ) -> Result<KernelExpr, ElaborateError> {
        let (recv_k, recv_ty) = self.elaborate_receiver(obj, span, scope)?;
        let (app, _) = self.apply_projection(recv_k, &recv_ty, field, span)?;
        Ok(app)
    }

    /// Elaborate a projection receiver to an expression plus its type.
    ///
    /// Locals come from the scope; nested field access recurses (the
    /// intermediate type is tracked structurally, so chains need no
    /// kernel inference). Any other shape has no inferable type in this
    /// delivery.
    fn elaborate_receiver(
        &mut self,
        obj: &theoria_syntax::Expr,
        span: Span,
        scope: &Scope,
    ) -> Result<(KernelExpr, KernelExpr), ElaborateError> {
        match obj {
            theoria_syntax::Expr::Ident {
                text,
                span: ident_span,
            } => match scope.lookup(text) {
                Some(idx) => {
                    let ty = scope
                        .lookup_index(idx)
                        .expect("lookup just succeeded")
                        .ty
                        .as_ref()
                        .clone();
                    Ok((KernelExpr::Var(idx), ty))
                }
                None => Err(ElaborateError::new(
                    *ident_span,
                    ElaborateErrorKind::UnknownIdentifier,
                    format!("unknown identifier `{text}`"),
                )),
            },
            theoria_syntax::Expr::Field {
                obj: inner,
                field: f,
                span: inner_span,
                ..
            } => {
                let (inner_k, inner_ty) = self.elaborate_receiver(inner, *inner_span, scope)?;
                self.apply_projection(inner_k, &inner_ty, f, span)
            }
            _ => Err(ElaborateError::new(
                span,
                ElaborateErrorKind::CannotInferFieldReceiver,
                "field access on a computed value needs type inference, not yet implemented"
                    .to_string(),
            )),
        }
    }

    /// Apply one projection step: `recv : T a₁ ... aₖ` and field `f`
    /// become `(T.f a₁ ... aₖ recv, F[T-params := a⃗])`.
    fn apply_projection(
        &mut self,
        recv_k: KernelExpr,
        recv_ty: &KernelExpr,
        field: &str,
        span: Span,
    ) -> Result<(KernelExpr, KernelExpr), ElaborateError> {
        // Peel the receiver type to its head constant and arguments.
        let mut args: Vec<KernelExpr> = Vec::new();
        let mut cur = recv_ty.clone();
        while let KernelExpr::App(f, a) = cur {
            let (ff, aa) = (*f, *a);
            args.push(aa);
            cur = ff;
        }
        let KernelExpr::Const(t_name, _) = cur else {
            return Err(ElaborateError::new(
                span,
                ElaborateErrorKind::UnknownField,
                format!("`{field}` is not a field: the receiver is not a structure application"),
            ));
        };
        let Some(record) = self.struct_records.get(&t_name) else {
            return Err(ElaborateError::new(
                span,
                ElaborateErrorKind::UnknownField,
                format!("`{field}` is not a field of a known structure"),
            ));
        };
        let Some(fld) = record.fields.iter().find(|f| f.name == field) else {
            return Err(ElaborateError::new(
                span,
                ElaborateErrorKind::UnknownField,
                format!("`{}` has no field `{field}`", record.source_name.clone()),
            ));
        };
        if args.len() != record.num_params {
            return Err(ElaborateError::new(
                span,
                ElaborateErrorKind::UnknownField,
                format!(
                    "`{}` expects {} parameter(s), the receiver supplies {}",
                    record.source_name.clone(),
                    record.num_params,
                    args.len(),
                ),
            ));
        }
        args.reverse();
        // Projections are monomorphic by construction.
        let proj = KernelExpr::Const(fld.proj, Vec::new());
        let inst_ty = crate::structure::instantiate_params(&fld.ty, &args);
        let mut app = proj;
        for a in args {
            app = KernelExpr::app(app, a);
        }
        Ok((KernelExpr::app(app, recv_k), inst_ty))
    }

    // -- match ----------------------------------------------------------

    /// Elaborate `match scrutinee: ...` against `expected` as a recursor
    /// application over the scrutinee's inductive.
    fn elaborate_match(
        &mut self,
        m: &theoria_syntax::MatchExpr,
        scope: &mut Scope,
        expected: &KernelExpr,
    ) -> Result<KernelExpr, ElaborateError> {
        // 1. Elaborate the scrutinee.
        let scrut_k = self.elaborate_expr(&m.scrutinee, scope)?;

        // 2. Determine the scrutinee's inductive.
        let ind_info = self.inductive_of(&scrut_k, scope, m.span)?;

        // 3. Collect and validate arms.
        let arms = self.check_match_arms(&m.arms, &ind_info)?;

        // 4. Build the recursor application:
        //
        //    Nat.rec.{1}  (λ _ : Nat.  expected)  minor₀  minor₁  scrut
        //    Bool.rec.{1} (λ _ : Bool. expected)  minorT  minorF  scrut
        //
        //    Universe level hardcoded to Sort 1 (as in delivery 7's `if`).
        let rec_name = self.names.intern(&format!("{}.rec", ind_info.source_name));
        if self.env.find(rec_name).is_none() {
            return Err(ElaborateError::new(
                m.span,
                ElaborateErrorKind::UnsupportedScrutinee,
                format!(
                    "no recursor `{}.rec` in the environment",
                    ind_info.source_name
                ),
            ));
        }
        let ind_ty = KernelExpr::Const(ind_info.name, Vec::new());
        let underscore = self.names.intern("_");
        // The motive body lives under the motive binder, one level
        // below `expected`: shift it down.
        let motive = KernelExpr::Lam(
            BinderInfo::Default,
            underscore,
            Box::new(ind_ty),
            Box::new(shift_vars(expected, 1, 0)),
        );

        // 5. Elaborate one minor premise per constructor, in
        //    constructor-declaration order.
        let minors = self.elaborate_arms(&arms, &ind_info, scope, expected)?;

        let mut app = KernelExpr::Const(rec_name, alloc::vec![Level::Zero.succ()]);
        app = KernelExpr::app(app, motive);
        for minor in minors {
            app = KernelExpr::app(app, minor);
        }
        let rec_app = KernelExpr::app(app, scrut_k);
        Ok(rec_app)
    }

    /// Determine the scrutinee's inductive *syntactically*: examine the
    /// shape of the elaborated scrutinee, not its kernel type.
    ///
    /// The elaborator keeps no `TypedContext` parallel to its `Scope`
    /// (that infrastructure arrives with bidirectional elaboration), so
    /// kernel-level typing of open terms is out of reach. Instead:
    ///
    /// * a local variable resolves through the scope to its already
    ///   elaborated source-level type;
    /// * a constant resolves through its declared type;
    /// * an application peels to its head constant, if it is one.
    ///
    /// Only `Nat` and `Bool` are admitted; anything else is
    /// `UnsupportedScrutinee` (general inductives need universe
    /// inference).
    fn inductive_of(
        &mut self,
        scrut_k: &KernelExpr,
        scope: &Scope,
        span: Span,
    ) -> Result<InductiveInfo, ElaborateError> {
        match scrut_k {
            KernelExpr::Var(idx) => {
                let binding = scope.lookup_index(*idx).ok_or_else(|| {
                    ElaborateError::new(
                        span,
                        ElaborateErrorKind::UnsupportedScrutinee,
                        "cannot determine the inductive of this scrutinee".to_string(),
                    )
                })?;
                self.inductive_from_ty_expr(&binding.ty, span)
            }
            KernelExpr::Const(name, _) => {
                let info = self.env.find(*name).ok_or_else(|| {
                    ElaborateError::new(
                        span,
                        ElaborateErrorKind::UnsupportedScrutinee,
                        "cannot determine the inductive of this scrutinee".to_string(),
                    )
                })?;
                self.inductive_from_ty_expr(info.ty(), span)
            }
            KernelExpr::App(..) => {
                // Peel to the head constant, then peel the head's
                // declared Pi binders by the argument count to reach the
                // result type (e.g. `Nat.ble a b : Bool`). Over-applied
                // heads are rejected; the kernel would reject them
                // opaquely, so fail here instead.
                let (head, nargs) = app_head_count(scrut_k);
                if let KernelExpr::Const(name, _) = head {
                    let info = self.env.find(*name).ok_or_else(|| {
                        ElaborateError::new(
                            span,
                            ElaborateErrorKind::UnsupportedScrutinee,
                            "cannot determine the inductive of this scrutinee".to_string(),
                        )
                    })?;
                    let mut ty: &KernelExpr = info.ty();
                    for _ in 0..nargs {
                        match ty {
                            KernelExpr::Pi(_, _, _, cod) => ty = cod,
                            _ => {
                                return Err(ElaborateError::unsupported_scrutinee(
                                    span,
                                    "cannot determine the inductive of this scrutinee",
                                ));
                            }
                        }
                    }
                    self.inductive_from_ty_expr(ty, span)
                } else {
                    Err(ElaborateError::unsupported_scrutinee(
                        span,
                        "cannot determine the inductive of this scrutinee",
                    ))
                }
            }
            _ => Err(ElaborateError::unsupported_scrutinee(
                span,
                "`match` is only supported on `Nat` and `Bool` in this delivery",
            )),
        }
    }

    /// Read an inductive out of a type expression: peel applications to
    /// the head constant and require an inductive of `Nat`/`Bool` shape.
    fn inductive_from_ty_expr(
        &mut self,
        ty_k: &KernelExpr,
        span: Span,
    ) -> Result<InductiveInfo, ElaborateError> {
        let (head, _) = app_head_count(ty_k);
        let KernelExpr::Const(name, _) = head else {
            return Err(ElaborateError::unsupported_scrutinee(
                span,
                "scrutinee's type is not an inductive family",
            ));
        };
        let info = self.env.find(*name).ok_or_else(|| {
            ElaborateError::unsupported_scrutinee(
                span,
                "scrutinee's type is not an inductive family",
            )
        })?;
        let inductive = match &*info {
            ConstantInfo::Inductive(v) => v,
            _ => {
                return Err(ElaborateError::unsupported_scrutinee(
                    span,
                    "scrutinee's type is not an inductive family",
                ));
            }
        };
        let known = InductiveInfo::from(inductive, &self.env, &self.names, span);
        // Delivery 8's whitelist: only `Nat` and `Bool` are admitted.
        if known.source_name != "Nat" && known.source_name != "Bool" {
            return Err(ElaborateError::unsupported_scrutinee(
                span,
                "match is only supported on `Nat` and `Bool` in this delivery",
            ));
        }
        Ok(known)
    }

    /// Validate arm patterns against the inductive: resolve constructor
    /// names, reject duplicates and double catch-alls, and enforce
    /// exhaustiveness. Returns the arms in source order.
    fn check_match_arms<'a>(
        &mut self,
        arms: &'a [theoria_syntax::MatchArm],
        ind: &InductiveInfo,
    ) -> Result<Vec<ValidatedArm<'a>>, ElaborateError> {
        let mut seen_ctors: BTreeMap<NameId, Span> = BTreeMap::new();
        let mut catch_all: Option<(String, Span)> = None;
        let mut out = Vec::with_capacity(arms.len());

        for arm in arms {
            match &arm.pattern {
                theoria_syntax::Pattern::Wildcard(span) => {
                    if let Some((prev_name, _)) = &catch_all {
                        return Err(ElaborateError::new(
                            arm.span,
                            ElaborateErrorKind::RedundantArm {
                                name: "_".to_string(),
                            },
                            format!(
                                "unreachable `case`: `{prev_name}` already covers every constructor"
                            ),
                        ));
                    }
                    catch_all = Some(("_".to_string(), *span));
                    out.push(ValidatedArm {
                        ctor: None,
                        pattern: &arm.pattern,
                        body: &arm.body,
                    });
                }
                theoria_syntax::Pattern::Var { name, name_span } => {
                    if let Some((prev_name, _)) = &catch_all {
                        return Err(ElaborateError::new(
                            arm.span,
                            ElaborateErrorKind::RedundantArm { name: name.clone() },
                            format!(
                                "unreachable `case`: `{prev_name}` already covers every constructor"
                            ),
                        ));
                    }
                    catch_all = Some((name.clone(), *name_span));
                    out.push(ValidatedArm {
                        ctor: None,
                        pattern: &arm.pattern,
                        body: &arm.body,
                    });
                }
                theoria_syntax::Pattern::Constructor {
                    name,
                    name_span,
                    args,
                    span: _,
                } => {
                    let ctor_id = self.names.intern(name);
                    let ctor_info = self.env.find(ctor_id);
                    let belongs = match ctor_info.as_deref() {
                        Some(ConstantInfo::Constructor(c)) => c.inductive == ind.name,
                        _ => false,
                    };
                    if !belongs {
                        return Err(ElaborateError::new(
                            arm.span,
                            ElaborateErrorKind::UnknownConstructor { name: name.clone() },
                            format!("`{name}` is not a constructor of `{}`", ind.source_name),
                        ));
                    }
                    // Arity check: the pattern must bind exactly as many
                    // fields as the constructor declares. Anything else
                    // would misalign the minor premise; the kernel would
                    // reject it opaquely, so fail here with the pattern's
                    // span instead.
                    let expected_arity = ind.field_arity(ctor_id);
                    if args.len() != expected_arity {
                        return Err(ElaborateError::new(
                            arm.span,
                            ElaborateErrorKind::UnknownConstructor { name: name.clone() },
                            format!(
                                "`{name}` takes {expected_arity} argument(s), pattern binds {}",
                                args.len()
                            ),
                        ));
                    }
                    if let Some(_first) = seen_ctors.insert(ctor_id, *name_span) {
                        return Err(ElaborateError::new(
                            arm.span,
                            ElaborateErrorKind::RedundantArm { name: name.clone() },
                            format!("duplicate `case` for `{name}`"),
                        ));
                    }
                    out.push(ValidatedArm {
                        ctor: Some(ctor_id),
                        pattern: &arm.pattern,
                        body: &arm.body,
                    });
                }
            }
            // A specific constructor after a catch-all is unreachable:
            // the catch-all already covers it.
            if catch_all.is_some()
                && out.last().map(|a: &ValidatedArm<'_>| a.ctor.is_some()) == Some(true)
            {
                let name = match &out.last().unwrap().pattern {
                    theoria_syntax::Pattern::Constructor { name, .. } => name.clone(),
                    _ => unreachable!("ctor arm has a constructor pattern"),
                };
                return Err(ElaborateError::new(
                    arm.span,
                    ElaborateErrorKind::RedundantArm { name: name.clone() },
                    format!("unreachable `case`: a catch-all already covers `{name}`"),
                ));
            }
        }

        if catch_all.is_none() {
            let missing: Vec<String> = ind
                .ctors
                .iter()
                .filter(|c| !seen_ctors.contains_key(c))
                .map(|c| self.names.resolve(*c).to_string())
                .collect();
            if !missing.is_empty() {
                return Err(ElaborateError::new(
                    ind.span,
                    ElaborateErrorKind::NonExhaustiveMatch { missing },
                    "non-exhaustive `match`: some constructors are not covered and there is no catch-all".to_string(),
                ));
            }
        }
        Ok(out)
    }

    /// Elaborate one validated arm into the minor premise for
    /// `target_ctor`: push the pattern's binders, elaborate the body in
    /// checked mode, pop, and wrap in lambdas.
    ///
    /// The minor binds one lambda per constructor field, outermost
    /// first, plus one induction-hypothesis lambda per recursive
    /// field (a field whose type is the inductive itself), innermost.
    /// `Nat.succ` is the only supported constructor with fields, so
    /// its minor is `λ k. λ ih. body`; niladic constructors take no
    /// binders. A catch-all arm grafts its body per constructor: the
    /// bound value is replaced by the reconstructed constructor
    /// application (`Nat.zero`, `Nat.succ n`).
    ///
    /// The IH type is the match's expected type shifted under the
    /// enclosing binders. That is sound because delivery 8 motives
    /// are constant (`λ _ : Ind. expected`), so `motive field` is
    /// `expected`; it must be spelled this way (rather than as the
    /// motive applied to the field) because the kernel infers every
    /// lambda domain, and an application headed by a lambda is not
    /// inferable.
    fn elaborate_arm(
        &mut self,
        arm: &ValidatedArm<'_>,
        ind: &InductiveInfo,
        target_ctor: NameId,
        scope: &mut Scope,
        expected: &KernelExpr,
    ) -> Result<KernelExpr, ElaborateError> {
        let field_tys = self.field_types_of(target_ctor);
        let nfields = field_tys.len();
        let ind_ty = KernelExpr::Const(ind.name, Vec::new());
        // Count of recursive fields (those whose type is the inductive
        // itself): each gets an IH binder.
        let nihs = field_tys.iter().filter(|ty| **ty == ind_ty).count();

        // Binders in declaration order: `Some(name)` for named source
        // binders, `None` for anonymous positions (`_` fields and
        // catch-all grafts, which get an unreferenceable scope entry so
        // that scope depth tracks lambda depth).
        let mut field_names: Vec<Option<String>> = Vec::with_capacity(nfields);
        // How many binders the body is elaborated under.
        let body_depth: u32 = match (arm.ctor, arm.pattern) {
            (Some(ctor), theoria_syntax::Pattern::Constructor { args, .. }) => {
                debug_assert_eq!(ctor, target_ctor);
                for (arg, ty) in args.iter().zip(field_tys.iter()) {
                    match arg {
                        theoria_syntax::PatternArg::Wildcard(_) => {
                            // Anonymous positions occupy a lambda (and a
                            // scope slot, keeping de Bruijn indices
                            // aligned) but bind no name.
                            scope.push(LocalBinding::bare(String::new(), Rc::new(ty.clone())));
                            field_names.push(None);
                        }
                        theoria_syntax::PatternArg::Var { name, name_span } => {
                            self.reject_global_shadow(name, *name_span)?;
                            scope.push(LocalBinding::bare(name.clone(), Rc::new(ty.clone())));
                            field_names.push(Some(name.clone()));
                        }
                    }
                }
                nfields as u32
            }
            (None, theoria_syntax::Pattern::Var { name, name_span }) => {
                let ty = ind_ty.clone();
                self.reject_global_shadow(name, *name_span)?;
                scope.push(LocalBinding::bare(name.clone(), Rc::new(ty)));
                field_names.resize(nfields, None);
                1
            }
            (None, theoria_syntax::Pattern::Wildcard(_)) => {
                field_names.resize(nfields, None);
                0
            }
            _ => unreachable!("validated arms are constructor or catch-all patterns"),
        };

        // The body's expected type lives one scope level per pushed
        // binder above the body: shift it down to the body.
        let body_expected = shift_vars(expected, body_depth, 0);
        let body_k = self.elaborate_expr_checked(arm.body, scope, &body_expected)?;

        for _ in 0..body_depth {
            let _ = scope.pop();
        }

        // Place the body under the full minor binders.
        let mut acc = match (arm.ctor, arm.pattern) {
            (Some(_), _) => shift_vars(&body_k, nihs as u32, 0),
            (None, theoria_syntax::Pattern::Var { .. }) => {
                // Reconstruct the bound value at the final depth:
                // `Ctor f₀ …` with field `fᵢ` at index
                // `nfields + nihs - 1 - i`.
                let mut repl = KernelExpr::Const(target_ctor, Vec::new());
                for i in 0..nfields {
                    repl = KernelExpr::app(
                        repl,
                        KernelExpr::Var(DeBruijnIndex((nfields + nihs - 1 - i) as u32)),
                    );
                }
                let net_extra = (nfields + nihs) as i32 - 1;
                graft_catchall_body(&body_k, net_extra, &repl)
            }
            (None, theoria_syntax::Pattern::Wildcard(_)) => {
                shift_vars(&body_k, (nfields + nihs) as u32, 0)
            }
            _ => unreachable!("validated arms are constructor or catch-all patterns"),
        };

        // Wrap the IH lambdas, innermost last: the j-th IH sees
        // `nfields + j` enclosing binders, so the expected type shifts
        // by that much. (See the doc comment for why the motive
        // application is spelled out as the shifted expected type.)
        for j in 0..nihs {
            let ih_ty = shift_vars(expected, (nfields + j) as u32, 0);
            let ih = self.names.intern("ih");
            acc = KernelExpr::Lam(BinderInfo::Default, ih, Box::new(ih_ty), Box::new(acc));
        }
        // Wrap the field lambdas, outermost first.
        for (name, ty) in field_names.into_iter().zip(field_tys.into_iter()).rev() {
            let nid = match name {
                Some(n) => self.names.intern(&n),
                None => NameId(u32::MAX),
            };
            acc = KernelExpr::Lam(BinderInfo::Default, nid, Box::new(ty), Box::new(acc));
        }
        Ok(acc)
    }

    /// Elaborate one minor premise per constructor of `ind`, in
    /// constructor-declaration order.
    fn elaborate_arms(
        &mut self,
        arms: &[ValidatedArm<'_>],
        ind: &InductiveInfo,
        scope: &mut Scope,
        expected: &KernelExpr,
    ) -> Result<Vec<KernelExpr>, ElaborateError> {
        let mut minors = Vec::with_capacity(ind.num_ctors);
        for ctor_id in &ind.ctors {
            let arm = arms
                .iter()
                .find(|a| a.ctor == Some(*ctor_id))
                .or_else(|| arms.iter().find(|a| a.ctor.is_none()))
                .expect("exhaustiveness checked above");
            minors.push(self.elaborate_arm(arm, ind, *ctor_id, scope, expected)?);
        }
        Ok(minors)
    }

    /// Field types of the four supported constructors. `Nat`/`Bool` have
    /// zero parameters and zero indices, so no substitution is needed.
    ///
    /// // TODO(general-constructor-types): extract from
    /// // `ConstructorVal.base.ty` once `Structure` lands.
    fn field_types_of(&mut self, ctor: NameId) -> Vec<KernelExpr> {
        let nat = self.names.intern("Nat");
        let nat_ty = KernelExpr::Const(nat, Vec::new());
        let zero = self.names.intern("Nat.zero");
        let succ = self.names.intern("Nat.succ");
        let _ = (zero, succ);
        if ctor == succ {
            alloc::vec![nat_ty]
        } else {
            // Nat.zero, Bool.true, Bool.false (and, unreachable here,
            // anything else): no fields.
            Vec::new()
        }
    }

    fn elaborate_lam(
        &mut self,
        params: &[theoria_syntax::LamParam],
        body: &theoria_syntax::Expr,
        scope: &mut Scope,
    ) -> Result<KernelExpr, ElaborateError> {
        // Elaborate each domain in order: the first parameter's type sees
        // no parameter in scope, the second sees only the first, and so
        // on. Binder info is always explicit in delivery 2.
        let mut elaborated: Vec<(NameId, KernelExpr)> = Vec::new();
        for p in params {
            self.reject_global_shadow(&p.name, p.name_span)?;
            let dom = self.elaborate_expr(&p.ty, scope)?;
            let unit = self.classify_type(&dom).exponents();
            let nid = self.names.intern(&p.name);
            let binding = match unit {
                Some(u) => LocalBinding::with_unit(p.name.clone(), Rc::new(dom.clone()), u),
                None => LocalBinding::bare(p.name.clone(), Rc::new(dom.clone())),
            };
            scope.push(binding);
            elaborated.push((nid, dom));
        }
        let mut body_e = self.elaborate_expr(body, scope)?;
        for _ in &elaborated {
            let _ = scope.pop();
        }
        for (nid, dom) in elaborated.into_iter().rev() {
            body_e = KernelExpr::Lam(BinderInfo::Default, nid, Box::new(dom), Box::new(body_e));
        }
        Ok(body_e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use theoria_kernel::DeBruijnIndex;
    use theoria_kernel::Level;
    use theoria_kernel::prelude::build_prelude;
    use theoria_syntax::Expr as SyntaxExpr;

    // ---- helpers -----------------------------------------------------

    fn sp() -> Span {
        Span::empty(0)
    }

    fn s_ident(text: &str) -> SyntaxExpr {
        SyntaxExpr::Ident {
            text: text.to_string(),
            span: sp(),
        }
    }

    fn s_nat(value: u64) -> SyntaxExpr {
        SyntaxExpr::Nat { value, span: sp() }
    }

    fn s_lam(params: Vec<(&str, SyntaxExpr)>, body: SyntaxExpr) -> SyntaxExpr {
        SyntaxExpr::Lam {
            params: params
                .into_iter()
                .map(|(name, ty)| theoria_syntax::LamParam {
                    name: name.to_string(),
                    name_span: sp(),
                    ty,
                    span: sp(),
                })
                .collect(),
            body: Box::new(body),
            span: sp(),
        }
    }

    fn s_call(func: SyntaxExpr, args: Vec<SyntaxExpr>) -> SyntaxExpr {
        SyntaxExpr::Call {
            func: Box::new(func),
            args,
            span: sp(),
        }
    }

    fn mk_fn(
        name: &str,
        params: Vec<(&str, SyntaxExpr)>,
        ret: Option<SyntaxExpr>,
        body: SyntaxExpr,
    ) -> theoria_syntax::FunctionDef {
        theoria_syntax::FunctionDef {
            name: name.to_string(),
            name_span: sp(),
            params: params
                .into_iter()
                .map(|(n, ty)| theoria_syntax::Param {
                    name: n.to_string(),
                    name_span: sp(),
                    ty,
                    span: sp(),
                })
                .collect(),
            return_ty: ret,
            body: alloc::vec![theoria_syntax::Stmt::Return {
                value: body,
                span: sp()
            }],
            span: Span::new(10, 20),
        }
    }

    fn mk_module(fns: Vec<theoria_syntax::FunctionDef>) -> Module {
        Module {
            decl: None,
            imports: Vec::new(),
            structures: Vec::new(),
            functions: fns,
            theorems: Vec::new(),
            span: sp(),
        }
    }

    fn parse_module_src(source: &str) -> Module {
        theoria_parser::parse_module(source).expect("fixture parses")
    }

    /// Elaborate `source` against the standard prelude.
    fn elab(source: &str) -> Result<ElaboratedModule, Vec<ElaborateError>> {
        elaborate_module(&parse_module_src(source))
    }

    fn def_body(env: &GlobalEnv, names: &mut NameTable, text: &str) -> KernelExpr {
        let id = names.intern(text);
        let info = env.find(id).expect("definition present");
        match &*info {
            ConstantInfo::Definition(d) => (*d.body).clone(),
            other => panic!("expected definition, got {other:?}"),
        }
    }

    // ---- trivial bodies ----------------------------------------------

    #[test]
    fn zero_elaborates_to_const() {
        let m = elab("Function zero() -> Nat:\n    return 0\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "zero");
        let nat_zero = names.intern("Nat.zero");
        assert_eq!(body, KernelExpr::Const(nat_zero, Vec::new()));
    }

    #[test]
    fn one_elaborates_to_succ_chain() {
        let m = elab("Function one() -> Nat:\n    return 1\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "one");
        let nat_zero = names.intern("Nat.zero");
        let nat_succ = names.intern("Nat.succ");
        let expected = KernelExpr::app(
            KernelExpr::Const(nat_succ, Vec::new()),
            KernelExpr::Const(nat_zero, Vec::new()),
        );
        assert_eq!(body, expected);
    }

    #[test]
    fn two_params_elaborate_to_nested_pi_lam() {
        let m = elab("Function const2(a: Nat, b: Nat) -> Nat:\n    return b\n").unwrap();
        let mut names = m.names;
        let id = names.intern("const2");
        let info = m.env.find(id).unwrap();
        match &*info {
            ConstantInfo::Definition(d) => {
                assert!(matches!((*d.base.ty).clone(), KernelExpr::Pi(..)));
                match (*d.body).clone() {
                    KernelExpr::Lam(_, _, _, inner) => {
                        assert!(matches!(*inner, KernelExpr::Lam(..)));
                    }
                    other => panic!("expected nested Lam, got {other:?}"),
                }
            }
            other => panic!("expected definition, got {other:?}"),
        }
        assert_eq!(m.functions[0].arity, 2);
    }

    // ---- name resolution ----------------------------------------------

    #[test]
    fn lambda_param_resolves_to_index() {
        // The body is a bare lambda so the kernel can check it against
        // the Pi return type (a β-redex head has no inferrable type).
        let m = elab("Function f() -> Nat -> Nat:\n    return λ (x : Nat). x\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        match body {
            KernelExpr::Lam(_, _, _, b) => {
                assert_eq!(*b, KernelExpr::Var(DeBruijnIndex(0)));
            }
            other => panic!("expected Lam, got {other:?}"),
        }
    }

    #[test]
    fn param_shadowing_global_is_rejected() {
        // λ (Nat : Nat). Nat — the parameter name matches a global, so
        // elaboration fails before resolution could go wrong.
        let errs = elab("Function f() -> Nat:\n    return (λ (Nat : Nat). Nat)(0)\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::ShadowedGlobal { .. }
        ));
    }

    #[test]
    fn unknown_identifier_errors() {
        let errs = elab("Function f() -> Nat:\n    return Foo\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::UnknownIdentifier);
    }

    #[test]
    fn self_reference_is_unknown() {
        let errs = elab("Function f() -> Nat:\n    return f\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::UnknownIdentifier);
    }

    #[test]
    fn forward_reference_is_unknown() {
        let src = "Function f() -> Nat:\n    return g\n\nFunction g() -> Nat:\n    return 0\n";
        let errs = elab(src).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::UnknownIdentifier);
    }

    // ---- lambdas ------------------------------------------------------

    #[test]
    fn identity_lambda_shape() {
        // (λ (n : Nat). n) against Nat -> Nat: domain is the elaborated
        // Nat constant, body is Var(0).
        let m = elab("Function f() -> Nat -> Nat:\n    return λ (n : Nat). n\n").unwrap();
        let mut names = m.names;
        let nat = names.intern("Nat");
        let body = def_body(&m.env, &mut names, "f");
        match body {
            KernelExpr::Lam(_, _, dom, b) => {
                assert_eq!(*dom, KernelExpr::Const(nat, Vec::new()));
                assert_eq!(*b, KernelExpr::Var(DeBruijnIndex(0)));
            }
            other => panic!("expected Lam, got {other:?}"),
        }
    }

    #[test]
    fn two_param_lambda_nests() {
        let m = elab("Function f() -> Nat -> Nat -> Nat:\n    return λ (a : Nat) (b : Nat). a\n")
            .unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        // Lam(a, Lam(b, Var(1)))
        match body {
            KernelExpr::Lam(_, _, _, inner) => match *inner {
                KernelExpr::Lam(_, _, _, b) => {
                    assert_eq!(*b, KernelExpr::Var(DeBruijnIndex(1)));
                }
                other => panic!("expected inner Lam, got {other:?}"),
            },
            other => panic!("expected outer Lam, got {other:?}"),
        }
    }

    // ---- shadowing -----------------------------------------------------

    #[test]
    fn param_shadowing_nat_is_rejected() {
        let errs = elab("Function f(Nat : Nat) -> Nat:\n    return 0\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::ShadowedGlobal { .. }
        ));
    }

    #[test]
    fn param_shadowing_bool_is_rejected() {
        let errs = elab("Function f(Bool : Bool) -> Bool:\n    return Bool.true\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::ShadowedGlobal { .. }
        ));
    }

    #[test]
    fn param_shadowing_by_name_only() {
        // `List` shadows the prelude even though the parameter's own type
        // is `Nat`: the shadow is by name, not by sort.
        let errs = elab("Function f(List : Nat) -> Nat:\n    return 0\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::ShadowedGlobal { .. }
        ));
    }

    #[test]
    fn second_param_shadowing_is_rejected() {
        let errs = elab("Function f(n : Nat, Nat : Nat) -> Nat:\n    return n\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::ShadowedGlobal { .. }
        ));
    }

    #[test]
    fn shadow_error_points_at_param_name() {
        // `Nat` the parameter starts at byte 11 (`Function f(` is 11
        // chars), so the span is 11..14.
        let errs = elab("Function f(Nat : Nat) -> Nat:\n    return 0\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].span, Span::new(11, 14));
        assert!(errs[0].message.contains("rename the parameter"));
    }

    #[test]
    fn ordinary_param_names_accepted() {
        elab("Function f(n : Nat) -> Nat:\n    return n\n").unwrap();
    }

    #[test]
    fn renamed_param_accepted() {
        elab("Function f(Nat0 : Nat) -> Nat:\n    return Nat0\n").unwrap();
    }

    #[test]
    fn param_param_shadowing_permitted() {
        elab("Function f(n : Nat, n : Nat) -> Nat:\n    return n\n").unwrap();
    }

    #[test]
    fn uppercase_param_accepted() {
        elab("Function f(A : Nat) -> Nat:\n    return A\n").unwrap();
        elab("Function g(B : Bool) -> Bool:\n    return B\n").unwrap();
    }

    // ---- Nat.rec (hand-built CST: the parser yields Field for dotted
    // ---- paths, which delivery 2 rejects; these test elaboration
    // ---- independently via bare dotted identifiers) -------------------

    /// `Nat.rec M Z S n` with `M`, `Z`, `S` as syntax exprs.
    fn nat_rec_call(
        motive: SyntaxExpr,
        zc: SyntaxExpr,
        sc: SyntaxExpr,
        major: SyntaxExpr,
    ) -> SyntaxExpr {
        s_call(s_ident("Nat.rec"), alloc::vec![motive, zc, sc, major])
    }

    fn pred_body() -> SyntaxExpr {
        // motive := λ (_ : Nat). Nat
        let motive = s_lam(vec![("_", s_ident("Nat"))], s_ident("Nat"));
        // Z := 0
        let zc = s_nat(0);
        // S := λ (k : Nat) (h : Nat). k
        let sc = s_lam(
            vec![("k", s_ident("Nat")), ("h", s_ident("Nat"))],
            s_ident("k"),
        );
        nat_rec_call(motive, zc, sc, s_ident("n"))
    }

    #[test]
    fn nat_rec_elaborates_with_level_one() {
        let pre = build_prelude();
        let mut elb = Elaborator::new(pre.env, pre.names);
        let mut scope = Scope::new();
        let nat_ty = elb.elaborate_expr(&s_ident("Nat"), &mut scope).unwrap();
        scope.push(LocalBinding::bare("n".to_string(), Rc::new(nat_ty)));
        let e = elb.elaborate_expr(&pred_body(), &mut scope).unwrap();
        // Head is Const(Nat.rec, [1]).
        let mut cur = &e;
        while let KernelExpr::App(f, _) = cur {
            cur = f.as_ref();
        }
        match cur {
            KernelExpr::Const(id, levels) => {
                assert_eq!(*id, elb.names.intern("Nat.rec"));
                assert_eq!(*levels, alloc::vec![Level::Zero.succ()]);
            }
            other => panic!("expected Const head, got {other:?}"),
        }
    }

    #[test]
    fn pred_via_nat_rec_typechecks() {
        let pre = build_prelude();
        let def = mk_fn(
            "pred",
            alloc::vec![("n", s_ident("Nat"))],
            Some(s_ident("Nat")),
            pred_body(),
        );
        let module = mk_module(alloc::vec![def]);
        elaborate_module_with_env(&module, pre.env, pre.names).unwrap();
    }

    #[test]
    fn nontype_motive_is_rejected_by_kernel() {
        // motive := λ (_ : Nat). 0 — a term, not a type.
        let pre = build_prelude();
        let bad_motive = s_lam(vec![("_", s_ident("Nat"))], s_nat(0));
        let body = nat_rec_call(
            bad_motive,
            s_nat(0),
            s_lam(
                vec![("k", s_ident("Nat")), ("h", s_ident("Nat"))],
                s_ident("k"),
            ),
            s_ident("n"),
        );
        let def = mk_fn(
            "bad",
            alloc::vec![("n", s_ident("Nat"))],
            Some(s_ident("Nat")),
            body,
        );
        let module = mk_module(alloc::vec![def]);
        let errs = elaborate_module_with_env(&module, pre.env, pre.names).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    // ---- literals ------------------------------------------------------

    #[test]
    fn zero_literal_desugars() {
        let pre = build_prelude();
        let mut elb = Elaborator::new(pre.env, pre.names);
        let mut scope = Scope::new();
        let e = elb.elaborate_expr(&s_nat(0), &mut scope).unwrap();
        let zero = elb.names.intern("Nat.zero");
        assert_eq!(e, KernelExpr::Const(zero, Vec::new()));
    }

    #[test]
    fn three_literal_desugars_to_succ_chain() {
        let pre = build_prelude();
        let mut elb = Elaborator::new(pre.env, pre.names);
        let mut scope = Scope::new();
        let e = elb.elaborate_expr(&s_nat(3), &mut scope).unwrap();
        let zero = elb.names.intern("Nat.zero");
        let succ = elb.names.intern("Nat.succ");
        let succ_of = |a: KernelExpr| KernelExpr::app(KernelExpr::Const(succ, Vec::new()), a);
        let expected = succ_of(succ_of(succ_of(KernelExpr::Const(zero, Vec::new()))));
        assert_eq!(e, expected);
    }

    #[test]
    fn literal_in_type_position_rejected_by_kernel() {
        // return type `1` elaborates to a succ-chain (a term, not a type).
        let def = mk_fn("f", Vec::new(), Some(s_nat(1)), s_nat(0));
        let module = mk_module(alloc::vec![def]);
        let pre = build_prelude();
        let errs = elaborate_module_with_env(&module, pre.env, pre.names).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    // ---- fences ----------------------------------------------------------

    #[test]
    fn list_nil_resolves_with_default_level() {
        // Hand-built: the parser yields Field for dotted paths, so use a
        // bare dotted identifier to test elaboration independently.
        // List.nil takes no usable position here (its type is a Pi), so
        // elaborate the identifier directly instead of checking a bogus
        // declaration.
        let pre = build_prelude();
        let mut elb = Elaborator::new(pre.env, pre.names);
        let mut scope = Scope::new();
        match elb
            .elaborate_expr(&s_ident("List.nil"), &mut scope)
            .unwrap()
        {
            KernelExpr::Const(id, levels) => {
                assert_eq!(id, elb.names.intern("List.nil"));
                assert_eq!(levels, alloc::vec![Level::Zero.succ()]);
            }
            other => panic!("expected Const, got {other:?}"),
        }
    }

    #[test]
    fn eq_refl_resolves_with_default_level() {
        let pre = build_prelude();
        let mut elb = Elaborator::new(pre.env, pre.names);
        let mut scope = Scope::new();
        match elb.elaborate_expr(&s_ident("Eq.refl"), &mut scope).unwrap() {
            KernelExpr::Const(id, levels) => {
                assert_eq!(id, elb.names.intern("Eq.refl"));
                assert_eq!(levels, alloc::vec![Level::Zero.succ()]);
            }
            other => panic!("expected Const, got {other:?}"),
        }
    }

    #[test]
    fn two_param_recursor_is_unsupported() {
        // Eq.rec, List.rec, Option.rec, and Quot.lift need universe
        // inference; they stay fenced.
        let def = mk_fn("f", Vec::new(), Some(s_ident("Nat")), s_ident("Eq.rec"));
        let module = mk_module(alloc::vec![def]);
        let pre = build_prelude();
        let errs = elaborate_module_with_env(&module, pre.env, pre.names).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::UnsupportedConstant);
    }

    #[test]
    fn str_literal_is_not_implemented() {
        let errs = elab("Function f() -> Nat:\n    return \"hi\"\n").unwrap_err();
        assert_eq!(errs[0].kind, ElaborateErrorKind::NotImplemented);
    }

    #[test]
    fn float_literal_is_not_implemented() {
        let errs = elab("Function f() -> Nat:\n    return 1.5\n").unwrap_err();
        assert_eq!(errs[0].kind, ElaborateErrorKind::NotImplemented);
    }

    #[test]
    fn field_access_on_non_structure_is_unknown_field() {
        // `a` is bound, so `a.b` is genuine projection, not a qualified
        // name — but `Nat` is not a structure with field `b`.
        let errs = elab("Function f(a : Nat) -> Nat:\n    return a.b\n").unwrap_err();
        assert_eq!(errs[0].kind, ElaborateErrorKind::UnknownField);
    }

    #[test]
    fn field_access_on_unknown_head_is_unknown_identifier() {
        // `a` is unbound and `a.b` is not a known qualified name: the
        // full path is reported.
        let errs = elab("Function f() -> Nat:\n    return a.b\n").unwrap_err();
        assert_eq!(errs[0].kind, ElaborateErrorKind::UnknownIdentifier);
        assert!(errs[0].message.contains("a.b"));
    }

    #[test]
    fn index_is_not_implemented() {
        let errs = elab("Function f() -> Nat:\n    return a[0]\n").unwrap_err();
        assert_eq!(errs[0].kind, ElaborateErrorKind::NotImplemented);
    }

    #[test]
    fn unknown_module_rejected() {
        let errs = elab("Import Standard.Units\nFunction f() -> Nat:\n    return 0\n").unwrap_err();
        assert_eq!(errs[0].kind, ElaborateErrorKind::UnknownModule);
    }

    #[test]
    fn prelude_import_is_noop() {
        elab("Import Standard.Prelude\nFunction f() -> Nat:\n    return 0\n").unwrap();
    }

    #[test]
    fn prelude_import_with_names_discarded() {
        elab("Import Standard.Prelude (Nat, Bool)\nFunction f() -> Nat:\n    return 0\n").unwrap();
    }

    // ---- qualified names -----------------------------------------------------

    #[test]
    fn qualified_name_resolves_through_source() {
        // `Nat.zero` parses as Field; the elaborator resolves the dotted
        // path globally.
        let m = elab("Function f() -> Nat:\n    return Nat.zero\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let nat_zero = names.intern("Nat.zero");
        assert_eq!(body, KernelExpr::Const(nat_zero, Vec::new()));
    }

    #[test]
    fn qualified_recursor_resolves_through_source() {
        // Bool.rec fully applied, all dotted paths resolved globally.
        let src = "Function f() -> Bool:\n    return Bool.rec(λ (_ : Bool). Bool, Bool.true, Bool.false, Bool.true)\n";
        let m = elab(src).unwrap();
        assert_eq!(m.functions[0].arity, 0);
    }

    // NOTE (delivery 5): a test lived here asserting that a parameter
    // named `Nat` does not block qualified `Bool.rec` resolution. That
    // scenario is now rejected outright — `ShadowedGlobal` fires on the
    // parameter before any body is elaborated (see the shadowing tests
    // below). Qualified resolution itself is still covered by
    // `qualified_name_resolves_through_source` and
    // `qualified_recursor_resolves_through_source`.

    #[test]
    fn call_head_field_needs_inference() {
        // The head is a Call, not an Ident/Field chain: genuine
        // projection on a computed value.
        let errs = elab("Function f() -> Nat:\n    return f(0).x\n").unwrap_err();
        assert_eq!(errs[0].kind, ElaborateErrorKind::CannotInferFieldReceiver);
    }

    #[test]
    fn paren_head_field_needs_inference() {
        let errs = elab("Function f() -> Nat:\n    return (Nat).zero\n").unwrap_err();
        assert_eq!(errs[0].kind, ElaborateErrorKind::CannotInferFieldReceiver);
    }

    // ---- structures --------------------------------------------------------

    #[test]
    fn structure_point_smoke() {
        let m = elab("Structure Point2D:\n    x : Nat\n    y : Nat\n").unwrap();
        assert_eq!(m.structures.len(), 1);
        assert_eq!(m.structures[0].source_name, "Point2D");
    }

    #[test]
    fn structure_triple_in_environment() {
        let mut m = elab("Structure Point2D:\n    x : Nat\n    y : Nat\n").unwrap();
        for name in [
            "Point2D",
            "Point2D.mk",
            "Point2D.rec",
            "Point2D.x",
            "Point2D.y",
        ] {
            let id = m.names.intern(name);
            assert!(m.env.find(id).is_some(), "{name} missing from environment");
        }
        let st = &m.structures[0];
        assert_eq!(st.num_params, 0);
        assert_eq!(st.num_fields, 2);
    }

    #[test]
    fn structure_projections_typecheck() {
        // The elaborator kernel-checks each projection before insertion;
        // reaching here means `Point2D.x` and `Point2D.y` checked.
        let mut m = elab(
            "Structure Point2D:\n    x : Nat\n    y : Nat\n\nFunction get_x(p : Point2D) -> Nat:\n    return p.x\n",
        )
        .unwrap();
        assert_eq!(m.functions.len(), 1);
        let _ = m.names.intern("Point2D.x");
    }

    #[test]
    fn structure_field_access_shape() {
        // `p.x` elaborates to `Point2D.x` applied to `p`. Source order
        // is irrelevant: structures always elaborate before functions.
        let m = elab(
            "Function get_x(p : Point2D) -> Nat:\n    return p.x\n\nStructure Point2D:\n    x : Nat\n    y : Nat\n",
        )
        .unwrap();
        let mut names = m.names;
        let proj_x = names.intern("Point2D.x");
        let body = def_body(&m.env, &mut names, "get_x");
        // Lam(p, rec-app): unwrap the parameter binder.
        let inner = match body {
            KernelExpr::Lam(_, _, _, inner) => *inner,
            other => panic!("expected Lam, got {other:?}"),
        };
        match inner {
            KernelExpr::App(f, a) => {
                assert_eq!(*f, KernelExpr::Const(proj_x, Vec::new()));
                assert_eq!(*a, KernelExpr::Var(DeBruijnIndex(0)));
            }
            other => panic!("expected App, got {other:?}"),
        }
    }

    #[test]
    fn structure_origin_evaluates() {
        use theoria_kernel::nbe::{Env as NbeEnv, Nbe, TransparencyMode};
        let mut m = elab(
            "Structure Point2D:\n    x : Nat\n    y : Nat\n\nFunction origin() -> Point2D:\n    return Point2D.mk(0, 0)\n",
        )
        .unwrap();
        let nbe = Nbe::new(&m.env, TransparencyMode::Semireducible);
        let origin = KernelExpr::Const(m.names.intern("origin"), Vec::new());
        let got = nbe.quote(0, &nbe.eval(&NbeEnv::empty(), &origin));
        let mk = KernelExpr::Const(m.names.intern("Point2D.mk"), Vec::new());
        let zero = KernelExpr::Const(m.names.intern("Nat.zero"), Vec::new());
        let expected = KernelExpr::app(KernelExpr::app(mk, zero.clone()), zero);
        assert_eq!(got, expected);
    }

    #[test]
    fn structure_projection_evaluates() {
        use theoria_kernel::nbe::{Env as NbeEnv, Nbe, TransparencyMode};
        let mut m = elab(
            "Structure Point2D:\n    x : Nat\n    y : Nat\n\nFunction get_x(p : Point2D) -> Nat:\n    return p.x\n",
        )
        .unwrap();
        let nbe = Nbe::new(&m.env, TransparencyMode::Semireducible);
        let mk = KernelExpr::Const(m.names.intern("Point2D.mk"), Vec::new());
        let get_x = KernelExpr::Const(m.names.intern("get_x"), Vec::new());
        fn nat(names: &mut NameTable, n: u64) -> KernelExpr {
            let mut e = KernelExpr::Const(names.intern("Nat.zero"), Vec::new());
            let succ = KernelExpr::Const(names.intern("Nat.succ"), Vec::new());
            for _ in 0..n {
                e = KernelExpr::app(succ.clone(), e);
            }
            e
        }
        let pt = KernelExpr::app(
            KernelExpr::app(mk, nat(&mut m.names, 3)),
            nat(&mut m.names, 5),
        );
        let got = nbe.quote(0, &nbe.eval(&NbeEnv::empty(), &KernelExpr::app(get_x, pt)));
        assert_eq!(got, nat(&mut m.names, 3));
    }

    #[test]
    fn structure_recursive_wrapper_elaborates() {
        let m = elab(
            "Structure Wrapper:\n    value : Nat\n    inner : Wrapper\n\nFunction peek(w : Wrapper) -> Nat:\n    return w.value\n",
        )
        .unwrap();
        assert_eq!(m.structures[0].source_name, "Wrapper");
        assert_eq!(m.structures[0].num_fields, 2);
    }

    #[test]
    fn structure_chained_access_evaluates() {
        use theoria_kernel::nbe::{Env as NbeEnv, Nbe, TransparencyMode};
        let mut m = elab(
            "Structure Wrapper:\n    value : Nat\n    inner : Wrapper\n\nFunction peek2(w : Wrapper) -> Nat:\n    return w.inner.value\n",
        )
        .unwrap();
        let nbe = Nbe::new(&m.env, TransparencyMode::Semireducible);
        let mk = KernelExpr::Const(m.names.intern("Wrapper.mk"), Vec::new());
        let peek2 = KernelExpr::Const(m.names.intern("peek2"), Vec::new());
        fn nat(names: &mut NameTable, n: u64) -> KernelExpr {
            let mut e = KernelExpr::Const(names.intern("Nat.zero"), Vec::new());
            let succ = KernelExpr::Const(names.intern("Nat.succ"), Vec::new());
            for _ in 0..n {
                e = KernelExpr::app(succ.clone(), e);
            }
            e
        }
        // `w.inner.value` where `w = outer`, `outer.inner = inner`
        // (value 7), `outer.value = 1`. The bottom of the nest is never
        // inspected, so it can be any term (evaluation is untyped).
        let inner = KernelExpr::app(
            KernelExpr::app(mk.clone(), nat(&mut m.names, 7)),
            nat(&mut m.names, 0),
        );
        let outer = KernelExpr::app(KernelExpr::app(mk, nat(&mut m.names, 1)), inner);
        let got = nbe.quote(
            0,
            &nbe.eval(&NbeEnv::empty(), &KernelExpr::app(peek2, outer)),
        );
        // `w.inner.value` where `w.inner = inner` (value 7).
        assert_eq!(got, nat(&mut m.names, 7));
    }

    #[test]
    fn structure_parameterised_box_elaborates() {
        let m = elab(
            "Structure Box(A : Type):\n    contents : A\n\nFunction unwrap(n : Nat) -> Nat:\n    return Box.contents(Nat, Box.mk(Nat, n))\n",
        )
        .unwrap();
        assert_eq!(m.structures[0].num_params, 1);
        assert_eq!(m.structures[0].num_fields, 1);
    }

    #[test]
    fn structure_stream_elaborates() {
        let m = elab("Structure Stream(A : Type):\n    head : A\n    tail : Stream(A)\n").unwrap();
        assert_eq!(m.structures[0].num_params, 1);
        assert_eq!(m.structures[0].num_fields, 2);
    }

    #[test]
    fn structure_rejects_non_sort_parameter() {
        let errs = elab("Structure Foo(n : Nat):\n    x : Nat\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::NonSortParameter);
    }

    #[test]
    fn structure_rejects_negative_field() {
        let errs = elab("Structure Bad:\n    arrow : Bad -> Bad\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::NonPositiveField);
    }

    #[test]
    fn structure_rejects_nested_recursion() {
        // `tail : Stream(Nat)` instantiates the parameter: not direct
        // recursion.
        let errs = elab("Structure Stream(A : Type):\n    head : A\n    tail : Stream(Nat)\n")
            .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::NestedRecursion);
    }

    #[test]
    fn structure_rejects_duplicate_name() {
        let errs = elab(
            "Structure Point2D:\n    x : Nat\n    y : Nat\n\nStructure Point2D:\n    x : Nat\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::DuplicateStructure);
    }

    #[test]
    fn structure_rejects_prelude_collision() {
        let errs = elab("Structure Nat:\n    x : Nat\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::DuplicateStructure);
    }

    #[test]
    fn structure_rejects_duplicate_field() {
        let errs = elab("Structure P:\n    x : Nat\n    x : Nat\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::DuplicateField);
    }

    #[test]
    fn structure_field_reference_is_unknown_identifier() {
        // Fields are never in scope: mentioning an earlier field fails
        // name resolution.
        let errs = elab("Structure P:\n    x : Nat\n    y : x\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::UnknownIdentifier);
    }

    #[test]
    fn structure_unknown_field_access() {
        let errs = elab(
            "Structure Point2D:\n    x : Nat\n    y : Nat\n\nFunction f(p : Point2D) -> Nat:\n    return p.z\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::UnknownField);
    }

    #[test]
    fn structure_field_access_on_call_needs_inference() {
        let errs = elab(
            "Structure Point2D:\n    x : Nat\n    y : Nat\n\nFunction f(p : Point2D) -> Nat:\n    return f(p).x\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::CannotInferFieldReceiver);
    }

    #[test]
    fn structure_name_collision_is_rejected_before_functions() {
        let errs = elab("Structure Nat:\n    x : Nat\n\nFunction f() -> Nat:\n    return 0\n")
            .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::DuplicateStructure);
        assert!(
            errs[0].message.contains("(structure)"),
            "expected the pre-flight message, got: {}",
            errs[0].message
        );
    }

    #[test]
    fn two_structures_same_name_in_module_rejected() {
        let errs = elab("Structure P:\n    x : Nat\n\nStructure P:\n    y : Nat\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::DuplicateStructure);
        assert!(
            errs[0].message.contains("twice in this module"),
            "expected the pre-flight message, got: {}",
            errs[0].message
        );
    }

    #[test]
    fn structure_empty_is_rejected() {
        // The parser guarantees non-empty fields; hand-built CST reaches
        // the elaborator's own gate.
        use theoria_syntax::StructureDef;
        let s = StructureDef {
            name: "Empty".to_string(),
            name_span: Span::empty(0),
            params: Vec::new(),
            fields: Vec::new(),
            span: Span::empty(0),
        };
        let module = Module {
            decl: None,
            imports: Vec::new(),
            structures: alloc::vec![s],
            functions: Vec::new(),
            theorems: Vec::new(),
            span: Span::empty(0),
        };
        let pre = build_prelude();
        let errs = elaborate_module_with_env(&module, pre.env, pre.names).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::EmptyStructure);
    }

    #[test]
    fn structure_recursor_iota_fires() {
        // `Point2D.x (Point2D.mk 3 5)` reduces to `3` through the
        // generated ι-rule: the recursor machinery works end to end.
        use theoria_kernel::nbe::{Env as NbeEnv, Nbe, TransparencyMode};
        let mut m = elab("Structure Point2D:\n    x : Nat\n    y : Nat\n").unwrap();
        let nbe = Nbe::new(&m.env, TransparencyMode::Semireducible);
        fn nat(names: &mut NameTable, n: u64) -> KernelExpr {
            let mut e = KernelExpr::Const(names.intern("Nat.zero"), Vec::new());
            let succ = KernelExpr::Const(names.intern("Nat.succ"), Vec::new());
            for _ in 0..n {
                e = KernelExpr::app(succ.clone(), e);
            }
            e
        }
        let mk = KernelExpr::Const(m.names.intern("Point2D.mk"), Vec::new());
        let proj_x = KernelExpr::Const(m.names.intern("Point2D.x"), Vec::new());
        let pt = KernelExpr::app(
            KernelExpr::app(mk, nat(&mut m.names, 3)),
            nat(&mut m.names, 5),
        );
        let got = nbe.quote(0, &nbe.eval(&NbeEnv::empty(), &KernelExpr::app(proj_x, pt)));
        assert_eq!(got, nat(&mut m.names, 3));
    }

    // ---- kernel errors -----------------------------------------------------

    #[test]
    fn application_of_non_function_is_kernel_error() {
        // Hand-built: Nat.zero applied to Nat.zero.
        let body = s_call(
            s_ident("Nat.zero"),
            alloc::vec![s_call(s_ident("Nat.zero"), vec![])],
        );
        let def = mk_fn("f", Vec::new(), Some(s_ident("Nat")), body);
        assert_eq!(def.span, Span::new(10, 20));
        let module = mk_module(alloc::vec![def]);
        let pre = build_prelude();
        let errs = elaborate_module_with_env(&module, pre.env, pre.names).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
        assert_eq!(errs[0].span, Span::new(10, 20));
    }

    #[test]
    fn return_type_mismatch_is_kernel_error() {
        let errs = elab("Function f() -> Bool:\n    return 0\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    // ---- end to end ----------------------------------------------------------

    #[test]
    fn double_via_hand_built_succ_apps_typechecks() {
        // Parser yields Field for `Nat.succ(n)`; hand-build the equivalent
        // dotted identifiers to test elaboration + kernel checking.
        let body = s_call(
            s_ident("Nat.succ"),
            alloc::vec![s_call(s_ident("Nat.succ"), alloc::vec![s_ident("n")])],
        );
        let def = mk_fn(
            "double",
            alloc::vec![("n", s_ident("Nat"))],
            Some(s_ident("Nat")),
            body,
        );
        let module = mk_module(alloc::vec![def]);
        let pre = build_prelude();
        let m = elaborate_module_with_env(&module, pre.env, pre.names).unwrap();
        assert_eq!(m.functions[0].arity, 1);
    }

    #[test]
    fn id_end_to_end_via_source() {
        let m = elab("Function id(n: Nat) -> Nat:\n    return n\n").unwrap();
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.functions[0].source_name, "id");
    }

    // ---- operators --------------------------------------------------------

    fn nat_op_head(body: &KernelExpr, op: &str, names: &mut NameTable) -> (KernelExpr, KernelExpr) {
        // Split `App(App(Const(op), lhs), rhs)` into `(lhs, rhs)`.
        match body {
            KernelExpr::App(f, rhs) => match &**f {
                KernelExpr::App(g, lhs) => match &**g {
                    KernelExpr::Const(id, _) => {
                        assert_eq!(names.resolve(*id), op);
                        ((**lhs).clone(), (**rhs).clone())
                    }
                    other => panic!("expected Const head, got {other:?}"),
                },
                other => panic!("expected App head, got {other:?}"),
            },
            other => panic!("expected App, got {other:?}"),
        }
    }

    #[test]
    fn add_elaborates_to_const_app() {
        let m = elab("Function f() -> Nat:\n    return 1 + 2\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let _ = nat_op_head(&body, "Nat.add", &mut names);
    }

    #[test]
    fn mul_elaborates_to_const_app() {
        let m = elab("Function f() -> Nat:\n    return 1 * 2\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let _ = nat_op_head(&body, "Nat.mul", &mut names);
    }

    #[test]
    fn sub_elaborates_to_const_app() {
        let m = elab("Function f() -> Nat:\n    return 1 - 2\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let _ = nat_op_head(&body, "Nat.sub", &mut names);
    }

    #[test]
    fn pow_elaborates_to_const_app() {
        let m = elab("Function f() -> Nat:\n    return 1 ^ 2\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let _ = nat_op_head(&body, "Nat.pow", &mut names);
    }

    #[test]
    fn add_mul_precedence_preserved() {
        // 1 + 2 * 3 == Nat.add(1, Nat.mul(2, 3))
        let m = elab("Function f() -> Nat:\n    return 1 + 2 * 3\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let (_, rhs) = nat_op_head(&body, "Nat.add", &mut names);
        let _ = nat_op_head(&rhs, "Nat.mul", &mut names);
    }

    #[test]
    fn pow_right_assoc_preserved() {
        // 2 ^ 3 ^ 2 == Nat.pow(2, Nat.pow(3, 2))
        let m = elab("Function f() -> Nat:\n    return 2 ^ 3 ^ 2\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let (_, rhs) = nat_op_head(&body, "Nat.pow", &mut names);
        let _ = nat_op_head(&rhs, "Nat.pow", &mut names);
    }

    #[test]
    fn add_left_assoc_preserved() {
        // 1 + 2 + 3 == Nat.add(Nat.add(1, 2), 3)
        let m = elab("Function f() -> Nat:\n    return 1 + 2 + 3\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let (lhs, _) = nat_op_head(&body, "Nat.add", &mut names);
        let _ = nat_op_head(&lhs, "Nat.add", &mut names);
    }

    #[test]
    fn arith_functions_typecheck() {
        elab("Function f() -> Nat:\n    return 1 + 2\n").unwrap();
        elab("Function g(n : Nat) -> Nat:\n    return n + 1\n").unwrap();
        elab("Function h(n : Nat) -> Nat:\n    return n * n + n\n").unwrap();
    }

    #[test]
    fn double_via_mul_evaluates_to_six() {
        use theoria_kernel::nbe::{Env as NbeEnv, Nbe, TransparencyMode};
        let mut m = elab("Function double(n : Nat) -> Nat:\n    return 2 * n\n").unwrap();
        let nbe = Nbe::new(&m.env, TransparencyMode::Semireducible);
        let double = m.names.intern("double");
        // double 3, with 3 as a successor chain.
        let three = {
            let zero = KernelExpr::Const(m.names.intern("Nat.zero"), Vec::new());
            let succ = KernelExpr::Const(m.names.intern("Nat.succ"), Vec::new());
            KernelExpr::app(
                succ.clone(),
                KernelExpr::app(succ.clone(), KernelExpr::app(succ, zero)),
            )
        };
        // NOTE: `m.names` is borrowed by `three`; reborrow for lookup.
        let v = nbe.eval(
            &NbeEnv::empty(),
            &KernelExpr::app(KernelExpr::Const(double, Vec::new()), three),
        );
        let back = nbe.quote(0, &v);
        // 2 * 3 = 6: six successors.
        let mut expected = KernelExpr::Const(m.names.intern("Nat.zero"), Vec::new());
        let succ = KernelExpr::Const(m.names.intern("Nat.succ"), Vec::new());
        for _ in 0..6 {
            expected = KernelExpr::app(succ.clone(), expected);
        }
        assert_eq!(back, expected);
    }

    // ---- comparisons --------------------------------------------------------

    #[test]
    fn eq_elaborates_to_beq() {
        let m = elab("Function f() -> Bool:\n    return 1 == 2\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let _ = nat_op_head(&body, "Nat.beq", &mut names);
    }

    #[test]
    fn ne_elaborates_to_bne() {
        let m = elab("Function f() -> Bool:\n    return 1 != 2\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let _ = nat_op_head(&body, "Nat.bne", &mut names);
    }

    #[test]
    fn lt_elaborates_to_blt() {
        let m = elab("Function f() -> Bool:\n    return 1 < 2\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let _ = nat_op_head(&body, "Nat.blt", &mut names);
    }

    #[test]
    fn le_elaborates_to_ble() {
        let m = elab("Function f() -> Bool:\n    return 1 <= 2\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let _ = nat_op_head(&body, "Nat.ble", &mut names);
    }

    #[test]
    fn gt_elaborates_to_bgt() {
        let m = elab("Function f() -> Bool:\n    return 1 > 2\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let _ = nat_op_head(&body, "Nat.bgt", &mut names);
    }

    #[test]
    fn ge_elaborates_to_bge() {
        let m = elab("Function f() -> Bool:\n    return 1 >= 2\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let _ = nat_op_head(&body, "Nat.bge", &mut names);
    }

    #[test]
    fn add_binds_tighter_than_eq() {
        // 1 + 2 == 3  ==  Nat.beq(Nat.add(1, 2), 3)
        let m = elab("Function f() -> Bool:\n    return 1 + 2 == 3\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let (lhs, _) = nat_op_head(&body, "Nat.beq", &mut names);
        let _ = nat_op_head(&lhs, "Nat.add", &mut names);
    }

    #[test]
    fn eq_binds_looser_on_both_sides() {
        // 1 == 2 + 3  ==  Nat.beq(1, Nat.add(2, 3))
        let m = elab("Function f() -> Bool:\n    return 1 == 2 + 3\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        let (_, rhs) = nat_op_head(&body, "Nat.beq", &mut names);
        let _ = nat_op_head(&rhs, "Nat.add", &mut names);
    }

    #[test]
    fn comparison_functions_typecheck() {
        elab("Function f() -> Bool:\n    return 1 == 2\n").unwrap();
        elab("Function is_zero(n : Nat) -> Bool:\n    return n == 0\n").unwrap();
        elab("Function less_than(a : Nat, b : Nat) -> Bool:\n    return a < b\n").unwrap();
    }

    #[test]
    fn bool_result_against_nat_rejected() {
        let errs = elab("Function bad() -> Nat:\n    return 1 == 2\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    #[test]
    fn bool_against_bool_comparison_rejected() {
        // Comparisons are Nat-only: `Nat.beq b b` fails to check.
        let errs = elab("Function f(b : Bool) -> Bool:\n    return b == b\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    #[test]
    fn is_zero_end_to_end_evaluates() {
        use theoria_kernel::nbe::{Env as NbeEnv, Nbe, TransparencyMode};
        let mut m = elab("Function is_zero(n : Nat) -> Bool:\n    return n == 0\n").unwrap();
        let is_zero = m.names.intern("is_zero");
        let bool_true = KernelExpr::Const(m.names.intern("Bool.true"), Vec::new());
        let bool_false = KernelExpr::Const(m.names.intern("Bool.false"), Vec::new());
        fn chain(names: &mut NameTable, n: u64) -> KernelExpr {
            let mut e = KernelExpr::Const(names.intern("Nat.zero"), Vec::new());
            let succ = KernelExpr::Const(names.intern("Nat.succ"), Vec::new());
            for _ in 0..n {
                e = KernelExpr::app(succ.clone(), e);
            }
            e
        }
        let z = chain(&mut m.names, 0);
        let five = chain(&mut m.names, 5);
        let nbe = Nbe::new(&m.env, TransparencyMode::Semireducible);
        let run = |nbe: &Nbe<'_>, arg: KernelExpr| {
            nbe.quote(
                0,
                &nbe.eval(
                    &NbeEnv::empty(),
                    &KernelExpr::app(KernelExpr::Const(is_zero, Vec::new()), arg),
                ),
            )
        };
        assert_eq!(run(&nbe, z), bool_true);
        assert_eq!(run(&nbe, five), bool_false);
    }

    #[test]
    fn negation_still_fenced() {
        let errs = elab("Function f() -> Nat:\n    return -1\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::NotImplemented);
    }

    // ---- if elaboration ----------------------------------------------------

    /// Split `App(App(f, a), b)` into `(f, [a, b])`, borrowing the spine.
    fn app_spine(e: &KernelExpr) -> (&KernelExpr, Vec<&KernelExpr>) {
        let mut args = Vec::new();
        let mut cur = e;
        while let KernelExpr::App(f, a) = cur {
            args.push(&**a);
            cur = &**f;
        }
        args.reverse();
        (cur, args)
    }

    #[test]
    fn if_elaborates_to_bool_rec() {
        let m = elab("Function f(a : Nat, b : Nat) -> Nat:\n    return if a <= b then a else b\n")
            .unwrap();
        let mut names = m.names;
        let bool_rec = names.intern("Bool.rec");
        let bool_id = names.intern("Bool");
        let nat = names.intern("Nat");
        let body = def_body(&m.env, &mut names, "f");
        // Lam(a, Lam(b, rec-app)): unwrap the parameter binders first.
        let rec_app = match body {
            KernelExpr::Lam(_, _, _, inner1) => match *inner1 {
                KernelExpr::Lam(_, _, _, inner2) => *inner2,
                other => panic!("expected inner Lam, got {other:?}"),
            },
            other => panic!("expected outer Lam, got {other:?}"),
        };
        // rec motive then else cond, with motive = λ _ : Bool. Nat,
        // then = Var(1) (a), else = Var(0) (b).
        let (head, args) = app_spine(&rec_app);
        match head {
            KernelExpr::Const(id, levels) => {
                assert_eq!(*id, bool_rec);
                assert_eq!(*levels, alloc::vec![Level::Zero.succ()]);
            }
            other => panic!("expected Const head, got {other:?}"),
        }
        assert_eq!(args.len(), 4);
        match args[0] {
            KernelExpr::Lam(_, _, dom, mot_body) => {
                assert_eq!(**dom, KernelExpr::Const(bool_id, Vec::new()));
                assert_eq!(**mot_body, KernelExpr::Const(nat, Vec::new()));
            }
            other => panic!("expected Lam motive, got {other:?}"),
        }
        assert_eq!(*args[1], KernelExpr::Var(DeBruijnIndex(1)));
        assert_eq!(*args[2], KernelExpr::Var(DeBruijnIndex(0)));
        // cond is Nat.ble applied twice; head check suffices.
        let (cond_head, _) = app_spine(args[3]);
        match cond_head {
            KernelExpr::Const(id, _) => {
                assert_eq!(names.resolve(*id), "Nat.ble");
            }
            other => panic!("expected ble head, got {other:?}"),
        }
    }

    #[test]
    fn if_motive_follows_bool_return_type() {
        // Return type Bool: motive is λ _ : Bool. Bool.
        let m = elab(
            "Function f(a : Nat) -> Bool:\n    return if a <= a then Bool.true else Bool.false\n",
        )
        .unwrap();
        let mut names = m.names;
        let bool_id = names.intern("Bool");
        let body = def_body(&m.env, &mut names, "f");
        // Lam(a, rec-app): unwrap the parameter binder first.
        let rec_app = match body {
            KernelExpr::Lam(_, _, _, inner) => *inner,
            other => panic!("expected Lam, got {other:?}"),
        };
        let (_, args) = app_spine(&rec_app);
        assert_eq!(args.len(), 4);
        match args[0] {
            KernelExpr::Lam(_, _, _, mot_body) => {
                assert_eq!(**mot_body, KernelExpr::Const(bool_id, Vec::new()));
            }
            other => panic!("expected Lam motive, got {other:?}"),
        }
    }

    #[test]
    fn nested_if_elaborates() {
        let m = elab(
            "Function f(a : Nat) -> Nat:\n    return if a <= a then if a <= a then a else a else a\n",
        )
        .unwrap();
        assert_eq!(m.functions[0].arity, 1);
    }

    #[test]
    fn non_bool_condition_rejected_by_kernel() {
        // Elaboration succeeds (the condition is elaborated, not
        // checked); the kernel rejects `Nat` where `Bool` is expected.
        let errs = elab("Function bad() -> Nat:\n    return if 1 then 2 else 3\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    #[test]
    fn mismatched_branches_rejected_by_kernel() {
        let errs = elab("Function bad() -> Nat:\n    return if 1 == 1 then 2 else Bool.true\n")
            .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    #[test]
    fn if_in_infer_position_rejected() {
        let errs = elab(
            "Function f(a : Nat, b : Nat) -> Nat:\n    return Nat.succ(if a <= b then a else b)\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::NotImplemented);
        assert!(errs[0].message.contains("`if` requires an expected type"));
    }

    // ---- let elaboration ---------------------------------------------------

    #[test]
    fn single_let_becomes_kernel_let() {
        let m =
            elab("Function f(n : Nat) -> Nat:\n    let m : Nat = n + 1\n    return m\n").unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        // Lam(n, Let(m, Nat, n+1, m))
        match body {
            KernelExpr::Lam(_, _, _, b) => match *b {
                KernelExpr::Let(_, _, _, _) => {}
                other => panic!("expected Let, got {other:?}"),
            },
            other => panic!("expected Lam, got {other:?}"),
        }
    }

    #[test]
    fn two_lets_chain() {
        let m = elab(
            "Function f(n : Nat) -> Nat:\n    let a : Nat = n\n    let b : Nat = a\n    return b\n",
        )
        .unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        // Lam(n, Let(a, _, _, Let(b, _, _, _)))
        match body {
            KernelExpr::Lam(_, _, _, b) => match *b {
                KernelExpr::Let(_, _, _, inner) => match *inner {
                    KernelExpr::Let(_, _, _, _) => {}
                    other => panic!("expected inner Let, got {other:?}"),
                },
                other => panic!("expected outer Let, got {other:?}"),
            },
            other => panic!("expected Lam, got {other:?}"),
        }
    }

    #[test]
    fn return_sees_both_let_bindings() {
        // `a` is Var(1) and `b` is Var(0) in the return position.
        let m =
            elab("Function f() -> Nat:\n    let a : Nat = 0\n    let b : Nat = a\n    return b\n")
                .unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        match body {
            KernelExpr::Let(_, _, _, inner) => match *inner {
                KernelExpr::Let(_, _, _, b) => {
                    assert_eq!(*b, KernelExpr::Var(DeBruijnIndex(0)));
                }
                other => panic!("expected inner Let, got {other:?}"),
            },
            other => panic!("expected outer Let, got {other:?}"),
        }
    }

    #[test]
    fn let_shadowing_global_rejected() {
        let errs =
            elab("Function f() -> Nat:\n    let Nat : Nat = 0\n    return Nat\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::ShadowedGlobal { .. }
        ));
    }

    #[test]
    fn let_annotation_not_a_type_rejected_by_kernel() {
        // `0` elaborates to a succ-chain (a term, not a type).
        let errs = elab("Function f() -> Nat:\n    let x : 0 = 0\n    return x\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    #[test]
    fn let_binding_used_as_argument() {
        let m = elab("Function f(n : Nat) -> Nat:\n    let m : Nat = n + 1\n    return m * m\n")
            .unwrap();
        assert_eq!(m.functions[0].arity, 1);
    }

    #[test]
    fn let_value_mismatch_rejected_by_kernel() {
        let errs =
            elab("Function f() -> Nat:\n    let x : Nat = Bool.true\n    return x\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    #[test]
    fn body_without_return_rejected() {
        use theoria_syntax::{FunctionDef, Module, Stmt};
        // Hand-built: the parser rejects let-only bodies, so construct
        // the module directly to test the elaborator's own gate.
        let func = FunctionDef {
            name: "f".to_string(),
            name_span: Span::empty(0),
            params: Vec::new(),
            return_ty: Some(s_ident("Nat")),
            body: alloc::vec![Stmt::Let {
                name: "x".to_string(),
                name_span: Span::empty(0),
                ty: s_ident("Nat"),
                value: s_nat(0),
                span: Span::empty(0),
            }],
            span: Span::empty(0),
        };
        let module = Module {
            decl: None,
            imports: Vec::new(),
            structures: Vec::new(),
            functions: alloc::vec![func],
            theorems: Vec::new(),
            span: Span::empty(0),
        };
        let pre = build_prelude();
        let errs = elaborate_module_with_env(&module, pre.env, pre.names).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::BodyNotReturning);
    }

    #[test]
    fn mid_body_return_rejected() {
        use theoria_syntax::{FunctionDef, Module, Stmt};
        let ret = |v| Stmt::Return {
            value: v,
            span: Span::empty(0),
        };
        let func = FunctionDef {
            name: "f".to_string(),
            name_span: Span::empty(0),
            params: Vec::new(),
            return_ty: Some(s_ident("Nat")),
            body: alloc::vec![
                ret(s_nat(0)),
                Stmt::Let {
                    name: "x".to_string(),
                    name_span: Span::empty(0),
                    ty: s_ident("Nat"),
                    value: s_nat(1),
                    span: Span::empty(0),
                },
                ret(s_nat(2)),
            ],
            span: Span::empty(0),
        };
        let module = Module {
            decl: None,
            imports: Vec::new(),
            structures: Vec::new(),
            functions: alloc::vec![func],
            theorems: Vec::new(),
            span: Span::empty(0),
        };
        let pre = build_prelude();
        let errs = elaborate_module_with_env(&module, pre.env, pre.names).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::BodyNotReturning);
    }

    #[test]
    fn let_shadowing_let_permitted() {
        // Inner `n` shadows outer `n`; both elaborate fine.
        let m =
            elab("Function f() -> Nat:\n    let n : Nat = 0\n    let n : Nat = 1\n    return n\n")
                .unwrap();
        let mut names = m.names;
        let body = def_body(&m.env, &mut names, "f");
        // Let(n, _, _, Let(n, _, _, Var(0))) — the return refers to the
        // inner binding.
        match body {
            KernelExpr::Let(_, _, _, inner) => match *inner {
                KernelExpr::Let(_, _, _, b) => {
                    assert_eq!(*b, KernelExpr::Var(DeBruijnIndex(0)));
                }
                other => panic!("expected inner Let, got {other:?}"),
            },
            other => panic!("expected outer Let, got {other:?}"),
        }
    }

    // ---- kernel checking -----------------------------------------------------

    #[test]
    fn min_typechecks() {
        elab("Function min(a : Nat, b : Nat) -> Nat:\n    return if a <= b then a else b\n")
            .unwrap();
    }

    #[test]
    fn let_chain_typechecks() {
        elab("Function f(n : Nat) -> Nat:\n    let m : Nat = n + 1\n    return m * m\n").unwrap();
    }

    #[test]
    fn combined_if_let_typechecks() {
        elab(
            "Function g(n : Nat) -> Nat:\n    let m : Nat = n + 1\n    return if m > 2 then m else n\n",
        )
        .unwrap();
    }

    #[test]
    fn non_bool_condition_rejected() {
        let errs = elab("Function bad() -> Nat:\n    return if 1 then 2 else 3\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    #[test]
    fn mismatched_branches_rejected() {
        let errs = elab("Function bad() -> Nat:\n    return if 1 == 1 then 2 else Bool.true\n")
            .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    #[test]
    fn mismatched_let_annotation_rejected() {
        let errs = elab("Function f() -> Nat:\n    let x : Bool = 0\n    return 0\n").unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    // ---- end to end ------------------------------------------------------------

    #[test]
    fn min_evaluates() {
        use theoria_kernel::nbe::{Env as NbeEnv, Nbe, TransparencyMode};
        let mut m =
            elab("Function min(a : Nat, b : Nat) -> Nat:\n    return if a <= b then a else b\n")
                .unwrap();
        let nbe = Nbe::new(&m.env, TransparencyMode::Semireducible);
        let min = m.names.intern("min");
        fn chain(names: &mut NameTable, n: u64) -> KernelExpr {
            let mut e = KernelExpr::Const(names.intern("Nat.zero"), Vec::new());
            let succ = KernelExpr::Const(names.intern("Nat.succ"), Vec::new());
            for _ in 0..n {
                e = KernelExpr::app(succ.clone(), e);
            }
            e
        }
        let v = nbe.eval(
            &NbeEnv::empty(),
            &KernelExpr::app(
                KernelExpr::app(KernelExpr::Const(min, Vec::new()), chain(&mut m.names, 3)),
                chain(&mut m.names, 5),
            ),
        );
        // min 3 5 = 3.
        assert_eq!(nbe.quote(0, &v), chain(&mut m.names, 3));
    }

    #[test]
    fn double_square_evaluates() {
        use theoria_kernel::nbe::{Env as NbeEnv, Nbe, TransparencyMode};
        let mut m = elab(
            "Function double_square(n : Nat) -> Nat:\n    let m : Nat = n + 1\n    return m * m\n",
        )
        .unwrap();
        let nbe = Nbe::new(&m.env, TransparencyMode::Semireducible);
        let f = m.names.intern("double_square");
        fn chain(names: &mut NameTable, n: u64) -> KernelExpr {
            let mut e = KernelExpr::Const(names.intern("Nat.zero"), Vec::new());
            let succ = KernelExpr::Const(names.intern("Nat.succ"), Vec::new());
            for _ in 0..n {
                e = KernelExpr::app(succ.clone(), e);
            }
            e
        }
        let v = nbe.eval(
            &NbeEnv::empty(),
            &KernelExpr::app(KernelExpr::Const(f, Vec::new()), chain(&mut m.names, 3)),
        );
        // double_square 3 = (3+1)² = 16.
        assert_eq!(nbe.quote(0, &v), chain(&mut m.names, 16));
    }

    // ---- match elaboration ---------------------------------------------------

    #[test]
    fn match_nat_rec_shape() {
        let src = "Function is_zero(n : Nat) -> Bool:\n    return match n:\n        case Nat.zero => Bool.true\n        case Nat.succ _ => Bool.false\n";
        let m = elab(src).unwrap();
        let mut names = m.names;
        let nat_rec = names.intern("Nat.rec");
        let nat = names.intern("Nat");
        let bool_id = names.intern("Bool");
        let body = def_body(&m.env, &mut names, "is_zero");
        // Lam(n, rec-app): unwrap the parameter binder.
        let rec_app = match body {
            KernelExpr::Lam(_, _, _, inner) => *inner,
            other => panic!("expected Lam, got {other:?}"),
        };
        // rec motive minor0 minor1 scrut.
        let (head, args) = app_spine(&rec_app);
        match head {
            KernelExpr::Const(id, levels) => {
                assert_eq!(*id, nat_rec);
                assert_eq!(*levels, alloc::vec![Level::Zero.succ()]);
            }
            other => panic!("expected Const head, got {other:?}"),
        }
        assert_eq!(args.len(), 4);
        // motive = λ _ : Nat. Bool.
        match args[0] {
            KernelExpr::Lam(_, _, dom, mot_body) => {
                assert_eq!(**dom, KernelExpr::Const(nat, Vec::new()));
                assert_eq!(**mot_body, KernelExpr::Const(bool_id, Vec::new()));
            }
            other => panic!("expected Lam motive, got {other:?}"),
        }
        // minor0 = Bool.true (no binders).
        assert_eq!(
            *args[1],
            KernelExpr::Const(names.intern("Bool.true"), Vec::new())
        );
        // minor1 = λ _. Bool.false.
        match args[2] {
            KernelExpr::Lam(_, _, _, inner) => match &**inner {
                KernelExpr::Lam(_, _, _, b) => {
                    assert_eq!(
                        **b,
                        KernelExpr::Const(names.intern("Bool.false"), Vec::new())
                    );
                }
                other => panic!("expected inner Lam, got {other:?}"),
            },
            other => panic!("expected Lam minor, got {other:?}"),
        }
        // scrut = Var(0) (n).
        assert_eq!(*args[3], KernelExpr::Var(DeBruijnIndex(0)));
    }

    #[test]
    fn match_motive_nat() {
        // Return type Nat: motive is λ _ : Nat. Nat.
        let src = "Function pred(n : Nat) -> Nat:\n    return match n:\n        case Nat.zero => Nat.zero\n        case Nat.succ k => k\n";
        let m = elab(src).unwrap();
        let mut names = m.names;
        let nat = names.intern("Nat");
        let body = def_body(&m.env, &mut names, "pred");
        let rec_app = match body {
            KernelExpr::Lam(_, _, _, inner) => *inner,
            other => panic!("expected Lam, got {other:?}"),
        };
        let (_, args) = app_spine(&rec_app);
        assert_eq!(args.len(), 4);
        match args[0] {
            KernelExpr::Lam(_, _, _, mot_body) => {
                assert_eq!(**mot_body, KernelExpr::Const(nat, Vec::new()));
            }
            other => panic!("expected Lam motive, got {other:?}"),
        }
    }

    #[test]
    fn match_succ_binder_shape() {
        // `case Nat.succ k => k` elaborates to `λ k. λ ih. Var(1)`:
        // the succ minor also binds the induction hypothesis, so the
        // field reference shifts by one.
        let src = "Function pred(n : Nat) -> Nat:\n    return match n:\n        case Nat.zero => Nat.zero\n        case Nat.succ k => k\n";
        let m = elab(src).unwrap();
        let mut names = m.names;
        let nat = names.intern("Nat");
        let body = def_body(&m.env, &mut names, "pred");
        let rec_app = match body {
            KernelExpr::Lam(_, _, _, inner) => *inner,
            other => panic!("expected Lam, got {other:?}"),
        };
        let (_, args) = app_spine(&rec_app);
        match args[2] {
            KernelExpr::Lam(_, _, _, inner) => match inner.as_ref() {
                KernelExpr::Lam(_, _, ih_ty, b) => {
                    assert_eq!(**ih_ty, KernelExpr::Const(nat, Vec::new()));
                    assert_eq!(**b, KernelExpr::Var(DeBruijnIndex(1)));
                }
                other => panic!("expected inner Lam minor, got {other:?}"),
            },
            other => panic!("expected Lam minor, got {other:?}"),
        }
    }

    #[test]
    fn match_wildcard_binder_shape() {
        // `case Nat.succ _ => Bool.false` elaborates to
        // `λ _. λ ih. Bool.false`.
        let src = "Function is_zero(n : Nat) -> Bool:\n    return match n:\n        case Nat.zero => Bool.true\n        case Nat.succ _ => Bool.false\n";
        let m = elab(src).unwrap();
        let mut names = m.names;
        let bool_false = names.intern("Bool.false");
        let body = def_body(&m.env, &mut names, "is_zero");
        let rec_app = match body {
            KernelExpr::Lam(_, _, _, inner) => *inner,
            other => panic!("expected Lam, got {other:?}"),
        };
        let (_, args) = app_spine(&rec_app);
        match args[2] {
            KernelExpr::Lam(_, _, _, inner) => match inner.as_ref() {
                KernelExpr::Lam(_, _, _, b) => {
                    assert_eq!(**b, KernelExpr::Const(bool_false, Vec::new()));
                }
                other => panic!("expected inner Lam minor, got {other:?}"),
            },
            other => panic!("expected Lam minor, got {other:?}"),
        }
    }

    #[test]
    fn match_top_var_binds_scrutinee() {
        // `case k => k` grafts the bound value per constructor:
        // `Nat.zero` for zero, `λ n. λ ih. Nat.succ n` for succ.
        let src = "Function f(n : Nat) -> Nat:\n    return match n:\n        case k => k\n";
        let m = elab(src).unwrap();
        let mut names = m.names;
        let nat_zero = names.intern("Nat.zero");
        let nat_succ = names.intern("Nat.succ");
        let body = def_body(&m.env, &mut names, "f");
        let rec_app = match body {
            KernelExpr::Lam(_, _, _, inner) => *inner,
            other => panic!("expected Lam, got {other:?}"),
        };
        let (_, args) = app_spine(&rec_app);
        // Both minors come from the single catch-all arm.
        assert_eq!(*args[1], KernelExpr::Const(nat_zero, Vec::new()));
        match args[2] {
            KernelExpr::Lam(_, _, _, inner) => match inner.as_ref() {
                KernelExpr::Lam(_, _, _, b) => match b.as_ref() {
                    KernelExpr::App(f, a) => {
                        assert_eq!(**f, KernelExpr::Const(nat_succ, Vec::new()));
                        assert_eq!(**a, KernelExpr::Var(DeBruijnIndex(1)));
                    }
                    other => panic!("expected App minor body, got {other:?}"),
                },
                other => panic!("expected inner Lam minor, got {other:?}"),
            },
            other => panic!("expected Lam minor, got {other:?}"),
        }
    }

    #[test]
    fn match_bool_rec_shape() {
        let src = "Function neg(b : Bool) -> Bool:\n    return match b:\n        case Bool.true => Bool.false\n        case Bool.false => Bool.true\n";
        let m = elab(src).unwrap();
        let mut names = m.names;
        let bool_rec = names.intern("Bool.rec");
        let body = def_body(&m.env, &mut names, "neg");
        let rec_app = match body {
            KernelExpr::Lam(_, _, _, inner) => *inner,
            other => panic!("expected Lam, got {other:?}"),
        };
        let (head, args) = app_spine(&rec_app);
        match head {
            KernelExpr::Const(id, _) => assert_eq!(*id, bool_rec),
            other => panic!("expected Const head, got {other:?}"),
        }
        assert_eq!(args.len(), 4);
    }

    #[test]
    fn match_reversed_arms_use_ctor_order() {
        // Arms in reverse source order still elaborate to minors in
        // constructor-declaration order (zero, then succ).
        let src = "Function pred(n : Nat) -> Nat:\n    return match n:\n        case Nat.succ k => k\n        case Nat.zero => Nat.zero\n";
        let m = elab(src).unwrap();
        let mut names = m.names;
        let nat_zero = names.intern("Nat.zero");
        let body = def_body(&m.env, &mut names, "pred");
        let rec_app = match body {
            KernelExpr::Lam(_, _, _, inner) => *inner,
            other => panic!("expected Lam, got {other:?}"),
        };
        let (_, args) = app_spine(&rec_app);
        // minors[0] is the zero case despite source order.
        assert_eq!(*args[1], KernelExpr::Const(nat_zero, Vec::new()));
        match args[2] {
            KernelExpr::Lam(_, _, _, inner) => match inner.as_ref() {
                KernelExpr::Lam(_, _, _, b) => {
                    assert_eq!(**b, KernelExpr::Var(DeBruijnIndex(1)));
                }
                other => panic!("expected inner Lam minor, got {other:?}"),
            },
            other => panic!("expected Lam minor, got {other:?}"),
        }
    }

    #[test]
    fn nested_match_in_arm_elaborates() {
        let src = "Function f(n : Nat) -> Bool:\n    return match n:\n        case Nat.zero => match n:\n            case Nat.zero => Bool.true\n            case Nat.succ _ => Bool.false\n        case Nat.succ _ => Bool.false\n";
        let m = elab(src).unwrap();
        assert_eq!(m.functions[0].arity, 1);
    }

    #[test]
    fn match_in_if_branch_elaborates() {
        let src = "Function f(a : Nat, b : Nat) -> Nat:\n    return if a <= b then match a:\n        case Nat.zero => Nat.zero\n        case Nat.succ k => k\n    else b\n";
        let m = elab(src).unwrap();
        assert_eq!(m.functions[0].arity, 2);
    }

    // ---- match typechecking --------------------------------------------------

    #[test]
    fn match_pred_typechecks() {
        elab(
            "Function pred(n : Nat) -> Nat:\n    return match n:\n        case Nat.zero => Nat.zero\n        case Nat.succ k => k\n",
        )
        .unwrap();
    }

    #[test]
    fn match_min_typechecks() {
        // Application scrutinee: `Nat.ble(a, b)` peels to the `Bool`
        // head. (Surface calls use `f(x, y)`; juxtaposition is not part
        // of the fragment.)
        elab(
            "Function min(a : Nat, b : Nat) -> Nat:\n    return match Nat.ble(a, b):\n        case Bool.true => a\n        case Bool.false => b\n",
        )
        .unwrap();
    }

    #[test]
    fn match_mismatched_arms_rejected() {
        let errs = elab(
            "Function bad(n : Nat) -> Nat:\n    return match n:\n        case Nat.zero => Nat.zero\n        case Nat.succ k => Bool.true\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    // ---- match fences --------------------------------------------------------

    #[test]
    fn match_in_infer_position_rejected() {
        // A `let` value is elaborated in infer mode. (A call argument
        // would be too, but indentation blocks cannot span the
        // line-continued parens: newlines inside brackets emit no
        // `Newline` tokens by lexer design.)
        let errs = elab(
            "Function f(n : Nat) -> Nat:\n    let x : Nat = match n:\n        case Nat.zero => Nat.zero\n        case Nat.succ k => k\n    return x\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::NotImplemented);
        assert!(
            errs[0]
                .message
                .contains("`match` requires an expected type")
        );
    }

    #[test]
    fn match_on_function_type_rejected() {
        // The scrutinee `g` has a Pi type: not an inductive family.
        let errs = elab(
            "Function f(g : Nat -> Nat) -> Nat:\n    return match g:\n        case Nat.zero => Nat.zero\n        case Nat.succ k => k\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::UnsupportedScrutinee);
    }

    #[test]
    fn match_unknown_constructor_rejected() {
        let errs = elab(
            "Function f(n : Nat) -> Nat:\n    return match n:\n        case Nat.zero => Nat.zero\n        case Foo.bar => Nat.zero\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::UnknownConstructor { .. }
        ));
    }

    #[test]
    fn match_missing_ctor_rejected() {
        let errs = elab(
            "Function f(n : Nat) -> Nat:\n    return match n:\n        case Nat.zero => Nat.zero\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        match &errs[0].kind {
            ElaborateErrorKind::NonExhaustiveMatch { missing } => {
                assert_eq!(missing, &alloc::vec!["Nat.succ".to_string()]);
            }
            other => panic!("expected NonExhaustiveMatch, got {other:?}"),
        }
    }

    #[test]
    fn match_duplicate_ctor_rejected() {
        let errs = elab(
            "Function f(n : Nat) -> Nat:\n    return match n:\n        case Nat.zero => Nat.zero\n        case Nat.zero => Nat.zero\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::RedundantArm { .. }
        ));
    }

    #[test]
    fn match_catchall_then_ctor_rejected() {
        let errs = elab(
            "Function f(n : Nat) -> Nat:\n    return match n:\n        case Nat.zero => Nat.zero\n        case _ => Nat.zero\n        case Nat.succ _ => Nat.zero\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::RedundantArm { .. }
        ));
    }

    #[test]
    fn match_single_catchall_accepted() {
        elab(
            "Function f(n : Nat) -> Nat:\n    return match n:\n        case Nat.zero => Nat.zero\n        case _ => Nat.zero\n",
        )
        .unwrap();
    }

    // ---- match end to end ------------------------------------------------------

    #[test]
    fn match_is_zero_evaluates() {
        use theoria_kernel::nbe::{Env as NbeEnv, Nbe, TransparencyMode};
        let mut m = elab(
            "Function is_zero(n : Nat) -> Bool:\n    return match n:\n        case Nat.zero => Bool.true\n        case Nat.succ _ => Bool.false\n",
        )
        .unwrap();
        let nbe = Nbe::new(&m.env, TransparencyMode::Semireducible);
        let is_zero = m.names.intern("is_zero");
        let bool_true = KernelExpr::Const(m.names.intern("Bool.true"), Vec::new());
        let bool_false = KernelExpr::Const(m.names.intern("Bool.false"), Vec::new());
        fn chain(names: &mut NameTable, n: u64) -> KernelExpr {
            let mut e = KernelExpr::Const(names.intern("Nat.zero"), Vec::new());
            let succ = KernelExpr::Const(names.intern("Nat.succ"), Vec::new());
            for _ in 0..n {
                e = KernelExpr::app(succ.clone(), e);
            }
            e
        }
        let run = |nbe: &Nbe<'_>, arg: KernelExpr| {
            nbe.quote(
                0,
                &nbe.eval(
                    &NbeEnv::empty(),
                    &KernelExpr::app(KernelExpr::Const(is_zero, Vec::new()), arg),
                ),
            )
        };
        assert_eq!(run(&nbe, chain(&mut m.names, 0)), bool_true);
        assert_eq!(run(&nbe, chain(&mut m.names, 5)), bool_false);
    }

    #[test]
    fn match_pred_evaluates() {
        use theoria_kernel::nbe::{Env as NbeEnv, Nbe, TransparencyMode};
        let mut m = elab(
            "Function pred(n : Nat) -> Nat:\n    return match n:\n        case Nat.zero => Nat.zero\n        case Nat.succ k => k\n",
        )
        .unwrap();
        let nbe = Nbe::new(&m.env, TransparencyMode::Semireducible);
        let pred = m.names.intern("pred");
        fn chain(names: &mut NameTable, n: u64) -> KernelExpr {
            let mut e = KernelExpr::Const(names.intern("Nat.zero"), Vec::new());
            let succ = KernelExpr::Const(names.intern("Nat.succ"), Vec::new());
            for _ in 0..n {
                e = KernelExpr::app(succ.clone(), e);
            }
            e
        }
        let run = |nbe: &Nbe<'_>, arg: KernelExpr| {
            nbe.quote(
                0,
                &nbe.eval(
                    &NbeEnv::empty(),
                    &KernelExpr::app(KernelExpr::Const(pred, Vec::new()), arg),
                ),
            )
        };
        assert_eq!(run(&nbe, chain(&mut m.names, 0)), chain(&mut m.names, 0));
        assert_eq!(run(&nbe, chain(&mut m.names, 3)), chain(&mut m.names, 2));
    }

    // ---- units -------------------------------------------------------

    #[test]
    fn quantity_simple_unit_elaborates() {
        // `Quantity(m)` elaborates to `Quantity.{1} U_1_0_0_0_0_0_0`.
        let m = elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m)) -> Quantity(m):\n    return r\n",
        )
        .unwrap();
        assert_eq!(m.functions[0].source_name, "f");
    }

    #[test]
    fn quantity_compound_unit_elaborates() {
        // `Quantity(m * s)` — unit multiplication on the argument.
        let m = elab(
            "Import Standard.Prelude\n\nFunction f(x : Quantity(m * s)) -> Quantity(m * s):\n    return x\n",
        )
        .unwrap();
        assert_eq!(m.functions[0].source_name, "f");
    }

    #[test]
    fn quantity_derived_unit_elaborates() {
        // `Quantity(kg * m / s^2)` — derived SI unit of force.
        let m = elab(
            "Import Standard.Prelude\n\nFunction f(F : Quantity(kg * m / s^2)) -> Quantity(kg * m / s^2):\n    return F\n",
        )
        .unwrap();
        assert_eq!(m.functions[0].source_name, "f");
    }

    #[test]
    fn quantity_powers_elaborate() {
        // `Quantity(m^2)` — squared metre.
        let m = elab(
            "Import Standard.Prelude\n\nFunction f(A : Quantity(m^2)) -> Quantity(m^2):\n    return A\n",
        )
        .unwrap();
        assert_eq!(m.functions[0].source_name, "f");
    }

    #[test]
    fn quantity_unknown_base_unit_rejected() {
        let errs = elab(
            "Import Standard.Prelude\n\nFunction f(x : Quantity(zzz)) -> Quantity(m):\n    return x\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::UnitError(theoria_units::UnitError::UnknownBaseUnit(_)),
        ));
    }

    #[test]
    fn quantity_type_mismatch_rejected_at_elaboration() {
        // With `Quantity(u)` defeq to `Nat`, the kernel would accept
        // `return r` regardless of the declared return unit. The
        // elaborator catches the mismatch.
        let errs = elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m)) -> Quantity(s):\n    return r\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::UnitError(theoria_units::UnitError::UnitMismatch { .. }),
        ));
    }

    #[test]
    fn quantity_different_units_are_different_types() {
        // `Quantity(m)` and `Quantity(s)` must be distinct types. This
        // test ensures the canonical-name interning does not collapse
        // them. Now `Quantity` is defeq to `Nat`, the kernel would
        // accept, but the elaborator's unit check rejects.
        let m = elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m)) -> Quantity(s):\n    return r\n",
        )
        .unwrap_err();
        assert_eq!(m.len(), 1);
        assert!(matches!(
            m[0].kind,
            ElaborateErrorKind::UnitError(theoria_units::UnitError::UnitMismatch { .. }),
        ));
    }

    // ---- unit-aware arithmetic ---------------------------------------

    #[test]
    fn quantity_add_same_unit_elaborates() {
        elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m)) -> Quantity(m):\n    return r + r\n",
        )
        .unwrap();
    }

    #[test]
    fn quantity_add_mismatched_units_rejected() {
        let errs = elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m), s : Quantity(s)) -> Quantity(m):\n    return r + s\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::UnitError(theoria_units::UnitError::UnitMismatch { .. }),
        ));
    }

    #[test]
    fn quantity_mul_squares_unit() {
        elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m)) -> Quantity(m^2):\n    return r * r\n",
        )
        .unwrap();
    }

    #[test]
    fn quantity_mul_different_units_multiplies() {
        elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m), s : Quantity(s)) -> Quantity(m * s):\n    return r * s\n",
        )
        .unwrap();
    }

    #[test]
    fn quantity_mul_by_literal_preserves_unit() {
        // `3 * r` where `r : Quantity(m)` — the literal is bare, the
        // result retains the quantity's unit.
        elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m)) -> Quantity(m):\n    return 3 * r\n",
        )
        .unwrap();
    }

    #[test]
    fn quantity_return_unit_mismatch_rejected() {
        // The body's unit is `m * s`; the declared return type is
        // `Quantity(m)`. The kernel would accept both sides (they are
        // both `Nat`); the elaborator rejects the mismatch.
        let errs = elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m), s : Quantity(s)) -> Quantity(m):\n    return r * s\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::UnitError(theoria_units::UnitError::UnitMismatch { .. }),
        ));
    }

    #[test]
    fn quantity_return_bare_to_quantity_accepted() {
        // `return 0` where the declared return is `Quantity(m)` — a bare
        // literal coerces to any unit.
        elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m)) -> Quantity(m):\n    return 0\n",
        )
        .unwrap();
    }

    #[test]
    fn quantity_add_quantity_and_literal_elaborates() {
        // `r + 1` where `r : Quantity(m)` — bare literal coerces.
        elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m)) -> Quantity(m):\n    return r + 1\n",
        )
        .unwrap();
    }

    #[test]
    fn quantity_power_propagates_unit() {
        // `r^2` where `r : Quantity(m)` must classify as
        // `Quantity(m^2)`, not `Bare`. This is what the classifier fix
        // delivers.
        elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m)) -> Quantity(m^2):\n    return r^2\n",
        )
        .unwrap();
    }

    #[test]
    fn quantity_power_mismatch_rejected() {
        // `r^2 + s` where `r : Quantity(m)` and `s : Quantity(s)`. Before
        // the classifier fix, `r^2` classified as Bare and the sum
        // classified as Quantity(s), silently accepting the dimensional
        // error. Now the mismatch fires.
        let errs = elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m), s : Quantity(s)) -> Quantity(s):\n    return r^2 + s\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::UnitError(theoria_units::UnitError::UnitMismatch { .. }),
        ));
    }

    #[test]
    fn quantity_power_of_compound_unit() {
        // `(m / s)^2 = m^2 * s^-2`.
        elab(
            "Import Standard.Prelude\n\nFunction f(v : Quantity(m / s)) -> Quantity(m^2 / s^2):\n    return v^2\n",
        )
        .unwrap();
    }

    #[test]
    fn quantity_non_literal_exponent_rejected() {
        // `r^k` where `k` is a parameter cannot be classified.
        let errs = elab(
            "Import Standard.Prelude\n\nFunction f(r : Quantity(m), k : Nat) -> Quantity(m):\n    return r^k\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::UnitError(theoria_units::UnitError::NonLiteralExponent),
        ));
    }

    #[test]
    fn dimensionless_quantity_elaborates() {
        elab(
            "Import Standard.Prelude\n\nFunction f(x : Quantity(dimensionless)) -> Quantity(dimensionless):\n    return x\n",
        )
        .unwrap();
    }

    #[test]
    fn dimensionless_equivalent_to_unit_ratio() {
        // `m / m` also classifies as dimensionless, so this body
        // typechecks: `x : Quantity(dimensionless)` returned as
        // `Quantity(m / m)`.
        elab(
            "Import Standard.Prelude\n\nFunction f(x : Quantity(dimensionless)) -> Quantity(m / m):\n    return x\n",
        )
        .unwrap();
    }

    #[test]
    fn newton_and_expansion_are_the_same_type() {
        // The critical test: `Quantity(N)` and `Quantity(kg * m / s^2)`
        // must be defeq. If they are not, the body below fails at the
        // kernel.
        elab(
            "Import Standard.Prelude\n\nFunction f(F : Quantity(N)) -> Quantity(kg * m / s^2):\n    return F\n",
        )
        .unwrap();
    }

    #[test]
    fn joule_and_newton_metre_are_the_same_type() {
        elab(
            "Import Standard.Prelude\n\nFunction f(E : Quantity(J)) -> Quantity(N * m):\n    return E\n",
        )
        .unwrap();
    }

    #[test]
    fn hertz_and_inverse_second_are_the_same_type() {
        elab(
            "Import Standard.Prelude\n\nFunction f(freq : Quantity(Hz)) -> Quantity(1 / s):\n    return freq\n",
        )
        .unwrap();
    }

    // ---- theorems ---------------------------------------------------

    #[test]
    fn theorem_refl_nat_registers() {
        let m = elab(
            "Import Standard.Prelude\n\nTheorem refl_nat:\n    Given:\n        n : Nat\n    Show: Eq(Nat, n, n)\n    Proof: Eq.refl(Nat, n)\n    QED\n",
        )
        .unwrap();
        assert_eq!(m.theorems.len(), 1);
        let t = &m.theorems[0];
        assert_eq!(t.source_name, "refl_nat");
        assert_eq!(t.num_given, 1);
        assert_eq!(t.num_assume, 0);
        assert!(m.env.find(t.kernel_name).is_some());
    }

    #[test]
    fn theorem_zero_eq_zero_no_given_registers() {
        let m = elab(
            "Import Standard.Prelude\n\nTheorem zero_eq_zero:\n    Show: Eq(Nat, 0, 0)\n    Proof: Eq.refl(Nat, 0)\n    QED\n",
        )
        .unwrap();
        assert_eq!(m.theorems.len(), 1);
        assert_eq!(m.theorems[0].num_given, 0);
        assert_eq!(m.theorems[0].num_assume, 0);
    }

    #[test]
    fn theorem_with_assume_registers() {
        // `h : Eq(Nat, 0, 0)` and `Proof: h` — the hypothesis is used
        // directly as the proof.
        let m = elab(
            "Import Standard.Prelude\n\nTheorem h_used:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof: h\n    QED\n",
        )
        .unwrap();
        assert_eq!(m.theorems.len(), 1);
        assert_eq!(m.theorems[0].num_assume, 1);
    }

    #[test]
    fn theorem_with_given_and_assume_registers() {
        let m = elab(
            "Import Standard.Prelude\n\nTheorem combine:\n    Given:\n        n : Nat\n    Assume:\n        h : Eq(Nat, n, 0)\n    Show: Eq(Nat, n, 0)\n    Proof: h\n    QED\n",
        )
        .unwrap();
        assert_eq!(m.theorems[0].num_given, 1);
        assert_eq!(m.theorems[0].num_assume, 1);
    }

    #[test]
    fn theorem_proof_mismatch_rejected() {
        // The proof is `Eq.refl(Nat, 0)` but the goal is `Eq(Nat, 1, 1)`.
        // The kernel's defeq check rejects the mismatch.
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem bad:\n    Show: Eq(Nat, 1, 1)\n    Proof: Eq.refl(Nat, 0)\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    #[test]
    fn theorem_duplicate_name_rejected() {
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof: Eq.refl(Nat, 0)\n    QED\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof: Eq.refl(Nat, 0)\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::NotImplemented);
    }

    #[test]
    fn theorem_can_shadow_prelude_name() {
        // The theorem's name `refl_nat` is unique, but a theorem whose
        // name is `Nat.zero` would shadow a prelude constant. That is
        // rejected by the same duplicate-declaration check as functions.
        // `expect_binding_ident` requires a bare identifier, so this
        // should be a parse error.
        let res = theoria_parser::parse_module(
            "Import Standard.Prelude\n\nTheorem Nat.zero:\n    Show: Eq(Nat, 0, 0)\n    Proof: Eq.refl(Nat, 0)\n    QED\n",
        );
        assert!(res.is_err(), "dotted theorem name should be parse error");
    }

    #[test]
    fn second_theorem_can_reference_first() {
        let m = elab(
            "Import Standard.Prelude\n\nTheorem t1:\n    Show: Eq(Nat, 0, 0)\n    Proof: Eq.refl(Nat, 0)\n    QED\n\nTheorem t2:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof: h\n    QED\n",
        )
        .unwrap();
        assert_eq!(m.theorems.len(), 2);
    }

    #[test]
    fn theorem_and_function_coexist() {
        let m = elab(
            "Import Standard.Prelude\n\nFunction f() -> Nat:\n    return 0\n\nTheorem t:\n    Show: Eq(Nat, f(), f())\n    Proof: Eq.refl(Nat, f())\n    QED\n",
        )
        .unwrap();
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.theorems.len(), 1);
    }

    #[test]
    fn theorem_goal_in_prop_accepted() {
        // `Eq(Nat, 0, 0)` lives in `Prop`; the check passes.
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof: Eq.refl(Nat, 0)\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn theorem_prop_goal_with_given_accepted() {
        // The goal's free variable `n` is bound by `Given`. The typed
        // context built for the check must contain `n : Nat`.
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Given:\n        n : Nat\n    Show: Eq(Nat, n, n)\n    Proof: Eq.refl(Nat, n)\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn theorem_prop_goal_with_assume_accepted() {
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof: h\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn theorem_prop_goal_with_given_and_assume_accepted() {
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Given:\n        n : Nat\n    Assume:\n        h : Eq(Nat, n, 0)\n    Show: Eq(Nat, n, 0)\n    Proof: h\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn theorem_goal_nat_rejected() {
        // `Nat` lives in `Sort 1`, not `Sort 0`.
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Nat\n    Proof: Nat.zero\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        match &errs[0].kind {
            ElaborateErrorKind::TheoremGoalNotAProposition { level } => {
                assert!(level.is_some(), "expected a Sort level, got None");
            }
            other => panic!("expected TheoremGoalNotAProposition, got {other:?}"),
        }
    }

    #[test]
    fn theorem_goal_bool_rejected() {
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Bool\n    Proof: Bool.true\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::TheoremGoalNotAProposition { .. },
        ));
    }

    #[test]
    fn theorem_goal_function_type_rejected() {
        // `Nat -> Nat` lives in `Sort 1` (`imax(1, 1) = 1`).
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Nat -> Nat\n    Proof: λ (n : Nat). n\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::TheoremGoalNotAProposition { .. },
        ));
    }

    #[test]
    fn theorem_goal_not_a_type_rejected() {
        // `0` is a `Nat` term, not a type. Its inferred type is `Nat`,
        // not a `Sort`.
        let errs =
            elab("Import Standard.Prelude\n\nTheorem t:\n    Show: 0\n    Proof: 0\n    QED\n")
                .unwrap_err();
        assert_eq!(errs.len(), 1);
        match &errs[0].kind {
            ElaborateErrorKind::TheoremGoalNotAProposition { level } => {
                assert_eq!(*level, None);
            }
            other => panic!("expected TheoremGoalNotAProposition with None, got {other:?}"),
        }
    }

    #[test]
    fn theorem_goal_error_fires_before_proof_error() {
        // Goal is `Nat` (rejected); proof is also bad (`Nat.zero` is not
        // of type `Nat`... wait, it is). The point is that the
        // goal error fires first: the proof would fail for an unrelated
        // reason, but we never get there.
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Nat\n    Proof: Nat.zero\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::TheoremGoalNotAProposition { .. },
        ));
    }

    #[test]
    fn theorem_erases_to_star() {
        // The kernel's `Eraser::erase_declaration` returns `Star` for
        // any `ConstantInfo::Theorem`. This test exercises the full
        // path: elaborator produces a `Theorem`, kernel environment
        // holds it, eraser extracts it.
        use theoria_kernel::erasure::{Eraser, LValue, PropOnlyErasureEnv};

        let m = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof: Eq.refl(Nat, 0)\n    QED\n",
        )
        .unwrap();
        let info = m
            .env
            .find(m.theorems[0].kernel_name)
            .expect("theorem present");
        let env = PropOnlyErasureEnv;
        let eraser = Eraser::new(&m.env, &env);
        let result = eraser
            .erase_declaration(&info)
            .expect("erasure succeeds on a theorem");
        assert!(
            matches!(result, Some(LValue::Star)),
            "expected `Star`, got {result:?}",
        );
    }

    #[test]
    fn theorem_with_binders_erases_to_star() {
        // Same as above but with `Given` and `Assume` binders; the
        // erasure path is unconditional for theorems, but a regression
        // that inspected the type structure would surface here.
        use theoria_kernel::erasure::{Eraser, LValue, PropOnlyErasureEnv};

        let m = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Given:\n        n : Nat\n    Assume:\n        h : Eq(Nat, n, 0)\n    Show: Eq(Nat, n, 0)\n    Proof: h\n    QED\n",
        )
        .unwrap();
        let info = m
            .env
            .find(m.theorems[0].kernel_name)
            .expect("theorem present");
        let env = PropOnlyErasureEnv;
        let eraser = Eraser::new(&m.env, &env);
        let result = eraser.erase_declaration(&info).expect("erasure succeeds");
        assert!(matches!(result, Some(LValue::Star)));
    }

    #[test]
    fn function_still_erases_to_its_body() {
        // Sanity: the erasure of a `Function` (which the elaborator
        // registers as a `Definition`) is not accidentally changed by
        // the theorem-path additions. A trivial identity function
        // erases to a λ.
        use theoria_kernel::erasure::{Eraser, LValue, PropOnlyErasureEnv};

        let m = elab("Import Standard.Prelude\n\nFunction id_nat(n : Nat) -> Nat:\n    return n\n")
            .unwrap();
        let info = m
            .env
            .find(m.functions[0].kernel_name)
            .expect("function present");
        let env = PropOnlyErasureEnv;
        let eraser = Eraser::new(&m.env, &env);
        let result = eraser.erase_declaration(&info).expect("erasure succeeds");
        assert!(matches!(result, Some(LValue::Lam(_))));
    }

    // ---- Declarative proof steps -----------------------------------

    #[test]
    fn single_exact_step_elaborates() {
        // `Exact` becomes the theorem's body directly.
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Exact h\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn let_step_elaborates() {
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Let n : Nat = 0\n        2. Exact h\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn have_step_elaborates() {
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Have g : Eq(Nat, 0, 0) = h\n        2. Exact g\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn let_and_have_chain_elaborates() {
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Let n : Nat = 0\n        2. Have g : Eq(Nat, 0, 0) = h\n        3. Exact g\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn step_using_previous_let_elaborates() {
        // `g`'s proof references `n`, introduced by step 1, both in
        // `g`'s type and in `g`'s proof. With the kernel fix in
        // delivery 19, the let-bound `n` is inlined during evaluation
        // of `g`'s type, so the typechecker sees `Eq(Nat, 0, 0)` where
        // the term has `Eq(Nat, n, n)`.
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Let n : Nat = 0\n        2. Have g : Eq(Nat, n, n) = Eq.refl(Nat, n)\n        3. Exact g\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn step_binding_used_in_show_elaborates() {
        // `Show:` refers to a step-bound `n`. The step chain must
        // inline `n` when checking the final `Exact`.
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Let n : Nat = 0\n        2. Exact Eq.refl(Nat, n)\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn two_step_bindings_chained_elaborate() {
        // `m` from step 1 used in step 2's type, `g` from step 2 used
        // in step 3.
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Let m : Nat = 0\n        2. Have g : Eq(Nat, m, m) = Eq.refl(Nat, m)\n        3. Exact g\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn missing_exact_step_rejected() {
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Let n : Nat = 0\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::InvalidProofStep { step: 1 },
        ));
    }

    #[test]
    fn step_after_exact_rejected() {
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Exact h\n        2. Exact h\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::InvalidProofStep { step: 2 },
        ));
    }

    #[test]
    fn duplicate_step_binding_rejected() {
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Let n : Nat = 0\n        2. Let n : Nat = 0\n        3. Exact Eq.refl(Nat, 0)\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::InvalidProofStep { step: 2 },
        ));
    }

    #[test]
    fn step_binding_shadowing_parameter_rejected() {
        // A step cannot reuse a name from `Given` or `Assume`.
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Given:\n        n : Nat\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Let n : Nat = 0\n        2. Exact Eq.refl(Nat, 0)\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::InvalidProofStep { step: 1 },
        ));
    }

    // ---- Assume: Prop-ness -----------------------------------------

    #[test]
    fn assume_nat_rejected() {
        // `h : Nat` — `Nat` lives in `Sort 1`, not `Sort 0`.
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Nat\n    Show: Eq(Nat, 0, 0)\n    Proof: Eq.refl(Nat, 0)\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::LocalBindingNotAProposition { level: Some(_), .. },
        ));
    }

    #[test]
    fn assume_bool_rejected() {
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Bool\n    Show: Eq(Nat, 0, 0)\n    Proof: Eq.refl(Nat, 0)\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::LocalBindingNotAProposition { level: Some(_), .. },
        ));
    }

    #[test]
    fn assume_prop_accepted() {
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof: h\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn second_assume_sees_first() {
        // `h1 : Eq(Nat, 0, 0)` and `h2 : Eq(Nat, 0, 0)`.
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h1 : Eq(Nat, 0, 0)\n        h2 : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof: h1\n    QED\n",
        )
        .unwrap();
    }

    // ---- Let inference ----------------------------------------------

    #[test]
    fn let_without_annotation_infers_nat() {
        // `Let n = 0` — the value's type `Nat` is inferred.
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Let n = 0\n        2. Exact Eq.refl(Nat, 0)\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn let_without_annotation_and_used_in_proof() {
        // `n` is used in the proof term `Eq.refl(Nat, n)`. Both the
        // inferred type and the step-level body check succeed.
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Let n = 0\n        2. Exact Eq.refl(Nat, n)\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn let_without_annotation_uses_enclosing_parameter() {
        // `Let m = n` where `n : Nat` is a `Given`. The inferred type
        // is `Nat`; the value is the parameter.
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Given:\n        n : Nat\n    Show: Eq(Nat, n, n)\n    Proof:\n        1. Let m = n\n        2. Exact Eq.refl(Nat, n)\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn let_with_annotation_still_works() {
        // The annotated form is unchanged.
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Let n : Nat = 0\n        2. Exact Eq.refl(Nat, n)\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn let_inference_wrong_prop_use_rejected() {
        // `Let n = 0` infers `n : Nat`; using `n` as a proposition
        // fails at the kernel.
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Let n = 0\n        2. Exact n\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    // ---- From step --------------------------------------------------

    #[test]
    fn from_step_uses_assumption() {
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. From h\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn from_step_uses_given() {
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Given:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. From h\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn from_step_uses_have() {
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Have g : Eq(Nat, 0, 0) = Eq.refl(Nat, 0)\n        2. From g\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn from_step_with_wrong_hypothesis_rejected() {
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 1, 1)\n    Proof:\n        1. From h\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    #[test]
    fn from_step_unknown_name_rejected() {
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. From nope\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].kind, ElaborateErrorKind::UnknownIdentifier);
    }

    #[test]
    fn from_step_after_have_chain() {
        // `From` in the middle of a chain, before the final proof step.
        // The final proof step (step 3) is the one that terminates the
        // chain; step 2's `From` is a `Have`-shaped lemma.
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. From h\n        2. Exact h\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        // The second step follows the first `From`, which already
        // supplied the proof.
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::InvalidProofStep { step: 2 },
        ));
    }

    // ---- Have: Prop-ness --------------------------------------------

    #[test]
    fn have_nat_type_rejected() {
        // `Have n : Nat = 0` — the lemma's type is not a proposition.
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Have n : Nat = 0\n        2. Exact Eq.refl(Nat, 0)\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0].kind,
            ElaborateErrorKind::LocalBindingNotAProposition { level: Some(_), .. },
        ));
    }

    #[test]
    fn wrong_exact_proof_rejected() {
        // `Eq.refl(Nat, 1)` is not a proof of `Eq(Nat, 0, 0)`.
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Exact Eq.refl(Nat, 1)\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    #[test]
    fn have_wrong_proof_rejected() {
        let errs = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Have g : Eq(Nat, 0, 0) = Eq.refl(Nat, 1)\n        2. Exact g\n    QED\n",
        )
        .unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0].kind, ElaborateErrorKind::Kernel(_)));
    }

    #[test]
    fn step_justification_ignored() {
        // The justification text is captured but has no effect on
        // elaboration.
        elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Exact h [by hypothesis]\n    QED\n",
        )
        .unwrap();
    }

    #[test]
    fn steps_and_term_mode_are_interchangeable() {
        // The same proof, once as a single term and once as a step
        // block, elaborates identically.
        let m_term = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof: h\n    QED\n",
        )
        .unwrap();
        let m_steps = elab(
            "Import Standard.Prelude\n\nTheorem t:\n    Assume:\n        h : Eq(Nat, 0, 0)\n    Show: Eq(Nat, 0, 0)\n    Proof:\n        1. Exact h\n    QED\n",
        )
        .unwrap();
        assert_eq!(m_term.theorems.len(), 1);
        assert_eq!(m_steps.theorems.len(), 1);
        // Both theorem bodies are the same constant `h`.
        let body_term = m_term
            .env
            .find(m_term.theorems[0].kernel_name)
            .and_then(|i| i.body().cloned());
        let body_steps = m_steps
            .env
            .find(m_steps.theorems[0].kernel_name)
            .and_then(|i| i.body().cloned());
        assert!(body_term.is_some() && body_steps.is_some());
        assert_eq!(body_term, body_steps);
    }
}
