# Project *Theoria*: Engineering Plan & Specification (`PLAN.MD`)

> **A Unified Language for High-Level Computation, Formal Verification, and Scientific Visualization**  
> *Combining the Logical Soundness of Lean 4, the Exploratory Power of Wolfram Alpha, and the Human Elegance of LaTeX/Typst.*

---

## 1. Vision & Core Philosophy

Modern scientific computing is fractured across three incompatible tools:
1. **Interactive Theorem Provers (Lean 4, Coq, Isabelle):** Uncompromising logical rigor, but cryptic lambda-calculus syntax and a steep learning curve.
2. **Computer Algebra Systems (Wolfram Alpha, Mathematica):** Instant answers, discovery, and plotting, but closed-source, heuristic, and prone to mathematical soundness bugs.
3. **Typesetting Engines (LaTeX, Typst):** Beautiful typography and visual presentation, but purely static with zero computational or logical semantics.

**Project *Theoria*** unites these three paradigms into a single language:
* **Strict & Purely Functional:** Predictable call-by-value evaluation, total termination checking, and immutable data structures.
* **Textbook-Grade Readability:** Clean mathematical prose (`Given`, `Assume`, `Show`, `Proof`, `QED`) readable by any scientist or mathematician without programming training.
* **Certified Computation:** Fast symbolic/numerical heuristics paired with a micro-kernel that verifies result certificates through computational reflection (the Poincaré Principle).
* **Zero-Cost Dimensional Analysis:** Compile-time physical units and standard scientific constants built directly into the type system.
* **Dual-Mode Workflow:** An interactive IDLE/REPL for quick Wolfram-style queries and a strict Script Mode (`.theoria`) for formal mathematics, papers, and software.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACES & TOOLING                                  │
│  • Interactive IDLE / REPL (Reedline)              • VS Code Extension (Tower-LSP)      │
│  • WebAssembly Notebook (SolidJS + CodeMirror 6)   • Document Generator (Typst Engine)  │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
┌────────────────────────────────────────────▼────────────────────────────────────────────┐
│                        COMPILER FRONTEND & ELABORATOR                                   │
│  • Parser (Tree-sitter / Chumsky)                  • Diagnostics (Miette / Ariadne)     │
│  • Bidirectional Elaborator                        • Implicit Argument & Class Resolver │
│  • Compile-Time Dimensional Analysis Engine        • Pattern Match Compiler             │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
┌───────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│     SYMBOLIC CAS & AUTOMATION ENGINE      │ │          TACTIC & PROOF ENGINE            │
│ • E-Graphs (egg/egglog) Rewriter          │ │ • Declarative Proof Tree Generator        │
│ • Exact Polynomial CAS (Symbolica)        │ │ • SMT Solvers (Z3, CVC5 via z3-sys)       │
│ • Arbitrary Precision Arithmetic (Rug)    │ │ • Equality Saturation & Decision Procs    │
│ • Numerical BLAS & Tensors (Faer, Candle) │ │ • Counterexample Falsifier Engine         │
└─────────────────────┬─────────────────────┘ └─────────────────────┬─────────────────────┘
                      │                                             │
                      │ Produces Proof Certificates & Witness Terms │
                      └──────────────────────┬──────────────────────┘
                                             │
┌────────────────────────────────────────────▼────────────────────────────────────────────┐
│                       TRUSTED LOGICAL MICRO-KERNEL                                      │
│ • Calculus of Inductive Constructions (CIC) with Cumulative Universes                  │
│ • Definitional Equality & Weak Head Normal Form (WHNF) Normalizer                       │
│ • Inductive Family Positivity & Guarded Termination Checker                             │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
┌───────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│             EXECUTION ENGINE              │ │        VISUAL & DIAGRAM SUBSYSTEM         │
│ • Cranelift JIT / Native Code Generator   │ │ • Constraint Layout Engine (Cassowary)    │
│ • Fast Bytecode Virtual Machine (WASM)    │ │ • 2D/3D Plotters & Vector Engine (Skia)   │
└───────────────────────────────────────────┘ └───────────────────────────────────────────┘
```

---

## 3. Language Specification & Syntax Guide

### 3.1 Data Structures, Types & Pure Functions
```theoria
Module Standard.Geometry.Shapes

Import Standard.Units (m, cm)
Import Standard.Physics.Constants (PI)

-- 1. Refinement Structure
Structure Circle:
    center : Point2D
    radius : Quantity(m)
    Constraint: radius > 0 * m

-- 2. Pure Functional Computation
Function compute_area(c : Circle) -> Quantity(m^2):
    return PI * (c.radius)^2

-- 3. Pattern Matching & Recursion
Function fibonacci(n : Natural) -> Natural:
    match n:
        case 0 => 0
        case 1 => 1
        case k + 2 => fibonacci(k + 1) + fibonacci(k)
```

---

### 3.2 Wolfram-Style Exploratory Queries (`Analyze` & `Solve`)
```theoria
Analyze Function f(x) = (x^2 - 9) / (x - 3):
    Domain: Real
    Compute:
        - Limit at x -> 3
        - Derivative f'(x)
        - Definite Integral from 0 to 6
    Plot:
        Range: x in [-2, 8]
        ShowHoles: True
```

**Multi-Panel Output Produced:**
* **Panel 1 (Symbolic):** Simplified form $f(x) = x + 3$ ($x \neq 3$), $\lim_{x \to 3} f(x) = 6$, $f'(x) = 1$, $\int_0^6 f(x) dx = 36$.
* **Panel 2 (Formal Verification):** Proof certificate checked and verified by micro-kernel.
* **Panel 3 (LaTeX):** `\lim_{x \to 3} \frac{x^2 - 9}{x - 3} = 6`.
* **Panel 4 (Interactive Diagram):** Interactive 2D Cartesian plane highlighting removable singularity at $(3, 6)$.

---

### 3.3 Declarative Proofs (Isar-Style)
```theoria
Theorem sum_of_odds_is_even:
    Given: a, b : Integer
    Assume:
        h1: is_odd(a)
        h2: is_odd(b)
    Show:
        is_even(a + b)

Proof:
    1. By definition of is_odd, there exists k : Integer such that a = 2*k + 1  [from h1]
    2. By definition of is_odd, there exists m : Integer such that b = 2*m + 1  [from h2]
    3. Then a + b = (2*k + 1) + (2*m + 1)                                     [by substitution]
    4. Then a + b = 2*(k + m + 1)                                              [by ring_arithmetic]
    5. Let n = k + m + 1
    6. Then a + b = 2*n                                                        [by substitution]
    7. Therefore is_even(a + b)                                                [by definition of is_even]
QED
```

---

### 3.4 Physics & Chemistry (Units, Constants, Equations)
```theoria
Module PhysicalChemistry

Import Standard.Physics.Constants (R, N_A)
Import Standard.Units (mol, K, m, Pa, J)

Structure GasState:
    n : Quantity(mol)
    T : Quantity(K)
    V : Quantity(m^3)
    Constraint: n > 0 * mol and T > 0 * K and V > 0 * m^3

Theorem isothermal_expansion_work:
    Given:
        gas : GasState
        V_final : Quantity(m^3)
    Assume:
        h1: V_final > gas.V
    Show:
        Work = gas.n * R * gas.T * ln(V_final / gas.V)

Proof:
    1. Work = Integral of ((gas.n * R * gas.T) / V) with respect to V from gas.V to V_final
    2. Since gas.n, R, gas.T are constant, Work = (gas.n * R * gas.T) * Integral of (1/V) from gas.V to V_final
    3. We know Integral of (1/V) from gas.V to V_final = ln(V_final / gas.V) [by standard_integral]
    4. Therefore Work = gas.n * R * gas.T * ln(V_final / gas.V)             [by substitution]
QED
```

---

### 3.5 Declarative Diagrams & Visual Layout
```theoria
Diagram InscribedPolygon:
    Geometry:
        O: Point at (0, 0)
        C: Circle(center=O, radius=5.0)
        P: RegularPolygon(inscribed_in=C, sides=6)
    Render:
        Draw C with stroke="gray", stroke_style="dashed"
        Draw P with stroke="blue", fill="rgba(0, 0, 255, 0.1)"
        Draw Points(P.vertices) with color="red", size=4
        Draw Label("Radius R = 5") on segment(O, P.vertices[0])
```

---

## 4. Execution Modes & Developer Workflow

### Mode 1: Interactive IDLE / REPL
* **Command:** `theoria`
* **Features:** Multi-line input, instant mathematical calculations, ASCII / Sixel / Web terminal plots, inline step explanations, type query commands (`:type`, `:latex`, `:steps`, `:prove`).

```theoria
$ theoria
>>> solve x^2 - 5*x + 6 == 0 for x
Roots: x = 2, x = 3
Verified: ✓ [Certificate Generated]

>>> plot [sin(x), cos(x)] for x in [-PI, PI]
[Interactive Plot Canvas Opened in Side Panel]
```

### Mode 2: Batch Script Mode
* **File Extensions:** `.theoria` or `.th`
* **Commands:**
  * `theoria check file.th` — Formally verifies all proofs, types, and dimensions (without running numerical loops).
  * `theoria run file.th` — Runs the executable functional code and renders plots.
  * `theoria test file.th` — Executes formal test suites and counterexample searches.
  * `theoria compile --target pdf file.th` — Compiles the file into a formatted paper (via Typst).
  * `theoria compile --target wasm file.th` — Compiles logic and code into a web package.

---

## 5. Type System & Built-in Library

### 5.1 Type Hierarchy

| Classification | Types | Description |
| :--- | :--- | :--- |
| **Scalars** | `Natural`, `Integer`, `Rational`, `Real`, `Complex`, `Boolean` | Infinite precision, exact symbolic representations, and arbitrary-precision floats. |
| **Refinement Types** | `PositiveReal`, `NonZeroInt`, `Interval[a, b]`, `{x: T \| P(x)}` | Subtypes with predicates validated at compile time. |
| **Physical Dimensions** | `Quantity(Unit)` (e.g., `Quantity(kg*m/s^2)`) | Zero-cost compile-time dimensional analysis with SI 7-tuple exponent tracker. |
| **Linear Algebra** | `Vector[T, N]`, `Matrix[T, Rows, Cols]`, `Tensor[T, Dims...]` | Compile-time dimension and shape tracking. |
| **Discrete Structures**| `List[T]`, `Set[T]`, `Map[K, V]`, `Graph[V, E]`, `Tree[T]` | Pure, persistent, immutable data structures. |
| **Logic & Proofs** | `Prop`, `Proof[P]`, `Eq[A, a, b]`, `Witness[x, P(x)]` | First-class dependent types for constructive mathematical proofs. |
| **Visual Types** | `Point2D`, `Point3D`, `Shape`, `Plot2D`, `Plot3D`, `Diagram` | First-class scene-graph objects. |

---

### 5.2 Standard Functions Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               STANDARD LIBRARY CATALOG                                 │
├────────────────────┬────────────────────┬──────────────────────┬───────────────────────┤
│ Calculus & Algebra │ Linear Algebra     │ Physics & Chemistry  │ ML & Statistics       │
│ • diff(f, x)       │ • det(M)           │ • Units (m, s, kg...)│ • grad, jacobian      │
│ • integrate(f, x)  │ • inv(M)           │ • Constants (c, h...)│ • hessian, autodiff   │
│ • limit(f, x, a)   │ • eigenvalues(M)   │ • ideal_gas_law(...) │ • Tensor operations   │
│ • taylor(f, x, n)  │ • eigenvectors(M)  │ • stoichiometric_bal │ • Normal, Poisson     │
│ • simplify(expr)   │ • svd(M)           │ • chemical_eq_solver │ • hypothesis_tests    │
│ • factor(expr)     │ • dot(u, v)        │ • gibbs_free_energy  │ • linear_regression   │
│ • solve(eq, var)   │ • cross(u, v)      │ • arrhenius_rate     │ • loss_functions      │
│ • solve_ode(eq, y) │ • qr_decompose(M)  │ • maxwell_boltzmann  │ • conv2d, relu, adam  │
└────────────────────┴────────────────────┴──────────────────────┴───────────────────────┘
```

---

## 6. Rust Workspace Crate Architecture

The system is organized into a modular, high-performance Rust workspace:

```
theoria/
├── Cargo.toml
├── crates/
│   ├── theoria_kernel/       # Calculus of Inductive Constructions (CIC) verifier
│   ├── theoria_syntax/       # Concrete/Abstract Syntax Tree, Spans, AST definitions
│   ├── theoria_parser/       # Tree-sitter & Chumsky parser implementations
│   ├── theoria_diagnostics/  # Miette/Ariadne error reporting with source spans
│   ├── theoria_elaborator/   # Bidirectional type checker, implicit resolution
│   ├── theoria_units/        # Compile-time dimensional analysis & SI unit engine
│   ├── theoria_cas/          # Egg/Egglog E-graphs, Symbolica algebraic simplifier
│   ├── theoria_solvers/      # SMT bindings (Z3, CVC5), Gröbner basis, Simplex
│   ├── theoria_num/          # Faer linear algebra, Rug precision, ODE integrators
│   ├── theoria_tensor/       # Burn/Candle tensor backend with autodiff
│   ├── theoria_diagrams/     # Cassowary constraint solver & Tiny-Skia renderer
│   ├── theoria_typst/        # Typst-powered mathematical typesetting exporter
│   ├── theoria_vm/           # Cranelift JIT compiler & bytecode interpreter
│   ├── theoria_lsp/          # Tower-LSP Language Server implementation
│   └── theoria_cli/          # Reedline REPL (IDLE) and command-line runner
├── web/                      # SolidJS + WASM Web Notebook UI
└── tests/                    # End-to-end mathematical verification suites
```

---

## 7. Concrete Tech Stack Selection

| Subsystem | Primary Technology | Justification |
| :--- | :--- | :--- |
| **Host Language** | **Rust (Edition 2024)** | Zero-cost abstractions, memory safety, thread safety for parallel proof checks, and WebAssembly compilation. |
| **Parsing & CST** | **Tree-sitter & Chumsky** | Incremental parsing for IDE responsiveness combined with expressive parser combinators for syntax validation. |
| **Diagnostics** | **Miette** | Modern, human-readable compiler error reporting with source-code snippets and contextual help. |
| **Logic Micro-Kernel** | **Custom Rust CIC** | Compact, mathematically auditable kernel (~2,500 lines of code) implementing dependent type verification. |
| **Term Rewriting** | **`egg` & `egglog`** | State-of-the-art equality saturation via E-graphs for algebraic simplification and trig rewrites. |
| **Symbolic Algebra** | **Symbolica** | High-speed multivariate polynomial algebra and rational function operations. |
| **Arbitrary Precision** | **`rug` (GMP, MPFR, MPC)** | Infinite-precision integers, exact rationals, and arbitrary-precision floats without rounding bugs. |
| **Automated SMT** | **Z3 (`z3-rs`) & CVC5** | First-order logic theorem proving and automated goal discharging. |
| **Linear Algebra Core** | **`faer` (Pure Rust)** | Native, parallelized linear algebra engine outperforming OpenBLAS in pure safe Rust. |
| **Tensors & Autodiff** | **`burn` / `candle`** | First-class tensor calculus, GPU execution (CUDA, Metal, WGPU), and automatic differentiation. |
| **Geometric Layout** | **`cassowary-rs` & `argmin`** | Optimization-based constraint solving for declarative geometric diagrams (Penrose-style). |
| **Vector Rendering** | **`tiny-skia` & `resvg`** | Lightweight, high-fidelity 2D rendering pipeline outputting SVG and PNG. |
| **Typesetting** | **Typst** | Fast, modern typesetting engine for compiling mathematical documents and PDF exports. |
| **JIT Code Generation**| **Cranelift** | Low-overhead code generator compiling functional expressions into native machine code. |
| **REPL Engine** | **`reedline`** | Shell engine with multi-line editing, bracket matching, history, and completions. |
| **LSP Server** | **`tower-lsp`** | Standard Language Server Protocol backend for VS Code, Neovim, and Zed. |
| **Frontend UI** | **SolidJS + CodeMirror 6 + WASM**| Reactive, lightweight web notebook interface running the full compiler client-side. |

---

## 8. Potential Features & Advanced Improvements

### 8.1 Counterexample Generation via Falsification Engine
* **Concept:** Before attempting an expensive formal proof search, the compiler runs an automated **counterexample finder** (using SMT bounded model checking and QuickCheck-style randomized property testing).
* **Behavior:** If a user attempts to prove a false statement (e.g., $a^2 + b^2 = (a + b)^2$), the compiler immediately responds with a concrete counterexample:  
  `Falsified: For a = 1, b = 2, LHS = 5 ≠ RHS = 9`.

### 8.2 AI/LLM Neural Tactic Engine (Hybrid Neuro-Symbolic Prover)
* **Concept:** Integrate local, privacy-preserving LLM backends (via `llama.cpp` bindings in Rust) trained on formal mathematics datasets.
* **Capabilities:**
  * **Informal-to-Formal Math Translation:** Translates LaTeX papers or textbook proofs into valid `.theoria` scripts.
  * **Proof Step Suggestion:** Proposes the next logical deduction step when the user is stuck in interactive proof mode.
  * **Soundness Guarantee:** Because the suggestions must pass through the strict trusted micro-kernel, the AI can hallucinate tactics, but it can **never prove a false theorem**.

### 8.3 Interactive ProofWidgets & 3D Manipulables
* **Concept:** Render interactive visual widgets directly inside the editor (VS Code / Web Notebook).
* **Behavior:** When working on a geometric or multivariable calculus proof, the user can click and drag points/surfaces in an interactive 3D WebGL viewport. The algebraic equations, coordinates, and proof obligations update dynamically in real time.

### 8.4 Certified Code Generation & CUDA/C Synthesis
* **Concept:** Once an algorithm (e.g., Runge-Kutta integrator, Kalman filter, or Neural Network forward pass) is formally proven correct, *Theoria* can export it as:
  * Pure, dependency-free **C99 code** (for embedded/mission-critical aerospace systems).
  * High-performance **CUDA kernels** or **Rust crates**.
  * **Verification Guarantee:** Mathematical invariants verified in *Theoria* are preserved in the generated target code.

### 8.5 Literate Scientific Publishing (Automatic Overleaf/Paper Export)
* **Concept:** A `.theoria` file serves as the single source of truth for both computation, formal proof, and publishing.
* **Output:** Running `theoria compile --target paper` generates a complete, publication-ready PDF (via Typst) with interactive diagrams, verified proofs, and dynamic figures, eliminating the need to sync separate LaTeX, Python, and Mathematica files.

---

## 9. Phased Implementation Roadmap

```
Phase 1: Logic Core       Phase 2: Frontend & VM    Phase 3: CAS & SMT
(Months 1-3)             (Months 4-6)              (Months 7-9)
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│ • Pure CIC Kernel     │ │ • Indentation Parser  │ │ • Egg E-Graphs        │
│ • Definitional WHNF   │ │ • Bidirectional Elab  │ │ • Symbolica Poly CAS  │
│ • Inductive Families  │ │ • Dimensional Units   │ │ • Z3 / CVC5 SMT       │
│ • Positivity Checker  │ │ • Cranelift JIT / VM  │ │ • Proof Certificates  │
└───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
Phase 4: Visuals & Typst  Phase 5: IDLE & Tooling   Phase 6: Advanced Engine
(Months 10-12)           (Months 13-15)            (Months 16+)
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│ • Cassowary Layout    │ │ • Reedline IDLE Shell │ │ • Counterexample Gen  │
│ • Tiny-Skia 2D Render │ │ • Tower-LSP Extension │ │ • AI Tactic Synthesis │
│ • Plotters 2D/3D Plot │ │ • WASM Web Notebook   │ │ • Verified C/CUDA Gen │
│ • Typst Doc Exporter  │ │ • SolidJS Frontend    │ │ • ProofWidgets 3D UI  │
└───────────────────────┘ └───────────────────────┘ └───────────────────────┘
```

---

## 10. Soundness Guarantees & Risk Management

1. **Kernel Isolation:** The proof kernel (`theoria_kernel`) must remain strictly isolated with zero dependencies on external solvers (Z3, Symbolica, or BLAS). Solvers act only as untrusted oracles that generate certificates; the kernel independently verifies every certificate.
2. **Termination & Total Functional Programming:** All functions used within proofs must pass structural recursion or measure-based termination checks to prevent infinite loops from introducing logical inconsistencies ($False$).
3. **Floating-Point Isolation:** Exact mathematical proofs cannot rely on IEEE-754 floating-point operations. The logic kernel works exclusively with exact rationals ($\mathbb{Q}$), symbolic reals ($\mathbb{R}$), and interval arithmetic. Floating-point numbers are strictly isolated to numerical approximations and visual plotting routines.