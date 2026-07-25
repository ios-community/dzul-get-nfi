# Architecture & Design Specification: Dzul's GET-NFI

**Role:** Architect Directive → Senior Engineer Blueprint
**Revision:** 1.3.0 | **Toolchain:** Rust 1.96.0+ (Edition 2024)
**Living Doc Anchor:** This document maps architectural invariants directly to `requirements.md` anchors.

## 1. Module Layout & Spec Anchors

### `src/weight.rs` (Anchor: FR-01, FR-16)
Defines the fixed-point representation and TSPLIB `EUC_2D` distance mode.

### `src/graph.rs` (Anchor: FR-02, FR-06)
Defines the CSR graph structure.

### `src/workspace.rs` (Anchor: FR-07)
Defines the pre-allocated execution state.

### `src/heuristic.rs` (Anchor: FR-09)
Defines the A* heuristic interface.

### `src/datasets.rs` (Anchor: FR-14)
Defines the TSPLIB dataset loader with deterministic padding.

### `src/solver.rs` (Anchor: FR-03, FR-04, FR-05, FR-08, FR-10, FR-11, FR-12, FR-13, FR-15)
Contains the core execution logic, including the in-place 2-Opt local search with $O(1)$ delta evaluation, $O(E \log V)$ Indexed Binary Heap for A* fallback, and the dynamic quadratic backtrack limit $M(N,d) = c \cdot N \cdot d$.

### `src/bench/lib.rs` (Anchor: FR-17)
Contains the expanded baseline heuristic suite (Farthest Insertion, Clarke-Wright Savings, Random Tour) with tour integrity assertions.

## 2. API Surface Contract & Assertions

```rust
// Anchor: FR-07, FR-10
pub struct TspResult<'a> {
    pub path: &'a [u32],
    pub total_cost: Weight,
    pub tour_type: TourType,
    pub is_complete_graph: bool,
}

// Anchor: FR-11
#[cfg(feature = "std")]
pub struct ParallelTspResult {
    pub path: Vec<u32>,
    pub total_cost: Weight,
    pub tour_type: TourType,
    pub is_complete_graph: bool,
}

// Anchor: FR-03, FR-04, FR-05, FR-08, FR-10, FR-12, FR-13
pub fn solve<'a, 'b, 'g, H: Heuristic>(
    graph: &mut Graph<'g>,
    workspace: &'b mut Workspace<'a>,
    heuristic: &H,
    config: &TspConfig,
) -> Result<TspResult<'b>, TspError>;

// Anchor: FR-12
pub fn two_opt(
    graph: &Graph<'_>,
    path: &mut [u32],
    dlb: &mut [bool],
    path_pos: &mut [i32],
) -> Result<Weight, TspError>;

pub fn calculate_path_cost(
    graph: &Graph<'_>,
    path: &[u32],
) -> Result<Weight, TspError>;

// Anchor: FR-14
pub fn get_dataset(name: &str) -> Option<Vec<(f64, f64)>>;
```

## 3. Living Documentation Verification Code

To ensure this design document does not drift from the implementation, the following assertion pattern must be used in the codebase:

```rust
// In src/solver.rs
#[doc = "Anchor: FR-03 (Static Bypass)"]
fn static_bypass(graph: &Graph<'_>) -> bool {
    // Implementation...
}
```
The automated living documentation test (`tests/living_doc.rs`) will scan for these exact doc attributes and comments.
```
