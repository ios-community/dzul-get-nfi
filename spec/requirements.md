# Requirements Specification: GET-NFI TSP Solver

**Role:** Architect Engineer → Senior Software Engineer
**Status:** Frozen | **MSRV:** Rust 1.96.0 | **Edition:** 2024
**Living Doc Protocol:** All `FR-XX` and `NFR-XX` IDs must exist as code comments (`// Anchor: ID`) and test names (`test_id_...`) in the codebase.

## Architect Directives

- **Primary Objective:** Implement GET-NFI heuristic algorithm to solve the TSP on complete/incomplete and directed/undirected graphs with zero heap allocations during execution and full `no_std` compatibility.
- **Crate Type:** Library (`lib`) with optional `std` support.
- **Memory Safety:** `#![deny(unsafe_code)]` enforced globally. No `unsafe` blocks permitted.

## Functional Requirements (FR) & Verification Anchors

| ID | Requirement | Owner | Description | Verification Anchor |
| --- | --- | --- | --- | --- |
| FR-01 | **Fixed-Point Weight** | Senior SE | Implement a fixed-point weight representation scaled by $10^6$ to avoid floating-point non-determinism. | `crate::weight::Weight`<br>`crate::weight::tests::test_fr_01_fixed_point` |
| FR-02 | **CSR Graph Layout** | Senior SE | Implement a Compressed Sparse Row (CSR) graph layout to eliminate pointer chasing and heap allocations. | `crate::graph::Graph`<br>`crate::graph::tests::test_fr_02_csr_layout` |
| FR-03 | **Static Bypass** | Senior SE | Implement a static bypass check to skip thresholding and NFI calculation if all edge weights are uniform. | `crate::solver::static_bypass`<br>`crate::solver::tests::test_fr_03_bypass` |
| FR-04 | **Geometric Threshold** | Senior SE | Calculate the geometric threshold theta to partition edges into low-cost and high-cost sets. | `crate::solver::calculate_threshold`<br>`crate::solver::tests::test_fr_04_threshold` |
| FR-05 | **Second-Order NFI** | Senior SE | Pre-calculate the Second-Order Node Friendliness Index (NFI) to prioritize isolated nodes. | `crate::solver::calculate_nfi`<br>`crate::solver::tests::test_fr_05_nfi` |
| FR-06 | **Edge Sorting** | Senior SE | Sort outgoing edges of each node according to the GET-NFI rules (low-cost first, then high-cost by target NFI). | `crate::graph::Graph::sort_edges`<br>`crate::graph::tests::test_fr_06_sorting` |
| FR-07 | **Workspace Validation** | Senior SE | Validate that the pre-allocated workspace buffers are large enough for the given node count. | `crate::workspace::Workspace`<br>`crate::workspace::tests::test_fr_07_workspace` |
| FR-08 | **Backtracking Traversal** | Senior SE | Implement the core GET-NFI traversal loop with index-based backtracking. | `crate::solver::solve`<br>`crate::solver::tests::test_fr_08_backtrack` |
| FR-09 | **Heuristic Interface** | Senior SE | Define the `Heuristic` trait and implement `ZeroHeuristic` and `EuclideanHeuristic`. | `crate::heuristic::Heuristic`<br>`crate::heuristic::tests::test_fr_09_heuristic` |
| FR-10 | **Tour Closure** | Senior SE | Close the TSP tour by returning to the starting node using direct edge or A* fallback. | `crate::solver::close_tour`<br>`crate::solver::tests::test_fr_10_closure` |
| FR-11 | **Multi-Start Parallelism** | Senior SE | Run the solver from every node in parallel using Rayon (gated under the `std` feature). | `crate::solver::solve_parallel`<br>`crate::parallel_tests::test_fr_11_parallel` |
| FR-12 | **In-Place 2-Opt Search** | Senior SE | Perform an in-place, zero-allocation 2-Opt local search with $O(1)$ delta evaluation. | `crate::solver::two_opt`<br>`crate::solver::tests::test_fr_12_2opt` |
| FR-13 | **Dynamic Threshold Tuning** | Senior SE | Allow tuning the geometric threshold theta dynamically via a multiplier in `TspConfig`. | `crate::solver::solve`<br>`crate::solver::tests::test_fr_13_threshold_multiplier` |
| FR-14 | **TSPLIB Dataset Loader** | Senior SE | Provide a deterministic dataset loader for standard TSPLIB instances with deterministic padding for truncated instances. | `crate::datasets::get_dataset`<br>`crate::datasets::tests::test_fr_14_dataset_loader` |
| FR-15 | **Dynamic Quadratic Backtrack Limit** | Senior SE | Compute the DFS backtrack limit dynamically via `M(N,d) = c·N·d`, bounding worst-case execution to O(N³). | `crate::solver::calculate_dynamic_backtrack_limit`<br>`crate::solver::tests::test_fr_15_dynamic_backtrack_limit` |
| FR-16 | **TSPLIB EUC_2D Distance Mode** | Senior SE | Compute edge weights using TSPLIB `EUC_2D` nearest-integer rounding (`nint(sqrt(dx²+dy²))`) to match Concorde benchmark optimal values. | `crate::weight::Weight::euc_2d`<br>`crate::weight::tests::test_fr_16_euc_2d` |
| FR-17 | **Expanded Baseline Suite** | Senior SE | Provide Farthest Insertion, Clarke-Wright Savings, and Random Tour baselines with tour integrity assertions (N+1 elements, all nodes covered). | `crate::bench::solve_farthest_insertion`<br>`crate::bench::tests::test_fr_17_baseline_suite` |

## Non-Functional Requirements (NFR) & Verification Anchors

| ID | Category | Constraint | Verification Anchor (Test Path) |
| --- | --- | --- | --- |
| **NFR-01** | Memory | Exactly `0` bytes allocated on the heap during `solve` execution. | `tests::alloc_tests::test_nfr_01_zero_alloc` |
| **NFR-02** | Safety | Complete absence of Undefined Behavior (UB). | `tests::determinism_tests::test_nfr_02_miri` |
| **NFR-03** | Portability | Strict `no_std` compatibility (compiles on bare-metal). | `tests::determinism_tests::test_nfr_03_portability` |
| **NFR-04** | Documentation | 100% documentation coverage, zero warnings, all doctests pass. | `tests::determinism_tests::test_nfr_04_documentation` |
| **NFR-05** | Determinism | Identical outputs (path and cost) for identical inputs across platforms. | `tests::determinism_tests::test_nfr_05_determinism` |

## Living Documentation Automation Protocol

To prevent documentation drift, the test suite contains a structural test (`tests/living_doc.rs`) that:
1. Parses `requirements.md` to extract all active `FR-XX` and `NFR-XX` IDs.
2. Scans the `src/` and `tests/` directories to verify that:
   - A code comment containing `// Anchor: ID` exists.
   - A test function containing `test_id_` or matching the verification anchor path exists.
3. Fails the build if any requirement is missing its corresponding code anchor or test.




