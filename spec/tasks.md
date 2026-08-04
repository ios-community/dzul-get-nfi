# Task Breakdown & Traceability: GET-NFI

**Role:** Architect Oversight → Senior Engineer Execution  
**Methodology:** Spec-Driven Development (SDD) with Living Doc Verification

## Phase Traceability Matrix

| ID | Task | Phase | Acceptance Criteria | Spec Anchors | Status | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| **T-01** | Project Init & Config | Setup | `Cargo.toml` compiles under `std` and `no_std`. `#![deny(unsafe_code)]` active. | NFR-03, NFR-04 | ✅ | Senior Eng |
| **T-02** | Fixed-Point Weight | Core | `Weight` struct implemented with $10^6$ scaling and `libm` integration. | FR-01, FR-04 | ✅ | Senior Eng |
| **T-03** | CSR Graph Layout | Core | `Graph` struct with CSR validation invariants. | FR-02 | ✅ | Senior Eng |
| **T-04** | Workspace Layout | Core | `Workspace` struct with pre-allocated slices for traversal and A*. | FR-07 | ✅ | Senior Eng |
| **T-05** | Heuristic Trait | Core | `Heuristic` trait, `EuclideanHeuristic`, and `ZeroHeuristic` implemented. | FR-09 | ✅ | Senior Eng |
| **T-06** | Preprocessing Engine | Core | Static Bypass, Geometric Threshold, NFI, and Edge Sorting implemented. | FR-03, FR-05, FR-06 | ✅ | Senior Eng |
| **T-07** | Traversal & Backtracking | Core | Core GET-NFI traversal loop with index-based backtracking. | FR-08 | ✅ | Senior Eng |
| **T-08** | Zero-Allocation A* | Core | A* pathfinding using only pre-allocated `Workspace` buffers. | FR-10 | ✅ | Senior Eng |
| **T-09** | Parallel Multi-Start | Integration | `solve_parallel` implemented using `rayon` (gated under `std`). | FR-11 | ✅ | Senior Eng |
| **T-10** | Living Doc Test Suite | Validation | Implement `tests/living_doc.rs` to automate anchor verification. | NFR-04 | ✅ | Senior Eng |
| **T-11** | Zero-Allocation Test | Validation | Integration test with custom allocator to verify zero heap allocations. | NFR-01 | ✅ | Senior Eng |
| **T-12** | Miri & Portability Gates | Compliance | `cargo miri test` passes. Bare-metal compilation succeeds. | NFR-02, NFR-03 | ✅ | Senior Eng |
| **T-13** | Benchmarking & Docs | Compliance | `criterion` benchmarks executed and baseline saved. | NFR-04, NFR-05 | ✅ | Senior Eng |
| **T-14** | In-Place 2-Opt Search | Core | Implement zero-allocation, in-place 2-Opt local search with $O(1)$ delta evaluation. | FR-12 | ✅ | Senior Eng |
| **T-15** | Dynamic Threshold Tuning | Core | Add threshold multiplier support to `TspConfig` and `solve`. | FR-13 | ✅ | Senior Eng |
| **T-16** | TSPLIB Dataset Loader | Core | Implement a deterministic dataset loader for standard TSPLIB instances with deterministic padding. | FR-14 | ✅ | Senior Eng |

## Validation Sequence (Per Phase)

1. **Living Documentation Verification:**
   ```bash
   # Run the living documentation test first to ensure no anchor drift
   cargo test --test living_doc
   ```
2. **Compilation Check:**
   ```bash
   # Verify bare-metal no_std compilation
   cargo build --target thumbv7m-none-eabi --no-default-features
   # Verify standard compilation
   cargo check --all-features
   ```
3. **Unit & Integration Testing:**
   ```bash
   cargo test --all-features
   ```
4. **Memory Safety & Undefined Behavior Verification:**
   ```bash
   cargo miri test
   ```
5. **Lint & Style Compliance:**
   ```bash
   cargo clippy --all-features -- -D warnings
   cargo fmt --check
   ```
6. **Documentation Verification:**
   ```bash
   RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
   ```
7. **Performance Benchmarking:**
   ```bash
   cargo bench -- --save-baseline stable
   ```

## Performance & Regression Guardrails

- **Zero-Allocation Enforcement:** The test `test_nfr_01_zero_alloc` must wrap the solver execution in a custom allocator that tracks active allocations. If the allocation count is greater than zero during `solve`, the test must fail.
- **Living Doc Enforcement:** The test `test_living_doc` must parse `requirements.md` and scan the codebase. If any `FR-XX` or `NFR-XX` anchor is missing from the code comments or test names, the build must fail.
- **Regression Threshold:** Any pull request that increases the average execution time of the `criterion` benchmark group by more than 5% compared to the `stable` baseline must be rejected and profiled.




