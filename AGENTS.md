# AGENTS.md — AI Developer Guide for `dzul-get-nfi`

High-performance TSP solver: **Dzul's GET-NFI** constructive heuristic + candidate-set 2-Opt local search. Rust workspace + Python benchmarking utilities.

## Crate architecture

| Crate | Type | Entry | Key constraints |
|---|---|---|---|
| `crates/dzul-core` | library (`no_std` by default) | `crates/dzul-core/src/lib.rs` — `solve()`, `solve_parallel()`, `solve_readonly()` | `#![deny(unsafe_code)]`, zero heap allocs during search, MSRV 1.96.0, edition 2024 |
| `crates/dzul-bench` | CLI binary + Divan bench | `crates/dzul-bench/src/main.rs` (`dzul-get-nfi` binary), `benches/solver_benches.rs` | Depends on `dzul-core` with `parallel` feature |

- Features on `dzul-core`: `default = []`, `std`, `parallel` (= `std` + `rayon`).
- Tests live in `src/` (unit tests) and `crates/dzul-bench/tests/` (integration: `alloc_tests`, `living_doc`, `determinism_tests`, `statistics`, `parallel_tests`).

## Verification pipeline (run in order)

```bash
# 1. Format
cargo fmt --all -- --check

# 2. Clippy (pedantic, zero warnings)
cargo clippy --all-targets --all-features -- -D clippy::pedantic

# 3. Tests (both feature sets)
cargo test --all-features
cargo test --no-default-features

# 4. Specialized tests
cargo test --test alloc_tests         # zero-allocation verification
cargo test --test living_doc          # requirements anchor drift check
cargo test --test determinism_tests   # miri UB, portability, determinism
cargo test --test statistics -- --nocapture  # optimality gaps, Wilcoxon

# 5. Bare-metal cross-compile
cargo build --target thumbv7m-none-eabi --no-default-features

# 6. Documentation
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

# 7. Microbenchmarks (Divan)
cargo bench --bench solver_benches
cargo bench --bench solver_benches -- ablation  # ablation study subset

# 8. Memory safety (requires `rustup +nightly component add miri`)
cargo miri test
```

## Python scripts

```bash
uv run ruff check scripts/
uv run ruff format --check scripts/
python scripts/dzul_get_nfi_bench.py
```

Python in `scripts/` is managed by `uv`, linted by `ruff` with `select = ["ALL"]` (see `scripts/pyproject.toml` for ignores).

## Key gotchas

- **CI tests against MSRV 1.96.0 AND stable** — do not bump MSRV without explicit approval.
- **Dataset caching**: Large TSPLIB instances (n > 500) are downloaded on first use from `http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/` and cached to `datasets/{name}.tsp`. Offline runs skip them.
- **Bench binary CLI**: `cargo run --release -- --instance eil51 --sparsity 1.0 --2opt --backtracks 5000 [--directed]`. Flags: `--instance`, `--sparsity`, `--2opt`, `--backtracks`, `--directed`.
- **Logarithmic builds**: `.cargo/config.toml` forces `rust-lld` linker on all x86_64 targets. Do not remove.
- **Stack buffers**: Workspace uses `PATH_STACK_MULTIPLIER = 4` for path_stack sizing. All solver buffers are caller-provided slices (zero allocs).
- **Commit style**: conventional commits (`feat:`, `fix:`, `perf:`, `refactor:`, etc.).
