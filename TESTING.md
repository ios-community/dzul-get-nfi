# Testing & CI/CD Guide

This document explains how to run the test suite, generate coverage reports, execute performance benchmarks, and understand the automated CI/CD pipeline for the `dzul-get-nfi` project.

## Prerequisites

Ensure you have the Rust toolchain installed. To run memory safety checks, you will also need `miri`:

```bash
rustup +nightly component add miri
```

---

## Running Tests Locally

### 1. Unit and Integration Tests
To run all unit tests (located at the bottom of each module file) and integration tests (located in `tests/`):

```bash
cargo test --all-features
```

### 2. Documentation Tests
To verify that all code examples in the documentation and doc comments compile and run correctly:

```bash
cargo test --doc
```

### 3. Testing Specific Features
To run tests without the optional `std` feature (verifying `no_std` compatibility):

```bash
cargo test --no-default-features
```

---

## Specialized Verification

### 1. Zero-Allocation Verification
We enforce a strict zero-allocation policy during solver execution. To verify that exactly zero bytes are allocated on the heap during search:

```bash
cargo test --test alloc_tests
```

This test wraps the solver execution in a custom tracking allocator and asserts that the allocation count remains unchanged.

### 2. Living Documentation Verification
To prevent documentation drift, we use a structural test that parses `requirements.md` and scans the codebase to verify that all requirements have corresponding code anchors and tests:

```bash
cargo test --test living_doc
```

### 3. Memory Safety (Miri)
To verify the complete absence of Undefined Behavior (UB) and memory leaks:

```bash
cargo miri test
```

---

## Benchmarking

We use `criterion` to measure the latency of critical operations (such as the GET-NFI traversal and 2-Opt local search).

To run the benchmarks:

```bash
cargo bench
```

This will execute the benchmarks defined in `benches/solver_benches.rs` and generate detailed HTML reports under `target/criterion/report/index.html`.



