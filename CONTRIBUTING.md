# Contributing to Dzul's GET-NFI

We welcome contributions from the scientific and open-source communities! To maintain high code quality and correctness, please follow these guidelines.

## How to Contribute

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Ensure all tests pass, including the specialized verification tests:
   ```bash
   cargo test --all-features
   cargo test --test alloc_tests
   cargo test --test living_doc
   ```
4. Submit a Pull Request with a detailed description of your changes.

## Code Style & Lints

We enforce strict compiler lints and formatting rules. Before submitting your PR, please run:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

## Documentation

All public APIs must be fully documented. To verify that the documentation builds without warnings:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```
