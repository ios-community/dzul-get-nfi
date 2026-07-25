//! Dzul's GET-NFI TSP Solver.
//!
//! This crate provides a zero-allocation, `no_std` compatible implementation
//! of Dzul's GET-NFI heuristic algorithm for solving the Travelling Salesperson Problem.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::private_intra_doc_links)]
#![deny(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::invalid_codeblock_attributes)]
#![deny(rustdoc::invalid_html_tags)]
#![deny(rustdoc::invalid_rust_codeblocks)]
#![deny(rustdoc::bare_urls)]
#![deny(unsafe_code)]

pub mod error;
pub mod graph;
pub mod heuristic;
pub mod solver;
pub mod weight;
pub mod workspace;

pub use error::TspError;
pub use graph::{Edge, Graph, Node};
pub use heuristic::{EuclideanHeuristic, Heuristic, ZeroHeuristic};
pub use solver::{
    TourType, TspConfig, TspResult, calculate_dynamic_backtrack_limit, calculate_nfi,
    calculate_path_cost, calculate_threshold, solve, solve_readonly, static_bypass, two_opt,
};
pub use weight::Weight;
pub use workspace::{PATH_STACK_MULTIPLIER, Workspace};

#[cfg(feature = "parallel")]
pub use solver::{ParallelTspResult, solve_parallel};
