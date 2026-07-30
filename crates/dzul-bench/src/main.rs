//! CLI Benchmark Runner for Dzul's GET-NFI TSP Solver.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use dzul_bench::{build_complete_graph, build_incomplete_graph, get_dataset};
use dzul_core::{Graph, TspConfig, Weight, Workspace, ZeroHeuristic, solve};
use std::time::Instant;

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut instance_name = "eil51";
    let mut sparsity: f64 = 1.0;
    let mut enable_2opt = false;
    let mut max_backtracks = 5000;
    let mut is_directed = false;
    let mut threshold_multiplier: f64 = 1.0;
    let mut has_custom_multiplier = false;
    let mut backtrack_factor: usize = 10;
    let mut has_custom_factor = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--instance" => {
                instance_name = &args[i + 1];
                i += 2;
            }
            "--sparsity" => {
                sparsity = args[i + 1].parse()?;
                i += 2;
            }
            "--2opt" => {
                enable_2opt = true;
                i += 1;
            }
            "--backtracks" => {
                max_backtracks = args[i + 1].parse()?;
                i += 2;
            }
            "--directed" => {
                is_directed = true;
                i += 1;
            }
            "--threshold-multiplier" => {
                threshold_multiplier = args[i + 1].parse()?;
                has_custom_multiplier = true;
                i += 2;
            }
            "--backtrack-factor" => {
                backtrack_factor = args[i + 1].parse()?;
                has_custom_factor = true;
                i += 2;
            }
            _ => i += 1,
        }
    }

    let coords =
        get_dataset(instance_name).ok_or_else(|| format!("Unknown instance: {instance_name}"))?;
    let n = coords.len();

    let (nodes, edges) = if (sparsity - 1.0).abs() < 1e-5 {
        build_complete_graph(&coords, is_directed)
    } else {
        build_incomplete_graph(&coords, sparsity, is_directed)
    };

    let mut edges = edges;
    let mut graph = Graph {
        nodes: &nodes,
        edges: &mut edges,
        is_directed,
    };

    let mut path_stack = vec![0u32; n * 4];
    let mut next_edge_idx = vec![0u32; n];
    let mut visited = vec![false; n];
    let mut a_star_parent = vec![0u32; n];
    let mut g_score = vec![0u64; n];
    let mut open_set = vec![false; n];
    let mut nfi_buffer = vec![Weight(0); n];
    let mut a_star_heap = vec![0u32; n];
    let mut a_star_heap_pos = vec![-1i32; n];
    let mut f_score = vec![0u64; n];
    let mut dlb = vec![false; n];

    let mut workspace = Workspace {
        path_stack: &mut path_stack,
        next_edge_idx: &mut next_edge_idx,
        visited: &mut visited,
        a_star_parent: &mut a_star_parent,
        g_score: &mut g_score,
        open_set: &mut open_set,
        nfi_buffer: &mut nfi_buffer,
        a_star_heap: &mut a_star_heap,
        a_star_heap_pos: &mut a_star_heap_pos,
        f_score: &mut f_score,
        dlb: &mut dlb,
    };

    let config = TspConfig {
        start_node: 0,
        max_backtracks: if has_custom_factor {
            None
        } else {
            Some(max_backtracks)
        },
        enable_2opt,
        threshold_multiplier: if has_custom_multiplier {
            Some(threshold_multiplier)
        } else {
            None
        },
        backtrack_factor: if has_custom_factor {
            backtrack_factor
        } else {
            10
        },
        candidate_set_size: 15,
    };

    let start_time = Instant::now();
    let result = solve(&mut graph, &mut workspace, &ZeroHeuristic, &config)?;
    let elapsed = start_time.elapsed();

    println!("ELAPSED_MS: {:.6}", elapsed.as_secs_f64() * 1000.0);
    println!("COST: {:.6}", result.total_cost.to_float());
    println!("TOUR_TYPE: {:?}", result.tour_type);

    Ok(())
}
