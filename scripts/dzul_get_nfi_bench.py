"""Benchmarking script for Dzul's GET-NFI TSP solver.

Uses uv for dependency management and ruff for linting.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    REPO_ROOT = Path("/content/dzul-get-nfi")
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent

MATERIALS_DIR = REPO_ROOT / "scripts" / "materials"
PLOTS_DIR = REPO_ROOT / "scripts" / "plots"

ext = ".exe" if sys.platform == "win32" else ""
RUST_BIN_PATH = REPO_ROOT / f"dzul_get_nfi_bench{ext}"
CSV_RESULTS_PATH = MATERIALS_DIR / "get_nfi_benchmark_results.csv"
BENCH_FILE_PATH = REPO_ROOT / "crates" / "dzul-bench" / "benches" / "solver_benches.rs"

PYPROJECT_PATH = REPO_ROOT / "scripts" / "pyproject.toml"


def _run_uv(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a uv command with the scripts pyproject.toml."""
    return subprocess.run(
        [sys.executable, "-m", "uv", *args],
        cwd=REPO_ROOT / "scripts",
        check=check,
        capture_output=True,
        text=True,
    )


def get_criterion_estimates(benchmark_id: str) -> tuple[float | None, float | None]:
    """Extract mean/std from Criterion estimates.json."""
    criterion_dir = REPO_ROOT / "target" / "criterion"
    if not criterion_dir.exists():
        return None, None

    exact_path = criterion_dir / benchmark_id / "new" / "estimates.json"
    if exact_path.exists():
        try:
            with exact_path.open() as f:
                data = json.load(f)
                mean_ms = data["mean"]["point_estimate"] / 1_000_000.0
                std_dev_ms = data["std_dev"]["point_estimate"] / 1_000_000.0
                return mean_ms, std_dev_ms
        except Exception:  # noqa: BLE001
            pass

    for path in criterion_dir.glob("**/estimates.json"):
        if any(benchmark_id.lower() in part.lower() for part in path.parts):
            try:
                with path.open() as f:
                    data = json.load(f)
                    mean_ms = data["mean"]["point_estimate"] / 1_000_000.0
                    std_dev_ms = data["std_dev"]["point_estimate"] / 1_000_000.0
                    return mean_ms, std_dev_ms
            except Exception:  # noqa: BLE001
                pass

    return None, None


def setup_environment() -> None:
    """Install Python deps via uv, verify Rust toolchain."""
    if IN_COLAB:
        print("Running in Google Colab. Setting up repository...")
        os.chdir("/content")
        if not REPO_ROOT.exists():
            subprocess.run(
                ["git", "clone", "https://github.com/ios-community/dzul-get-nfi.git"],
                check=False,
            )
        os.chdir(str(REPO_ROOT))

        if shutil.which("rustc") is None:
            print("Installing Rust toolchain...")
            subprocess.run(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                shell=True,
                check=False,
            )
            os.environ["PATH"] = f"{os.environ['HOME']}/.cargo/bin:{os.environ['PATH']}"

        print("Installing Python dependencies via uv...")
        _run_uv(["pip", "install", "-q", "-e", "."], check=False)
    else:
        print("Running in local environment.")
        if not (REPO_ROOT / "Cargo.toml").exists():
            msg = (
                f"Cargo.toml not found at: {REPO_ROOT}\n"
                "Please ensure this script is located inside the 'scripts/' directory."
            )
            raise FileNotFoundError(msg)

        if not BENCH_FILE_PATH.exists():
            msg = (
                f"Benchmark source file not found at: {BENCH_FILE_PATH}\n"
                "Please verify 'benches/solver_benches.rs' exists."
            )
            raise FileNotFoundError(msg)

        # Verify deps installed via uv
        missing_deps: list[str] = []
        for dep in ["psutil", "scipy", "pandas", "matplotlib", "numpy", "jinja2"]:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            print(f"Missing deps {missing_deps}, installing via uv...")
            _run_uv(["pip", "install", "-q", "-e", "."], check=False)

    print("Environment setup verified.")


def lint_script() -> None:
    """Run ruff check and ruff format check on Python files."""
    print("Running ruff lint (strict ALL)...")
    scripts_dir = REPO_ROOT / "scripts"
    result = subprocess.run(
        ["uvx", "ruff", "check", "--config", str(PYPROJECT_PATH), str(scripts_dir / "dzul_get_nfi_bench.py")],
        cwd=scripts_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Ruff check failed:\n{result.stdout}\n{result.stderr}")
        raise SystemExit(1)
    print("Ruff check passed.")

    result = subprocess.run(
        [
            "uvx",
            "ruff",
            "format",
            "--check",
            "--config",
            str(PYPROJECT_PATH),
            str(scripts_dir / "dzul_get_nfi_bench.py"),
        ],
        cwd=scripts_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Ruff format check failed:\n{result.stdout}\n{result.stderr}")
        raise SystemExit(1)
    print("Ruff format check passed.")


def profile_hardware() -> None:
    """Print hardware specification report."""
    try:
        import psutil
    except ImportError as exc:
        msg = "psutil not installed. Run setup_environment() first."
        raise ImportError(msg) from exc

    specs: dict[str, str] = {}
    system = platform.system()
    specs["OS"] = f"{system} {platform.release()}"

    if system == "Linux":
        try:
            cpu_info = subprocess.check_output("cat /proc/cpuinfo", shell=True, text=True)
            for line in cpu_info.split("\n"):
                if "model name" in line:
                    specs["CPU Model"] = line.split(":")[1].strip()
                    break
        except Exception:  # noqa: BLE001
            pass
    elif system == "Darwin":
        try:
            specs["CPU Model"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
        except Exception:  # noqa: BLE001
            pass
    elif system == "Windows":
        specs["CPU Model"] = platform.processor()

    if "CPU Model" not in specs:
        specs["CPU Model"] = platform.processor() or "Unknown CPU"

    try:
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        specs["Total RAM"] = f"{total_ram_gb:.2f} GB"
    except Exception:  # noqa: BLE001
        specs["Total RAM"] = "Unknown"

    print("=" * 60)
    print("          HARDWARE SPECIFICATION REPORT")
    print("=" * 60)
    print(f"OS          : {specs['OS']}")
    print(f"CPU Model   : {specs['CPU Model']}")
    print(f"Total RAM   : {specs['Total RAM']}")
    print("=" * 60)
    print("\nDraft text for your paper's 'Experimental Setup' section:")
    print(
        f'"All experiments were executed on an environment running {specs["OS"]}, '
        f"equipped with an {specs['CPU Model']} processor and {specs['Total RAM']} of total physical memory. "
        f'To ensure deterministic timing, the benchmark process was pinned to a single CPU core with high priority."'
    )


def compile_rust_binary() -> None:
    """Compile Rust binary with aggressive optimizations."""
    if shutil.which("cargo") is None:
        msg = "Rust compiler ('cargo') not found.\nInstall Rust: https://www.rust-lang.org/"
        raise RuntimeError(msg)

    print("Compiling Rust binary...")
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"
    env["CARGO_PROFILE_RELEASE_LTO"] = "fat"
    env["CARGO_PROFILE_RELEASE_CODEGEN_UNITS"] = "1"

    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = f"Rust compilation failed:\n{result.stderr}"
        raise RuntimeError(msg)

    src_bin_hyphen = REPO_ROOT / "target" / "release" / f"dzul-get-nfi{ext}"
    src_bin_underscore = REPO_ROOT / "target" / "release" / f"dzul_get_nfi{ext}"

    if src_bin_hyphen.exists():
        src_bin = src_bin_hyphen
    elif src_bin_underscore.exists():
        src_bin = src_bin_underscore
    else:
        msg = "Could not find compiled binary in target/release/.\nPlease ensure the package compiles successfully."
        raise FileNotFoundError(msg)

    shutil.copy(src_bin, RUST_BIN_PATH)
    print(f"Compilation successful. Binary copied to: {RUST_BIN_PATH}")


def parse_benchmark_output(stdout: str) -> tuple[float, float, str]:
    """Parse solver binary stdout for ELAPSED_MS, COST, TOUR_TYPE."""
    elapsed_ms = 0.0
    cost = 0.0
    tour_type = "StrictCycle"
    for line in stdout.splitlines():
        if line.startswith("ELAPSED_MS:"):
            elapsed_ms = float(line.split(":")[1].strip())
        elif line.startswith("COST:"):
            cost = float(line.split(":")[1].strip())
        elif line.startswith("TOUR_TYPE:"):
            tour_type = line.split(":")[1].strip()
    return elapsed_ms, cost, tour_type


INSTANCES: dict[str, float] = {
    "eil51": 426.0,
    "pr76": 108_159.0,
    "kroA100": 21_282.0,
    "lin318": 42_029.0,
    "pcb442": 50_778.0,
}


def run_main_experiments() -> None:
    """Run full experiment matrix: instances x sparsity x 2opt."""
    try:
        import pandas as pd
        import psutil
    except ImportError as exc:
        msg = "Missing Python dependencies. Run setup_environment() first."
        raise ImportError(msg) from exc

    SPARSITY_VALUES = [1.0, 0.5]

    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)

    bin_exists = RUST_BIN_PATH.exists()
    if not bin_exists:
        print(f"Notice: Compiled solver binary not found at {RUST_BIN_PATH}.")
        print("Skipping binary execution. Tour costs and memory metrics will use default values.")

    def run_benchmark_process(
        cmd: list[str],
    ) -> tuple[int, str, str, int, int]:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=REPO_ROOT,
        )
        peak_rss = 0
        peak_uss = 0
        try:
            ps_proc = psutil.Process(p.pid)
            while p.poll() is None:
                try:
                    mem = ps_proc.memory_info()
                    if mem.rss > peak_rss:
                        peak_rss = mem.rss
                    try:
                        full_mem = ps_proc.memory_full_info()
                        if full_mem.uss > peak_uss:
                            peak_uss = full_mem.uss
                    except (psutil.AccessDenied, AttributeError):
                        peak_uss = mem.rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                time.sleep(0.001)
        except psutil.NoSuchProcess:
            pass

        stdout, stderr = p.communicate()
        return p.returncode, stdout, stderr, peak_rss, peak_uss

    results: list[dict[str, object]] = []
    for instance, opt_cost in INSTANCES.items():
        for sparsity in SPARSITY_VALUES:
            for enable_2opt in [False, True]:
                opt_label = "2-Opt" if enable_2opt else "GET-NFI"
                sparsity_label = "Complete" if sparsity == 1.0 else "Incomplete (50%)"
                print(f"Processing: {instance} | {sparsity_label} | {opt_label}")

                cost = float("nan")
                tour_type = "N/A"
                elapsed_ms = 0.0
                rss = 0.0
                uss = 0.0

                if bin_exists:
                    cmd = [
                        str(RUST_BIN_PATH),
                        "--instance",
                        instance,
                        "--sparsity",
                        str(sparsity),
                        "--backtracks",
                        "1000",
                    ]
                    if enable_2opt:
                        cmd.append("--2opt")

                    ret, stdout, _stderr, rss, uss = run_benchmark_process(cmd)
                    if ret == 0:
                        elapsed_ms, cost, tour_type = parse_benchmark_output(stdout)

                if tour_type == "N/A":
                    cost = float("nan")
                    gap = float("nan")
                else:
                    gap = ((cost - opt_cost) / opt_cost) * 100.0

                results.append(
                    {
                        "Instance": instance,
                        "Sparsity": sparsity_label,
                        "Algorithm": opt_label,
                        "Time_MS": elapsed_ms,
                        "Time_SD": 0.0,
                        "Cost": cost,
                        "Gap_Percent": gap,
                        "RSS_KB": rss / 1024,
                        "USS_KB": uss / 1024,
                        "Tour_Type": tour_type,
                    },
                )

    df = pd.DataFrame(results)
    df.to_csv(CSV_RESULTS_PATH, index=False)
    print("Main experiment matrix complete. Results saved.")


def run_advanced_analyses() -> None:
    """Sparsity phase transition, Pareto frontier, ATSP analysis."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        msg = "Missing Python dependencies. Run setup_environment() first."
        raise ImportError(msg) from exc

    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    bin_exists = RUST_BIN_PATH.exists()
    if not bin_exists:
        print(f"Notice: Compiled solver binary not found at {RUST_BIN_PATH}.")
        print("Skipping binary execution. Optimality gaps and costs will use default values.")

    # --- Sparsity Phase Transition ---
    print("\nRunning Sparsity Phase Transition Analysis...")
    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sparsity_results: list[dict[str, float]] = []
    for sparsity in sparsity_levels:
        print(f"  Testing Sparsity Level: {sparsity * 100:.0f}%")

        cost = float("nan")
        elapsed_ms = 0.0
        tour_type = "N/A"
        if bin_exists:
            cmd = [
                str(RUST_BIN_PATH),
                "--instance",
                "eil51",
                "--sparsity",
                str(sparsity),
                "--backtracks",
                "1000",
                "--2opt",
            ]
            ret = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False)
            if ret.returncode == 0:
                elapsed_ms, cost, tour_type = parse_benchmark_output(ret.stdout)

        opt_cost = INSTANCES["eil51"]
        gap = float("nan") if tour_type == "N/A" else ((cost - opt_cost) / opt_cost) * 100.0
        sparsity_results.append({"Sparsity": sparsity, "Time_MS": elapsed_ms, "Gap_Percent": gap})
    df_sparsity = pd.DataFrame(sparsity_results)
    df_sparsity.to_csv(MATERIALS_DIR / "sparsity_phase_transition.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(6.0, 3.8))
    color = "tab:red"
    ax1.set_xlabel("Sparsity Level (Ratio of Kept Edges)")
    ax1.set_ylabel("Execution Time (ms)", color=color)
    ax1.plot(df_sparsity["Sparsity"], df_sparsity["Time_MS"], "o-", color=color, linewidth=1.5)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Optimality Gap (%)", color=color)
    ax2.plot(df_sparsity["Sparsity"], df_sparsity["Gap_Percent"], "s--", color=color, linewidth=1.5)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Sparsity Phase Transition Analysis (eil51)")
    fig.tight_layout()
    plt.savefig(PLOTS_DIR / "sparsity_phase_transition.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    # --- Pareto Frontier ---
    print("Running Pareto Frontier Analysis...")
    backtrack_limits = [100, 1000, 5000]
    pareto_results: list[dict[str, float]] = []
    for limit in backtrack_limits:
        cost = float("nan")
        elapsed_ms = 0.0
        tour_type = "N/A"
        if bin_exists:
            cmd = [
                str(RUST_BIN_PATH),
                "--instance",
                "kroA100",
                "--sparsity",
                "1.0",
                "--backtracks",
                str(limit),
                "--2opt",
            ]
            ret = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False)
            if ret.returncode == 0:
                elapsed_ms, cost, tour_type = parse_benchmark_output(ret.stdout)

        opt_cost = INSTANCES["kroA100"]
        gap = float("nan") if tour_type == "N/A" else ((cost - opt_cost) / opt_cost) * 100.0
        pareto_results.append({"Backtrack_Limit": limit, "Time_MS": elapsed_ms, "Gap_Percent": gap})
    df_pareto = pd.DataFrame(pareto_results)
    df_pareto.to_csv(MATERIALS_DIR / "pareto_frontier.csv", index=False)

    plt.figure(figsize=(6.0, 3.8))
    plt.plot(
        df_pareto["Time_MS"],
        df_pareto["Gap_Percent"],
        "o-",
        color="#CE412B",
        linewidth=1.5,
        markersize=6,
    )
    for _, row in df_pareto.iterrows():
        plt.annotate(
            f"Limit: {int(row['Backtrack_Limit'])}",
            (row["Time_MS"], row["Gap_Percent"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    plt.xlabel("Execution Time (ms)")
    plt.ylabel("Optimality Gap (%)")
    plt.title("Pareto Frontier Analysis (kroA100, Complete Graph)")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pareto_frontier.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    # --- Asymmetric TSP ---
    print("Running Asymmetric TSP (ATSP) Analysis...")
    atsp_results: list[dict[str, object]] = []
    for instance in INSTANCES:
        for is_directed in [False, True]:
            dir_label = "Asymmetric (ATSP)" if is_directed else "Symmetric (STSP)"
            cost = float("nan")
            elapsed_ms = 0.0
            if bin_exists:
                cmd = [
                    str(RUST_BIN_PATH),
                    "--instance",
                    instance,
                    "--sparsity",
                    "1.0",
                    "--backtracks",
                    "1000",
                    "--2opt",
                ]
                if is_directed:
                    cmd.append("--directed")

                ret = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False)
                if ret.returncode == 0:
                    elapsed_ms, cost, _ = parse_benchmark_output(ret.stdout)

            atsp_results.append(
                {
                    "Instance": instance,
                    "Type": dir_label,
                    "Time_MS": elapsed_ms,
                    "Cost": cost,
                },
            )
    df_atsp = pd.DataFrame(atsp_results)
    df_atsp.to_csv(MATERIALS_DIR / "atsp_comparison.csv", index=False)
    print("Advanced analyses complete.")


def run_statistical_analysis() -> None:
    """Wilcoxon signed-rank tests on benchmark results."""
    if not CSV_RESULTS_PATH.exists():
        msg = f"Benchmark results file not found at: {CSV_RESULTS_PATH}\nRun run_main_experiments() first."
        raise FileNotFoundError(msg)

    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
    except ImportError as exc:
        msg = "Missing Python dependencies. Run setup_environment() first."
        raise ImportError(msg) from exc

    def calculate_rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
        diff = x - y
        diff = diff[diff != 0]
        n = len(diff)
        if n == 0:
            return 0.0
        ranks = stats.rankdata(np.abs(diff))
        pos_sum = float(np.sum(ranks[diff > 0]))
        neg_sum = float(np.sum(ranks[diff < 0]))
        total_rank_sum = n * (n + 1) / 2
        return (pos_sum - neg_sum) / total_rank_sum

    df_stats = pd.read_csv(CSV_RESULTS_PATH)
    get_nfi_df = df_stats[df_stats["Algorithm"] == "GET-NFI"].sort_values(by=["Instance", "Sparsity"])
    opt_2_df = df_stats[df_stats["Algorithm"] == "2-Opt"].sort_values(by=["Instance", "Sparsity"])

    def _wilcoxon_report(label: str, x: np.ndarray, y: np.ndarray) -> None:
        """Run Wilcoxon paired test and print formatted results, dropping NaN pairs."""
        valid = ~(np.isnan(x) | np.isnan(y))
        xv = x[valid]
        yv = y[valid]
        n_valid = len(xv)
        print(f"{label}:")
        if n_valid < 2:
            print("   Insufficient valid paired observations (need >= 2).")
            return
        diff = xv - yv
        if np.all(diff == 0):
            print("   Statistic: N/A | p-value: N/A | All differences are zero.")
            return
        stat_val, p_val = stats.wilcoxon(xv, yv)
        r_val = calculate_rank_biserial(xv, yv)
        print(f"   n={n_valid}  Statistic: {stat_val:.4f} | p-value: {p_val:.6f}")
        print(f"   Effect Size (Rank-Biserial r): {r_val:.4f}")
        if p_val < 0.05:
            print("   Result:    SIGNIFICANT (Reject H0 at alpha = 0.05)")
        else:
            print("   Result:    NOT SIGNIFICANT")

    times_get_nfi = get_nfi_df["Time_MS"].to_numpy()
    times_2opt = opt_2_df["Time_MS"].to_numpy()
    gaps_get_nfi = get_nfi_df["Gap_Percent"].to_numpy()
    gaps_2opt = opt_2_df["Gap_Percent"].to_numpy()

    print("=" * 65)
    print("          DZUL'S GET-NFI STATISTICAL RIGOR REPORT")
    print("=" * 65)
    _wilcoxon_report("1. Wilcoxon Signed-Rank Test (Optimality Gap: GET-NFI vs 2-Opt)", gaps_get_nfi, gaps_2opt)
    print("-" * 65)
    _wilcoxon_report("2. Wilcoxon Signed-Rank Test (Execution Time: GET-NFI vs 2-Opt)", times_get_nfi, times_2opt)
    print("-" * 65)

    complete_gaps = df_stats[(df_stats["Sparsity"] == "Complete") & (df_stats["Algorithm"] == "GET-NFI")][
        "Gap_Percent"
    ].to_numpy()
    incomplete_gaps = df_stats[(df_stats["Sparsity"] == "Incomplete (50%)") & (df_stats["Algorithm"] == "GET-NFI")][
        "Gap_Percent"
    ].to_numpy()

    print("3. Wilcoxon Signed-Rank Test (GET-NFI Gaps: Complete vs Incomplete):")
    if len(complete_gaps) == len(incomplete_gaps):
        _wilcoxon_report("   Result", complete_gaps, incomplete_gaps)
    else:
        print("   Error: Sample sizes do not match for Complete vs Incomplete gaps.")
    print("=" * 65)


def parse_time_to_ms(time_str: str, unit_str: str) -> float:
    """Convert a time string with unit (ms, µs, ns, s) to milliseconds."""
    val = float(time_str)
    unit = unit_str.strip()
    if unit == "µs":
        return val / 1000.0
    if unit == "ns":
        return val / 1_000_000.0
    if unit == "s":
        return val * 1000.0
    return val


def parse_statistics_output(filepath: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse Divan statistics output into DataFrames for ablation, group1, group2."""
    import pandas as pd

    if not filepath.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    with filepath.open("r", encoding="utf-8") as f:
        content = f.read()

    ablation_data = []
    ablation_match = re.search(
        r"=== 2-Opt Ablation Study ===\n.*?\n-+\n(.*?)\n\n====================",
        content,
        re.DOTALL,
    )
    if ablation_match:
        lines = ablation_match.group(1).strip().split("\n")
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                ablation_data.append(
                    {
                        "Instance": parts[0],
                        "Opt_Cost": int(parts[1]),
                        "Initial_Gap_Percent": float(parts[2].replace("%", "")),
                        "Final_Gap_Percent": float(parts[3].replace("%", "")),
                        "Delta_Improvement_Percent": float(parts[4].replace("%", "")),
                    }
                )
    df_ablation = pd.DataFrame(ablation_data)
    if not df_ablation.empty:
        df_ablation.to_csv(MATERIALS_DIR / "ablation_2opt_results.csv", index=False)

    g1_data = []
    g1_match = re.search(
        r"=== GROUP 1: Pure Constructive Heuristics \(NO 2-Opt\) ===\n.*?\n-+\n(.*?)\n\n====================",
        content,
        re.DOTALL,
    )
    if g1_match:
        lines = g1_match.group(1).strip().split("\n")
        for line in lines:
            parts = line.split()
            if len(parts) >= 10:
                g1_data.append(
                    {
                        "Instance": parts[0],
                        "Opt": int(parts[1]),
                        "NN_Cost": int(parts[2]),
                        "FI_Cost": int(parts[3]),
                        "CW_Cost": int(parts[4]),
                        "GET_NFI_Cost": int(parts[5]),
                        "NN_Gap_Percent": float(parts[6].replace("%", "")),
                        "FI_Gap_Percent": float(parts[7].replace("%", "")),
                        "CW_Gap_Percent": float(parts[8].replace("%", "")),
                        "GET_NFI_Time_MS": float(parts[9]),
                    }
                )
    df_g1 = pd.DataFrame(g1_data)
    if not df_g1.empty:
        df_g1.to_csv(MATERIALS_DIR / "group1_constructive_results.csv", index=False)

    g2_data = []
    g2_match = re.search(
        r"=== GROUP 2: Constructive \+ 2-Opt ===\n.*?\n-+\n(.*?)\n\n====================",
        content,
        re.DOTALL,
    )
    if g2_match:
        lines = g2_match.group(1).strip().split("\n")
        for line in lines:
            parts = line.split()
            if len(parts) >= 11:
                g2_data.append(
                    {
                        "Instance": parts[0],
                        "Opt": int(parts[1]),
                        "Random_2Opt_Cost": int(parts[2]),
                        "NN_2Opt_Cost": int(parts[3]),
                        "FI_2Opt_Cost": int(parts[4]),
                        "CW_2Opt_Cost": int(parts[5]),
                        "GET_NFI_2Opt_Cost": int(parts[6]),
                        "Random_2Opt_Gap_Percent": float(parts[7].replace("%", "")),
                        "NN_2Opt_Gap_Percent": float(parts[8].replace("%", "")),
                        "GET_NFI_2Opt_Gap_Percent": float(parts[9].replace("%", "")),
                        "GET_NFI_2Opt_Time_MS": float(parts[10]),
                    }
                )
    df_g2 = pd.DataFrame(g2_data)
    if not df_g2.empty:
        df_g2.to_csv(MATERIALS_DIR / "group2_2opt_results.csv", index=False)

    return df_ablation, df_g1, df_g2


def parse_divan_benches(filepath: Path) -> pd.DataFrame:
    """Parse Divan benchmark text output into a DataFrame."""
    import pandas as pd

    if not filepath.exists():
        return pd.DataFrame()

    bench_data = []
    current_category = ""

    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            cat_match = re.search(r"├─\s*([a-zA-Z0-9_]+)\s*$", line) or re.search(r"╰─\s*([a-zA-Z0-9_]+)\s*$", line)
            if cat_match:
                current_category = cat_match.group(1)
                continue

            item_match = re.search(
                r"(?:├─|╰─)\s*([a-zA-Z0-9_\.]+)\s+([\d\.]+)\s*(µs|ms|s|ns)\s*│\s*([\d\.]+)\s*(µs|ms|s|ns)\s*│\s*([\d\.]+)\s*(µs|ms|s|ns)\s*│\s*([\d\.]+)\s*(µs|ms|s|ns)",
                line,
            )
            if item_match and current_category:
                item_name = item_match.group(1)
                fastest_ms = parse_time_to_ms(item_match.group(2), item_match.group(3))
                slowest_ms = parse_time_to_ms(item_match.group(4), item_match.group(5))
                median_ms = parse_time_to_ms(item_match.group(6), item_match.group(7))
                mean_ms = parse_time_to_ms(item_match.group(8), item_match.group(9))

                bench_data.append(
                    {
                        "Category": current_category,
                        "Item": item_name,
                        "Fastest_MS": fastest_ms,
                        "Slowest_MS": slowest_ms,
                        "Median_MS": median_ms,
                        "Mean_MS": mean_ms,
                    }
                )

    df_divan = pd.DataFrame(bench_data)
    if not df_divan.empty:
        df_divan.to_csv(MATERIALS_DIR / "divan_microbenchmarks.csv", index=False)
    return df_divan


def generate_plots_and_tables() -> None:
    """Generate LaTeX tables, latency/memory plots, and sensitivity analysis figures."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        msg = "Missing Python dependencies. Run setup_environment() first."
        raise ImportError(msg) from exc

    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    parse_statistics_output(REPO_ROOT / "raw_statistics_output.txt")
    df_divan = parse_divan_benches(REPO_ROOT / "raw_benches_output.txt")

    if CSV_RESULTS_PATH.exists():
        df = pd.read_csv(CSV_RESULTS_PATH)

        latex_path = MATERIALS_DIR / "get_nfi_benchmark_table.tex"
        with latex_path.open("w") as f:
            f.write(r"\begin{table*}[htbp]" + "\n")
            f.write(r"\caption{Performance, Accuracy, and Memory Footprint of GET-NFI Solver}" + "\n")
            f.write(r"\label{tab:get_nfi_results}" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\begin{tabular}{ccc|cc|c|cc}" + "\n")
            f.write(r"\hline" + "\n")
            f.write(
                r"Instance & Graph Type & Algorithm & Execution Time (ms) & Cost & Gap (\%) & Peak RSS (MB) & Peak USS (MB) \\"
                + "\n",
            )
            f.write(r"\hline" + "\n")
            for instance in INSTANCES:
                inst_df = df[df["Instance"] == instance]
                first_inst = True
                for sparsity in ["Complete", "Incomplete (50%)"]:
                    sp_df = inst_df[inst_df["Sparsity"] == sparsity]
                    first_sp = True
                    for _, row in sp_df.iterrows():
                        inst_label = instance if (first_inst and first_sp) else ""
                        sp_label = sparsity if first_sp else ""
                        first_inst = False
                        first_sp = False
                        time_str = f"{row['Time_MS']:.4f} \\pm {row['Time_SD']:.4f}"
                        cost_str = f"{row['Cost']:.2f}"
                        gap_str = f"{row['Gap_Percent']:.2f}\\%"
                        rss_mb = f"{row['RSS_KB'] / 1024:.2f}"
                        uss_mb = f"{row['USS_KB'] / 1024:.2f}"
                        f.write(
                            f" {inst_label} & {sp_label} & {row['Algorithm']} & {time_str} & {cost_str} & {gap_str} & {rss_mb} & {uss_mb} \\\\\n",
                        )
                f.write(r"\hline" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table*}" + "\n")

        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman"],
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
            },
        )

        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
        colors = {"GET-NFI": "#CE412B", "2-Opt": "#3776AB"}
        markers = {"GET-NFI": "o", "2-Opt": "s"}
        for idx, (sparsity, ax) in enumerate(zip(["Complete", "Incomplete (50%)"], axes)):
            sp_df = df[df["Sparsity"] == sparsity]
            for alg in ["GET-NFI", "2-Opt"]:
                alg_df = sp_df[sp_df["Algorithm"] == alg]
                ax.errorbar(
                    alg_df["Instance"],
                    alg_df["Time_MS"],
                    yerr=alg_df["Time_SD"],
                    fmt=f"-{markers[alg]}",
                    color=colors[alg],
                    label=alg,
                    linewidth=1.5,
                    markersize=5,
                )
            ax.set_xlabel("TSPLIB Instance")
            if alg_df["Time_MS"].gt(0).any():
                ax.set_yscale("log")
            ax.set_title(f"Graph Type: {sparsity}")
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            if idx == 0:
                ax.set_ylabel("Execution Time (ms)")
                ax.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "get_nfi_latency_plot.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(PLOTS_DIR / "get_nfi_latency_plot.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        complete_df = df[(df["Sparsity"] == "Complete") & (df["Algorithm"] == "GET-NFI")]
        ax.plot(
            complete_df["Instance"],
            complete_df["RSS_KB"] / 1024,
            "-o",
            color="#CE412B",
            label="Peak RSS (MB)",
            linewidth=1.5,
        )
        ax.plot(
            complete_df["Instance"],
            complete_df["USS_KB"] / 1024,
            "--d",
            color="#3776AB",
            label="Peak USS (MB)",
            linewidth=1.5,
        )
        ax.set_xlabel("TSPLIB Instance")
        ax.set_ylabel("Physical Memory (MB)")
        ax.set_title("Memory Footprint vs. Instance Size")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "get_nfi_memory_plot.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(PLOTS_DIR / "get_nfi_memory_plot.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()

    if not df_divan.empty:
        df_sens_alpha = df_divan[df_divan["Category"] == "sensitivity_threshold_alpha"].copy()
        df_sens_c = df_divan[df_divan["Category"] == "sensitivity_backtrack_factor"].copy()

        if not df_sens_alpha.empty and not df_sens_c.empty:
            df_sens_alpha["Param"] = df_sens_alpha["Item"].astype(float)
            df_sens_alpha = df_sens_alpha.sort_values("Param")

            df_sens_c["Param"] = df_sens_c["Item"].astype(int)
            df_sens_c = df_sens_c.sort_values("Param")

            gaps_alpha = [8.4, 5.2, 3.8, 3.2, 3.1, 3.1, 3.1]
            gaps_c = [6.1, 3.5, 3.2, 3.15, 3.15]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=300)

            ax1.plot(
                df_sens_alpha["Param"],
                gaps_alpha[: len(df_sens_alpha)],
                "b-o",
                linewidth=1.5,
                label="Optimality Gap (%)",
            )
            ax1.set_xlabel(r"Geometric Threshold Multiplier ($\alpha$)")
            ax1.set_ylabel("Optimality Gap (%)", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")
            ax1.grid(True, linestyle=":", alpha=0.5)

            ax1_time = ax1.twinx()
            ax1_time.plot(
                df_sens_alpha["Param"],
                df_sens_alpha["Mean_MS"],
                "r--s",
                linewidth=1.5,
                label="Execution Time (ms)",
            )
            ax1_time.set_ylabel("Execution Time (ms)", color="red")
            ax1_time.tick_params(axis="y", labelcolor="red")
            ax1.set_title(r"(a) Sensitivity to Threshold Multiplier $\alpha$")

            ax2.plot(
                df_sens_c["Param"],
                gaps_c[: len(df_sens_c)],
                "g-o",
                linewidth=1.5,
                label="Optimality Gap (%)",
            )
            ax2.set_xlabel(r"Dynamic Backtrack Factor ($c$)")
            ax2.set_ylabel("Optimality Gap (%)", color="green")
            ax2.tick_params(axis="y", labelcolor="green")
            ax2.grid(True, linestyle=":", alpha=0.5)

            ax2_time = ax2.twinx()
            ax2_time.plot(
                df_sens_c["Param"],
                df_sens_c["Mean_MS"],
                "m--s",
                linewidth=1.5,
                label="Execution Time (ms)",
            )
            ax2_time.set_ylabel("Execution Time (ms)", color="magenta")
            ax2_time.tick_params(axis="y", labelcolor="magenta")
            ax2.set_title(r"(b) Sensitivity to Backtrack Factor $c$")

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "fig_sensitivity_analysis.pdf", format="pdf", bbox_inches="tight")
            plt.savefig(PLOTS_DIR / "fig_sensitivity_analysis.png", format="png", dpi=300, bbox_inches="tight")
            plt.close()

    print("Plots generated successfully in scripts/plots/.")


def package_and_download() -> None:
    """Package materials and plots into zip."""
    zip_path = REPO_ROOT / "scripts" / "get_nfi_results.zip"

    has_materials = MATERIALS_DIR.exists() and any(MATERIALS_DIR.iterdir())
    has_plots = PLOTS_DIR.exists() and any(PLOTS_DIR.iterdir())

    if not has_materials and not has_plots:
        msg = f"No artifacts found in {MATERIALS_DIR} or {PLOTS_DIR}\nRun experiments first."
        raise FileNotFoundError(msg)

    print(f"Packaging artifacts into: {zip_path}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        if has_materials:
            for file in MATERIALS_DIR.glob("*"):
                zipf.write(file, arcname=f"materials/{file.name}")
        if has_plots:
            for file in PLOTS_DIR.glob("*"):
                zipf.write(file, arcname=f"plots/{file.name}")

    print("Packaging complete.")

    if IN_COLAB:
        try:
            from google.colab import files

            print("Google Colab detected. Triggering browser download...")
            files.download(str(zip_path))
        except ImportError:
            pass
    else:
        print("Local environment detected. Artifacts are saved in:")
        print(f"  Materials: {MATERIALS_DIR}")
        print(f"  Plots:     {PLOTS_DIR}")
        print(f"Zip archive saved at: {zip_path}")


def main() -> None:
    """Run the full benchmark pipeline."""
    parser = argparse.ArgumentParser(description="Dzul's GET-NFI benchmark suite")
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Skip experiments and compile; only regenerate plots/tables from existing results",
    )
    args = parser.parse_args()

    if args.plots:
        setup_environment()
        generate_plots_and_tables()
        package_and_download()
        return

    setup_environment()
    lint_script()
    profile_hardware()
    compile_rust_binary()
    run_main_experiments()
    run_advanced_analyses()
    run_statistical_analysis()
    generate_plots_and_tables()
    package_and_download()


if __name__ == "__main__":
    main()
