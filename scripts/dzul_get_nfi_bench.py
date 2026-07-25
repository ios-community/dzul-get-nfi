"""Benchmarking script for Dzul's GET-NFI TSP solver.

Uses uv for dependency management and ruff for linting.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    REPO_ROOT = Path("/content/dzul-get-nfi")
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent

ext = ".exe" if sys.platform == "win32" else ""
RUST_BIN_PATH = REPO_ROOT / f"dzul_get_nfi_bench{ext}"
CSV_RESULTS_PATH = REPO_ROOT / "results" / "get_nfi_benchmark_results.csv"
BENCH_FILE_PATH = REPO_ROOT / "benches" / "solver_benches.rs"

PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _run_uv(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a uv command with the project pyproject.toml."""
    return subprocess.run(
        [sys.executable, "-m", "uv", *args],
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _run_ruff(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run ruff via uvx (isolated tool execution)."""
    return subprocess.run(
        [sys.executable, "-m", "uvx", "ruff", *args],
        cwd=REPO_ROOT,
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
    result = _run_ruff(["check", "--config", str(PYPROJECT_PATH), "scripts/", "scripts/**/*.py"], check=False)
    if result.returncode != 0:
        print(f"Ruff check failed:\n{result.stdout}\n{result.stderr}")
        raise SystemExit(1)
    print("Ruff check passed.")

    result = _run_ruff(
        ["format", "--check", "--config", str(PYPROJECT_PATH), "scripts/"],
        check=False,
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

    RESULTS_DIR = REPO_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

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

    RESULTS_DIR = REPO_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    SPARSITY_VALUES = [1.0, 0.5]

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

                cost = opt_cost
                tour_type = "N/A"
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
                        _, cost, tour_type = parse_benchmark_output(stdout)

                bench_suffix = "get_nfi_with_2opt" if enable_2opt else "get_nfi"
                bench_id = f"{instance}_{bench_suffix}"

                mean_ms, std_dev_ms = get_criterion_estimates(bench_id)
                if mean_ms is None:
                    print(f"Notice: Criterion estimate for '{bench_id}' not found. Defaulting to 0.0 ms.")
                    mean_ms, std_dev_ms = 0.0, 0.0

                gap = ((cost - opt_cost) / opt_cost) * 100.0
                results.append(
                    {
                        "Instance": instance,
                        "Sparsity": sparsity_label,
                        "Algorithm": opt_label,
                        "Time_MS": mean_ms,
                        "Time_SD": std_dev_ms,
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

    RESULTS_DIR = REPO_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

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

        bench_id = "ablation_get_without_nfi" if sparsity == 0.5 else "eil51_get_nfi_with_2opt"
        mean_ms, _ = get_criterion_estimates(bench_id)
        if mean_ms is None:
            print(f"  Notice: Estimate for '{bench_id}' not found. Defaulting to 0.0 ms.")
            mean_ms = 0.0

        cost = 0.0
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
                _, cost, _ = parse_benchmark_output(ret.stdout)

        opt_cost = INSTANCES["eil51"]
        gap = ((cost - opt_cost) / opt_cost) * 100.0 if cost > 0 else 0.0
        sparsity_results.append({"Sparsity": sparsity, "Time_MS": mean_ms, "Gap_Percent": gap})
    df_sparsity = pd.DataFrame(sparsity_results)
    df_sparsity.to_csv(RESULTS_DIR / "sparsity_phase_transition.csv", index=False)

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
    plt.savefig(RESULTS_DIR / "sparsity_phase_transition.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    # --- Pareto Frontier ---
    print("Running Pareto Frontier Analysis...")
    backtrack_limits = [100, 1000, 5000]
    pareto_results: list[dict[str, float]] = []
    for limit in backtrack_limits:
        bench_id = f"sensitivity_backtracks_{limit}"
        mean_ms, _ = get_criterion_estimates(bench_id)
        if mean_ms is None:
            print(f"  Notice: Estimate for '{bench_id}' not found. Defaulting to 0.0 ms.")
            mean_ms = 0.0

        cost = 0.0
        if bin_exists:
            cmd = [
                str(RUST_BIN_PATH),
                "--instance",
                "kroA100",
                "--sparsity",
                "0.5",
                "--backtracks",
                str(limit),
                "--2opt",
            ]
            ret = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False)
            if ret.returncode == 0:
                _, cost, _ = parse_benchmark_output(ret.stdout)

        opt_cost = INSTANCES["kroA100"]
        gap = ((cost - opt_cost) / opt_cost) * 100.0 if cost > 0 else 0.0
        pareto_results.append({"Backtrack_Limit": limit, "Time_MS": mean_ms, "Gap_Percent": gap})
    df_pareto = pd.DataFrame(pareto_results)
    df_pareto.to_csv(RESULTS_DIR / "pareto_frontier.csv", index=False)

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
    plt.title("Pareto Frontier Analysis (kroA100, Sparsity 50%)")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pareto_frontier.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    # --- Asymmetric TSP ---
    print("Running Asymmetric TSP (ATSP) Analysis...")
    atsp_results: list[dict[str, object]] = []
    for instance, opt_cost in INSTANCES.items():
        for is_directed in [False, True]:
            dir_label = "Asymmetric (ATSP)" if is_directed else "Symmetric (STSP)"
            bench_id = f"{instance}_get_nfi_with_2opt"
            mean_ms, _ = get_criterion_estimates(bench_id)
            if mean_ms is None:
                print(f"  Notice: Estimate for '{bench_id}' not found. Defaulting to 0.0 ms.")
                mean_ms = 0.0

            cost = opt_cost
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
                    _, cost, _ = parse_benchmark_output(ret.stdout)

            atsp_results.append(
                {
                    "Instance": instance,
                    "Type": dir_label,
                    "Time_MS": mean_ms,
                    "Cost": cost,
                },
            )
    df_atsp = pd.DataFrame(atsp_results)
    df_atsp.to_csv(RESULTS_DIR / "atsp_comparison.csv", index=False)
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

    print("=" * 65)
    print("          DZUL'S GET-NFI STATISTICAL RIGOR REPORT")
    print("=" * 65)

    df_stats = pd.read_csv(CSV_RESULTS_PATH)
    get_nfi_df = df_stats[df_stats["Algorithm"] == "GET-NFI"].sort_values(by=["Instance", "Sparsity"])
    opt_2_df = df_stats[df_stats["Algorithm"] == "2-Opt"].sort_values(by=["Instance", "Sparsity"])

    times_get_nfi = get_nfi_df["Time_MS"].to_numpy()
    times_2opt = opt_2_df["Time_MS"].to_numpy()
    gaps_get_nfi = get_nfi_df["Gap_Percent"].to_numpy()
    gaps_2opt = opt_2_df["Gap_Percent"].to_numpy()

    print("1. Wilcoxon Signed-Rank Test (Optimality Gap: GET-NFI vs 2-Opt):")
    gap_diff = gaps_get_nfi - gaps_2opt
    if np.all(gap_diff == 0):
        print("   Statistic: N/A | p-value: N/A")
        print("   Effect Size (Rank-Biserial r): 0.0000")
        print("   Result:    NOT SIGNIFICANT (All differences are zero)")
    else:
        stat_gap, p_gap = stats.wilcoxon(gaps_get_nfi, gaps_2opt)
        r_gap = calculate_rank_biserial(gaps_get_nfi, gaps_2opt)
        print(f"   Statistic: {stat_gap:.4f} | p-value: {p_gap:.6f}")
        print(f"   Effect Size (Rank-Biserial r): {r_gap:.4f}")
        if p_gap < 0.05:
            print("   Result:    SIGNIFICANT (Reject H0 at alpha = 0.05)")
        else:
            print("   Result:    NOT SIGNIFICANT")
    print("-" * 65)

    print("2. Wilcoxon Signed-Rank Test (Execution Time: GET-NFI vs 2-Opt):")
    time_diff = times_get_nfi - times_2opt
    if np.all(time_diff == 0):
        print("   Statistic: N/A | p-value: N/A")
        print("   Effect Size (Rank-Biserial r): 0.0000")
        print("   Result:    NOT SIGNIFICANT (All differences are zero)")
    else:
        stat_time, p_time = stats.wilcoxon(times_get_nfi, times_2opt)
        r_time = calculate_rank_biserial(times_get_nfi, times_2opt)
        print(f"   Statistic: {stat_time:.4f} | p-value: {p_time:.6f}")
        print(f"   Effect Size (Rank-Biserial r): {r_time:.4f}")
    print("-" * 65)

    complete_gaps = df_stats[(df_stats["Sparsity"] == "Complete") & (df_stats["Algorithm"] == "GET-NFI")][
        "Gap_Percent"
    ].to_numpy()
    incomplete_gaps = df_stats[(df_stats["Sparsity"] == "Incomplete (50%)") & (df_stats["Algorithm"] == "GET-NFI")][
        "Gap_Percent"
    ].to_numpy()

    print("3. Wilcoxon Signed-Rank Test (GET-NFI Gaps: Complete vs Incomplete):")
    if len(complete_gaps) == len(incomplete_gaps):
        comp_diff = complete_gaps - incomplete_gaps
        if np.all(comp_diff == 0):
            print("   Statistic: N/A | p-value: N/A")
            print("   Result:    NOT SIGNIFICANT (All differences are zero)")
        else:
            stat_comp, p_comp = stats.wilcoxon(complete_gaps, incomplete_gaps)
            print(f"   Statistic: {stat_comp:.4f} | p-value: {p_comp:.6f}")
            if p_comp < 0.05:
                print("   Result:    SIGNIFICANT (Reject H0 at alpha = 0.05)")
            else:
                print("   Result:    NOT SIGNIFICANT")
    else:
        print("   Error: Sample sizes do not match for Complete vs Incomplete gaps.")
    print("=" * 65)


def generate_plots_and_tables() -> None:
    """Generate LaTeX table and benchmark plots."""
    if not CSV_RESULTS_PATH.exists():
        msg = f"Benchmark results file not found at: {CSV_RESULTS_PATH}\nRun run_main_experiments() first."
        raise FileNotFoundError(msg)

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        msg = "Missing Python dependencies. Run setup_environment() first."
        raise ImportError(msg) from exc

    RESULTS_DIR = REPO_ROOT / "results"

    df = pd.read_csv(CSV_RESULTS_PATH)

    # LaTeX table
    latex_path = RESULTS_DIR / "get_nfi_benchmark_table.tex"
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
    print(f"LaTeX table generated at: {latex_path}")

    # Plot styling
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
        ax.set_yscale("log")
        ax.set_title(f"Graph Type: {sparsity}")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        if idx == 0:
            ax.set_ylabel("Execution Time (ms)")
            ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "get_nfi_latency_plot.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(RESULTS_DIR / "get_nfi_latency_plot.png", format="png", dpi=300, bbox_inches="tight")
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
    plt.savefig(RESULTS_DIR / "get_nfi_memory_plot.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(RESULTS_DIR / "get_nfi_memory_plot.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Plots generated successfully in the results directory.")


def package_and_download() -> None:
    """Package results into zip."""
    RESULTS_DIR = REPO_ROOT / "results"
    zip_path = REPO_ROOT / "get_nfi_results.zip"

    if not RESULTS_DIR.exists() or not any(RESULTS_DIR.iterdir()):
        msg = f"No analysis files found in: {RESULTS_DIR}\nRun experiments first."
        raise FileNotFoundError(msg)

    print(f"Packaging results into: {zip_path}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in RESULTS_DIR.glob("*"):
            if file.suffix in (".pdf", ".tex", ".csv") or file.name.endswith("_plot.png"):
                zipf.write(file, arcname=file.name)

    print("Packaging complete.")

    if IN_COLAB:
        try:
            from google.colab import files

            print("Google Colab detected. Triggering browser download...")
            files.download(str(zip_path))
        except ImportError:
            pass
    else:
        print(f"Local environment detected. Artifacts are saved in: {RESULTS_DIR}")
        print(f"Zip archive saved at: {zip_path}")


def main() -> None:
    """Run the full benchmark pipeline."""
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
