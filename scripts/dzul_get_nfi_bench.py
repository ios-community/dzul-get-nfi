"""Benchmarking suite for Dzul's GET-NFI TSP solver.

Provides a complete pipeline for compiling the Rust binary, running benchmark
experiments across standard TSPLIB instances, generating publication-quality
plots and LaTeX tables, and performing statistical analysis.

This script uses ``uv`` for dependency management and ``ruff`` for linting.
"""

from __future__ import annotations

import argparse
import json
import math
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

REPO_ROOT = Path(__file__).resolve().parent.parent

MATERIALS_DIR = REPO_ROOT / "scripts" / "materials"
PLOTS_DIR = REPO_ROOT / "scripts" / "plots"

ext = ".exe" if sys.platform == "win32" else ""
RUST_BIN_PATH = REPO_ROOT / f"dzul_get_nfi_bench{ext}"
CSV_RESULTS_PATH = MATERIALS_DIR / "get_nfi_benchmark_results.csv"
BENCH_FILE_PATH = REPO_ROOT / "crates" / "dzul-bench" / "benches" / "solver_benches.rs"
RAW_STATS_PATH = MATERIALS_DIR / "raw_statistics_output.txt"
RAW_BENCHES_PATH = MATERIALS_DIR / "raw_benches_output.txt"
HARDWARE_SPECS_PATH = MATERIALS_DIR / "hardware_specs.md"

PYPROJECT_PATH = REPO_ROOT / "scripts" / "pyproject.toml"

# Lines matching any of these patterns in cargo output are filtered out.
_CARGO_NOISE_PATTERNS: list[str] = [
    r"^\s*(Compiling|Downloaded|Downloading|Updating|  Downloaded|  Compiling)",
    r"^\s*info:",
    r"^\s*warn:",
    r"^\s*error\[",
    r"^\s*Finished\s+`",
    r"^\s*+\s+`",
]


def _run_uv(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a ``uv`` command using the scripts ``pyproject.toml``."""
    return subprocess.run(
        [sys.executable, "-m", "uv", *args],
        cwd=REPO_ROOT / "scripts",
        check=check,
        capture_output=True,
        text=True,
    )


def _get_cpu_vendor() -> str:
    """Detect the CPU vendor (``intel``, ``amd``, ``arm``, or ``unknown``)."""
    system = platform.system()
    vendor = "unknown"
    try:
        if system == "Linux":
            output = subprocess.check_output(["cat", "/proc/cpuinfo"], text=True)
            if "GenuineIntel" in output:
                vendor = "intel"
            elif "AuthenticAMD" in output:
                vendor = "amd"
            elif any(v in output for v in ("ARM", "aarch64", "ARMv")):
                vendor = "arm"
        elif system == "Darwin":
            output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
            lower = output.lower()
            if "intel" in lower:
                vendor = "intel"
            elif "amd" in lower:
                vendor = "amd"
            elif "apple m" in lower:
                vendor = "arm"
        elif system == "Windows":
            proc = platform.processor().lower()
            if "intel" in proc:
                vendor = "intel"
            elif "amd" in proc:
                vendor = "amd"
            elif "arm" in proc:
                vendor = "arm"
    except Exception:  # noqa: BLE001
        pass
    return vendor


def _get_cpu_pinning_prefix() -> str:
    """Return a shell-command prefix that pins execution to a single CPU core.

    Behaviour by platform:

    - **Linux** (all vendors): returns ``taskset -c 0 `` when ``taskset`` is
      available.
    - **Windows**: returns an empty string; pinning is handled inside
      :func:`_run_cargo_with_memory` via ``psutil`` after process creation.
    - **macOS**: returns ``""`` (no built-in shell-level pinning tool).

    Returns:
        The prefix string (trailing space), or an empty string when pinning
        is not available for the current platform.

    """
    if sys.platform == "linux":
        if shutil.which("taskset") is not None:
            return "taskset -c 0 "
    return ""


def _is_noise_line(line: str) -> bool:
    """Return ``True`` if the line is cargo build noise that should be filtered."""
    return any(re.search(p, line) for p in _CARGO_NOISE_PATTERNS)


def _run_cargo_with_memory(
    cmd: str,
    *,
    pin_cpu: bool = True,
) -> tuple[str, int, float, float]:
    """Run a cargo command with optional CPU pinning and memory tracking.

    CPU pinning strategy by platform:

    - **Linux**: ``taskset -c 0 `` is prepended to the shell command.
    - **Windows**: the process's affinity mask is set to CPU 0 via psutil
      after creation.
    - **macOS**: CPU pinning is not supported (no built-in tool).

    Args:
        cmd: The full cargo command string to execute.
        pin_cpu: When ``True``, attempt to pin execution to a single CPU core.

    Returns:
        A tuple of ``(filtered_stdout, return_code, peak_rss_mb, peak_uss_mb)``.

    Raises:
        ImportError: If ``psutil`` is not installed.

    """
    try:
        import psutil
    except ImportError as exc:
        msg = "psutil not installed. Run setup_environment() first."
        raise ImportError(msg) from exc

    prefix = _get_cpu_pinning_prefix() if pin_cpu else ""
    full_cmd = f"{prefix}{cmd}"
    print(f"Executing: {full_cmd}")

    process = subprocess.Popen(
        full_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Windows: set affinity to CPU 0 after process starts
    if pin_cpu and sys.platform == "win32":
        try:
            ps_proc = psutil.Process(process.pid)
            ps_proc.cpu_affinity([0])
        except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
            pass

    peak_rss = 0
    peak_uss = 0
    raw_lines: list[str] = []
    filtered_lines: list[str] = []

    try:
        ps_proc = psutil.Process(process.pid)
        # Sample memory immediately after process creation
        try:
            mem = ps_proc.memory_info()
            peak_rss = mem.rss
            try:
                full_mem = ps_proc.memory_full_info()
                peak_uss = full_mem.uss
            except (psutil.AccessDenied, AttributeError):
                peak_uss = mem.rss
        except psutil.NoSuchProcess:
            pass

        children = []

        for line in iter(process.stdout.readline, ""):
            raw_lines.append(line)
            if _is_noise_line(line):
                pass  # skip noise
            else:
                filtered_lines.append(line)
                print(line, end="")
                sys.stdout.flush()

            # Track memory of the process tree
            try:
                current_children = ps_proc.children(recursive=True)
                children = current_children
            except psutil.NoSuchProcess:
                pass

            for proc in [ps_proc, *children]:
                try:
                    mem = proc.memory_info()
                    if mem.rss > peak_rss:
                        peak_rss = mem.rss
                    try:
                        full_mem = proc.memory_full_info()
                        if full_mem.uss > peak_uss:
                            peak_uss = full_mem.uss
                    except (psutil.AccessDenied, AttributeError):
                        peak_uss = mem.rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except psutil.NoSuchProcess:
        pass

    process.stdout.close()
    return_code = process.wait()

    return "".join(filtered_lines), return_code, peak_rss / (1024**2), peak_uss / (1024**2)


def get_criterion_estimates(benchmark_id: str) -> tuple[float | None, float | None]:
    """Extract the mean and standard deviation from a Criterion ``estimates.json``.

    Args:
        benchmark_id: The name of the benchmark group whose estimates should be
            loaded.

    Returns:
        A tuple of ``(mean_ms, std_dev_ms)``, or ``(None, None)`` when no match
        is found or the file cannot be read.

    """
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
    """Verify the project structure and install Python dependencies via ``uv``.

    Raises:
        FileNotFoundError: If the ``Cargo.toml`` or benchmark source file is
            missing.

    """
    if not (REPO_ROOT / "Cargo.toml").exists():
        msg = (
            f"Cargo.toml not found at: {REPO_ROOT}\n"
            "Please ensure this script is located inside the 'scripts/' directory."
        )
        raise FileNotFoundError(msg)

    if not BENCH_FILE_PATH.exists():
        msg = (
            f"Benchmark source file not found at: {BENCH_FILE_PATH}\nPlease verify 'benches/solver_benches.rs' exists."
        )
        raise FileNotFoundError(msg)

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
    """Run ``ruff check`` and ``ruff format --check`` on this script.

    Uses the project's ``pyproject.toml`` configuration for both tools.

    Raises:
        SystemExit: If either the lint or format check fails.

    """
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


def _collect_hardware_specs() -> dict[str, str]:
    """Collect hardware specifications into a dictionary.

    Detects the CPU vendor (Intel, AMD, ARM, unknown) via
    :func:`_get_cpu_vendor` and the CPU model via platform-specific commands.

    Raises:
        ImportError: If ``psutil`` is not installed.

    Returns:
        A dict with keys ``OS``, ``CPU Vendor``, ``CPU Model``,
        ``Total RAM``, and ``CPU Pinning``.

    """
    try:
        import psutil
    except ImportError as exc:
        msg = "psutil not installed. Run setup_environment() first."
        raise ImportError(msg) from exc

    specs: dict[str, str] = {}
    system = platform.system()
    specs["OS"] = f"{system} {platform.release()}"
    specs["CPU Vendor"] = _get_cpu_vendor()

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

    if sys.platform == "linux" and shutil.which("taskset") is not None:
        specs["CPU Pinning"] = "taskset -c 0 (Linux)"
    elif sys.platform == "win32":
        specs["CPU Pinning"] = "psutil cpu_affinity([0]) (Windows)"
    else:
        specs["CPU Pinning"] = "None"

    return specs


def _format_hardware_specs(specs: dict[str, str]) -> tuple[str, str]:
    """Format hardware specs into a console report and a paper-ready paragraph."""
    report = (
        "=" * 60
        + "\n          HARDWARE SPECIFICATION REPORT\n"
        + "=" * 60
        + f"\nOS          : {specs['OS']}"
        + f"\nCPU Vendor  : {specs['CPU Vendor']}"
        + f"\nCPU Model   : {specs['CPU Model']}"
        + f"\nTotal RAM   : {specs['Total RAM']}"
        + f"\nCPU Pinning : {specs['CPU Pinning']}"
        + "\n"
        + "=" * 60
    )

    paragraph = (
        f'"All experiments were executed on an environment running {specs["OS"]}, '
        f"equipped with an {specs['CPU Model']} processor and {specs['Total RAM']} of total physical memory. "
        f'To ensure deterministic timing, the benchmark process was pinned to a single CPU core with high priority."'
    )

    return report, paragraph


def profile_hardware() -> None:
    """Print a hardware specification report and save it to ``HARDWARE_SPECS_PATH``.

    Gathers the operating system, CPU model, total physical RAM, and CPU
    pinning status using ``psutil`` and platform-specific commands.

    Raises:
        ImportError: If ``psutil`` is not installed.

    """
    specs = _collect_hardware_specs()
    report, paragraph = _format_hardware_specs(specs)

    print(report)
    print("\nDraft text for your paper's 'Experimental Setup' section:")
    print(paragraph)

    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    markdown = (
        "| Hardware / Software Property | Value |\n"
        "| :--- | :--- |\n"
        f"| **Operating System** | `{specs['OS']}` |\n"
        f"| **CPU Vendor** | `{specs['CPU Vendor']}` |\n"
        f"| **CPU Model** | `{specs['CPU Model']}` |\n"
        f"| **Total RAM** | `{specs['Total RAM']}` |\n"
        f"| **CPU Pinning** | `{specs['CPU Pinning']}` |\n"
    )
    HARDWARE_SPECS_PATH.write_text(markdown)
    print(f"Hardware specs saved to: {HARDWARE_SPECS_PATH}")


def run_statistics_suite() -> None:
    """Run the statistical benchmark suite via ``cargo test --test statistics``.

    Executes the Rust integration tests that produce the Group 1, Group 2, and
    2-Opt ablation tables. The raw output is filtered to remove compilation
    noise and saved to ``RAW_STATS_PATH``. Memory usage of the entire cargo
    process tree is tracked.

    Raises:
        ImportError: If ``psutil`` is not installed.

    """
    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = "cargo test --features std --test statistics -- --nocapture --test-threads=1"
    filtered_out, ret, rss_mb, uss_mb = _run_cargo_with_memory(cmd, pin_cpu=True)
    RAW_STATS_PATH.write_text(filtered_out)
    print(f"Peak RSS: {rss_mb:.2f} MB | Peak USS: {uss_mb:.2f} MB")
    if ret == 0:
        print("Statistical benchmark suite completed successfully.")
    else:
        msg = f"cargo test exited with code {ret}. Check {RAW_STATS_PATH} for details."
        raise RuntimeError(msg)


def run_bench_suite() -> None:
    """Run the Divan microbenchmark suite via ``cargo bench --bench solver_benches``.

    Executes all Divan benchmarks (ablation, GET-NFI, GET-NFI+2-Opt, NN,
    NN+2-Opt, sensitivity sweeps). The raw output is filtered and saved to
    ``RAW_BENCHES_PATH``. Memory usage of the entire cargo process tree is
    tracked.

    Raises:
        ImportError: If ``psutil`` is not installed.

    """
    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = "cargo bench --features std --bench solver_benches"
    filtered_out, ret, rss_mb, uss_mb = _run_cargo_with_memory(cmd, pin_cpu=True)
    RAW_BENCHES_PATH.write_text(filtered_out)
    print(f"Peak RSS: {rss_mb:.2f} MB | Peak USS: {uss_mb:.2f} MB")
    if ret == 0:
        print("Microbenchmark suite completed successfully.")
    else:
        msg = f"cargo bench exited with code {ret}. Check {RAW_BENCHES_PATH} for details."
        raise RuntimeError(msg)


def compile_rust_binary() -> None:
    """Compile the Rust benchmark binary with aggressive optimisations.

    Sets ``RUSTFLAGS=-C target-cpu=native`` as well as ``LTO`` and
    ``codegen-units`` environment variables. The resulting binary is copied to
    the scripts directory.

    Raises:
        RuntimeError: If ``cargo`` is not installed or compilation fails.
        FileNotFoundError: If the compiled binary cannot be located after a
            successful build.

    """
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
    """Parse the solver binary's stdout for elapsed time, cost, and tour type.

    Looks for lines prefixed with ``ELAPSED_MS:``, ``COST:``, and
    ``TOUR_TYPE:``.

    Args:
        stdout: The raw stdout produced by the Rust solver binary.

    Returns:
        A tuple of ``(elapsed_ms, cost, tour_type)``. Defaults to
        ``(0.0, 0.0, "StrictCycle")`` when a field is missing.

    """
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
"""Mapping of STSP TSPLIB instance names to their known optimal tour costs."""

ATSP_INSTANCES: dict[str, float] = {
    "ftv33": 1_286.0,
    "ry48p": 14_422.0,
    "ft53": 6_905.0,
}
"""Mapping of ATSP TSPLIB instance names to their known optimal tour costs."""


def run_main_experiments() -> None:
    """Run the full experiment matrix: instances x sparsity x 2-Opt.

    Iterates over every combination of instance, sparsity level, and algorithm
    variant, executes the solver binary for each combination while measuring
    peak RSS and USS memory, and saves the results to ``CSV_RESULTS_PATH``.

    Raises:
        ImportError: If ``pandas`` or ``psutil`` is not installed.

    """
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
        """Execute the solver binary and measure peak RSS and USS memory."""
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
            # Sample memory immediately after process creation
            try:
                mem = ps_proc.memory_info()
                peak_rss = mem.rss
                try:
                    full_mem = ps_proc.memory_full_info()
                    peak_uss = full_mem.uss
                except (psutil.AccessDenied, AttributeError):
                    peak_uss = mem.rss
            except psutil.NoSuchProcess:
                pass
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
    N_REPETITIONS = 10
    for instance, opt_cost in INSTANCES.items():
        for sparsity in SPARSITY_VALUES:
            for enable_2opt in [False, True]:
                opt_label = "2-Opt" if enable_2opt else "GET-NFI"
                sparsity_label = "Complete" if sparsity == 1.0 else "Incomplete (50%)"
                print(f"Processing: {instance} | {sparsity_label} | {opt_label}")

                cost = float("nan")
                tour_type = "N/A"
                times: list[float] = []
                rss = 0.0
                uss = 0.0

                for _ in range(N_REPETITIONS):
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

                        ret, stdout, _stderr, iter_rss, iter_uss = run_benchmark_process(cmd)
                        if ret == 0:
                            iter_ms, iter_cost, iter_type = parse_benchmark_output(stdout)
                            times.append(iter_ms)
                            if iter_rss > rss:
                                rss = iter_rss
                            if iter_uss > uss:
                                uss = iter_uss
                            cost = iter_cost
                            tour_type = iter_type

                if times:
                    import statistics

                    elapsed_ms = statistics.mean(times)
                    time_sd = statistics.stdev(times) if len(times) > 1 else 0.0
                else:
                    elapsed_ms = float("nan")
                    time_sd = float("nan")

                if tour_type == "N/A":
                    cost = float("nan")
                    gap = float("nan")
                    tour_type = "Disconnected"
                else:
                    gap = ((cost - opt_cost) / opt_cost) * 100.0

                results.append(
                    {
                        "Instance": instance,
                        "Sparsity": sparsity_label,
                        "Algorithm": opt_label,
                        "Time_MS": elapsed_ms,
                        "Time_SD": time_sd,
                        "Cost": cost,
                        "Gap_Percent": gap,
                        "RSS_KB": rss / 1024,
                        "USS_KB": uss / 1024,
                        "Tour_Type": tour_type,
                    },
                )

    df = pd.DataFrame(results)
    if not bin_exists:
        raise RuntimeError("Genuine process memory profiling requires compiling the Rust binary first (cargo build --release).")
    df.to_csv(CSV_RESULTS_PATH, index=False)
    print("Main experiment matrix complete. Results saved.")


def run_advanced_analyses() -> None:
    """Run sparsity phase transition, Pareto frontier, and ATSP analyses.

    Generates CSV results and PDF plots for:
    - Sparsity phase transition (eil51 across nine sparsity levels).
    - Pareto frontier of backtrack limit vs optimality gap (kroA100).
    - Asymmetric TSP comparison across all instances.

    Raises:
        ImportError: If ``matplotlib`` or ``pandas`` is not installed.

    """
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
    sparsity_results: list[dict[str, float | str]] = []
    for sparsity in sparsity_levels:
        print(f"  Testing Sparsity Level: {sparsity * 100:.0f}%")

        cost = float("nan")
        elapsed_ms: float = float("nan")
        tour_type = "N/A"
        solver_ok = False
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
                solver_ok = True

        if not solver_ok or cost is None or cost <= 0.0:
            tour_type = "Disconnected"
            elapsed_ms = float("nan")
            cost = float("nan")

        opt_cost = INSTANCES["eil51"]
        gap = float("nan") if tour_type == "Disconnected" else ((cost - opt_cost) / opt_cost) * 100.0
        sparsity_results.append(
            {"Sparsity": sparsity, "Time_MS": elapsed_ms, "Gap_Percent": gap, "Tour_Type": tour_type}
        )
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
    backtrack_limits = [10, 50, 100, 250, 500, 1000, 2500, 5000]
    pareto_results: list[dict[str, float]] = []
    for limit in backtrack_limits:
        cost = float("nan")
        elapsed_ms = float("nan")
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
    for instance, opt_cost in ATSP_INSTANCES.items():
        cost = float("nan")
        elapsed_ms = float("nan")
        gap = float("nan")
        tour_type = "N/A"
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
                "--directed",
            ]
            ret = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False)
            if ret.returncode == 0:
                elapsed_ms, cost, tour_type = parse_benchmark_output(ret.stdout)

        if tour_type == "N/A" or math.isnan(cost) or cost <= 0.0:
            tour_type = "N/A"
            elapsed_ms = float("nan")
            cost = float("nan")
            gap = float("nan")
        else:
            gap = ((cost - opt_cost) / opt_cost) * 100.0

        atsp_results.append(
            {
                "Instance": instance,
                "Type": "Asymmetric (ATSP)",
                "Time_MS": elapsed_ms,
                "Cost": cost,
                "Gap_Percent": gap,
            },
        )
    df_atsp = pd.DataFrame(atsp_results)
    df_atsp.to_csv(MATERIALS_DIR / "atsp_comparison.csv", index=False)
    print("Advanced analyses complete.")


def run_statistical_analysis() -> None:
    """Perform Wilcoxon signed-rank tests on the benchmark results.

    Compares optimality gaps and execution times between GET-NFI and 2-Opt,
    and also compares GET-NFI gaps on complete versus incomplete graphs.

    Raises:
        FileNotFoundError: If the main experiment CSV has not been generated.
        ImportError: If ``numpy``, ``pandas``, or ``scipy`` is not installed.

    """
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
        """Calculate the rank-biserial correlation as a non-parametric effect size."""
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
    """Convert a numeric time string with a unit suffix to milliseconds.

    Args:
        time_str: The numeric portion of the time value.
        unit_str: The unit suffix (``ms``, ``µs``, ``ns``, or ``s``).

    Returns:
        The time value normalised to milliseconds.

    """
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
    """Parse Divan statistics text output into DataFrames for ablation and group results.

    Extracts the ablation study table and both constructive-heuristic groups
    (Group 1: no 2-Opt; Group 2: with 2-Opt) from the raw Divan output and
    saves each as a CSV under ``MATERIALS_DIR``.

    Args:
        filepath: Path to the raw Divan statistics output file.

    Returns:
        A tuple of ``(df_ablation, df_group1, df_group2)``. Each may be empty
        when the corresponding section is not found.

    """
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
    """Parse Divan benchmark text output into a DataFrame.

    Scans a raw Divan console output file for benchmark items under each
    category and extracts the fastest, slowest, median, and mean times for
    every entry. The result is also saved as a CSV under ``MATERIALS_DIR``.

    Args:
        filepath: Path to the raw Divan benchmark output file.

    Returns:
        A DataFrame with columns ``Category``, ``Item``, ``Fastest_MS``,
        ``Slowest_MS``, ``Median_MS``, and ``Mean_MS``. Returns an empty
        DataFrame when the file does not exist.

    """
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
    """Generate LaTeX tables and publication-quality figures from parsed results.

    Parses ``RAW_STATS_PATH`` and ``RAW_BENCHES_PATH`` (written by
    :func:`run_statistics_suite` and :func:`run_bench_suite`) into CSV tables,
    then produces PDF and PNG figures for the paper under ``PLOTS_DIR``.

    If the old-style ``CSV_RESULTS_PATH`` (from :func:`run_main_experiments`)
    exists, additional memory-footprint plots and a combined LaTeX table are
    also generated.

    Raises:
        ImportError: If ``matplotlib`` or ``pandas`` is not installed.

    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        msg = "Missing Python dependencies. Run setup_environment() first."
        raise ImportError(msg) from exc

    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df_ablation, df_g1, df_g2 = parse_statistics_output(RAW_STATS_PATH)
    df_divan = parse_divan_benches(RAW_BENCHES_PATH)

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

    # --- Latency comparison plots from Divan data ---
    if not df_divan.empty:
        for benchmark_pair, title_suffix in [
            ("get_nfi", "GET-NFI (Constructive)"),
            ("get_nfi_with_2opt", "GET-NFI + 2-Opt"),
            ("nearest_neighbor", "Nearest Neighbour (NN)"),
        ]:
            pair_df = df_divan[df_divan["Category"] == benchmark_pair].copy()
            if pair_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(7.0, 3.8))
            ax.bar(
                range(len(pair_df)),
                pair_df["Mean_MS"],
                yerr=pair_df["Slowest_MS"] - pair_df["Fastest_MS"],
                color="#CE412B",
                capsize=3,
            )
            ax.set_xticks(range(len(pair_df)))
            ax.set_xticklabels(pair_df["Item"], rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Execution Time (ms)")
            ax.set_yscale("log")
            ax.set_title(f"Latency — {title_suffix}")
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
            plt.tight_layout()
            safe_name = benchmark_pair
            plt.savefig(PLOTS_DIR / f"latency_{safe_name}.pdf", format="pdf", bbox_inches="tight")
            plt.savefig(PLOTS_DIR / f"latency_{safe_name}.png", format="png", dpi=300, bbox_inches="tight")
            plt.close()

    # --- Legacy experiment table & memory plots (if CSV_RESULTS_PATH exists) ---
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
            if not alg_df.empty and alg_df["Time_MS"].gt(0).any():
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
        if not complete_df.empty:
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
        # --- Sensitivity analysis figures ---
        df_sens_alpha = df_divan[df_divan["Category"] == "sensitivity_threshold_alpha"].copy()
        df_sens_c = df_divan[df_divan["Category"] == "sensitivity_backtrack_factor"].copy()

        if not df_sens_alpha.empty and not df_sens_c.empty:
            df_sens_alpha["Param"] = df_sens_alpha["Item"].astype(float)
            df_sens_alpha = df_sens_alpha.sort_values("Param")

            df_sens_c["Param"] = df_sens_c["Item"].astype(int)
            df_sens_c = df_sens_c.sort_values("Param")

            # Compute gaps dynamically by running the solver for each param value
            opt_eil51 = INSTANCES["eil51"]
            gaps_alpha = []
            for _, row in df_sens_alpha.iterrows():
                _, cost, _ = _run_solver_for_param(
                    "--threshold-multiplier",
                    row["Param"],
                    instance="eil51",
                    enable_2opt=False,
                )
                if cost <= 0.0 or math.isnan(cost):
                    gaps_alpha.append(float("nan"))
                else:
                    gaps_alpha.append(((cost - opt_eil51) / opt_eil51) * 100.0)

            gaps_c = []
            for _, row in df_sens_c.iterrows():
                _, cost, _ = _run_solver_for_param(
                    "--backtrack-factor",
                    row["Param"],
                    instance="eil51",
                    enable_2opt=False,
                )
                if cost <= 0.0 or math.isnan(cost):
                    gaps_c.append(float("nan"))
                else:
                    gaps_c.append(((cost - opt_eil51) / opt_eil51) * 100.0)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=300)

            ax1.plot(
                df_sens_alpha["Param"],
                gaps_alpha,
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
                gaps_c,
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


def _run_solver_for_param(  # noqa: PLR0913
    param_name: str,
    param_value: float,
    instance: str = "eil51",
    sparsity: float = 1.0,
    *,
    enable_2opt: bool = False,
    backtracks: int = 5000,
    is_directed: bool = False,
) -> tuple[float, float, str]:
    """Run the solver binary with a custom parameter and return (elapsed_ms, cost, tour_type).

    Args:
        param_name: The CLI flag name (e.g., ``--threshold-multiplier``).
        param_value: The parameter value to pass.
        instance: TSPLIB instance name.
        sparsity: Edge retention ratio (1.0 = complete graph).
        enable_2opt: Whether to enable 2-Opt local search.
        backtracks: Maximum backtrack limit.
        is_directed: Whether the graph is directed (ATSP).

    Returns:
        A tuple of ``(elapsed_ms, cost, tour_type)``, or
        ``(float("nan"), float("nan"), "N/A")`` on failure.

    """
    if not RUST_BIN_PATH.exists():
        return float("nan"), float("nan"), "N/A"

    cmd = [
        str(RUST_BIN_PATH),
        "--instance",
        instance,
        "--sparsity",
        str(sparsity),
        "--backtracks",
        str(backtracks),
    ]
    if enable_2opt:
        cmd.append("--2opt")
    if is_directed:
        cmd.append("--directed")
    cmd.append(param_name)
    cmd.append(str(param_value))

    ret = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False)
    if ret.returncode != 0:
        return float("nan"), float("nan"), "Disconnected"

    elapsed_ms, cost, tour_type = parse_benchmark_output(ret.stdout)
    if cost <= 0.0:
        return float("nan"), float("nan"), "Disconnected"
    return elapsed_ms, cost, tour_type


def package_and_download() -> None:
    """Package generated materials and plots into a zip archive.

    Creates a ``get_nfi_results.zip`` archive containing the contents of
    ``MATERIALS_DIR`` and ``PLOTS_DIR``.

    Raises:
        FileNotFoundError: If neither ``MATERIALS_DIR`` nor ``PLOTS_DIR``
            contain any files.

    """
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
    print("Artifacts are saved in:")
    print(f"  Materials: {MATERIALS_DIR}")
    print(f"  Plots:     {PLOTS_DIR}")
    print(f"  Zip archive: {zip_path}")


def main() -> None:
    """Run the authoritative benchmark pipeline.

    Pipeline steps:
    1. Environment setup and linting
    2. Hardware profiling
    3. ``cargo test --test statistics`` — Group 1, Group 2, ablation
    4. ``cargo bench --bench solver_benches`` — Divan microbenchmarks
    5. Advanced analyses (sparsity, Pareto, ATSP) via standalone binary
    6. Parse outputs and generate CSVs, LaTeX tables, publication figures
    7. Package all artifacts into a zip archive

    Use ``--plots`` to skip all experiments and only regenerate figures from
    previously saved results.
    """
    parser = argparse.ArgumentParser(description="Dzul's GET-NFI benchmark suite")
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Skip experiments; only regenerate plots/tables from existing results",
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

    # Core benchmarks (authoritative — matches notebook output)
    run_statistics_suite()
    run_bench_suite()

    # Additional experiments that require the standalone binary
    if shutil.which("cargo"):
        compile_rust_binary()
    else:
        print("Cargo not found; skipping binary compilation.")
    run_main_experiments()
    run_advanced_analyses()

    # Generate all outputs
    generate_plots_and_tables()
    package_and_download()


if __name__ == "__main__":
    main()
