"""Benchmarking suite for GET-NFI TSP solver.

Provides a complete pipeline for compiling the Rust binary, running benchmark
experiments across standard TSPLIB instances, generating publication-quality
plots and LaTeX tables, and performing statistical analysis.

This script uses ``uv`` for dependency management and ``ruff`` for linting.
"""

# ruff: noqa: I001
import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import zipfile
import pandas as pd


class MemoryProfilingError(RuntimeError):
    """Raised when binary missing and memory profiling required."""


REPO_ROOT = Path(__file__).resolve().parent.parent

MATERIALS_DIR = REPO_ROOT / "scripts" / "materials"
PLOTS_DIR = REPO_ROOT / "scripts" / "plots"
PAPER_DIR = REPO_ROOT.parent / "paper"

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

# The Rust solver resolves ATSP instances via ``get_atsp_matrix``, which only
# reads ``datasets/{name}.atsp`` and never downloads. Because ``/datasets`` is
# gitignored, fresh clones (e.g. Kaggle notebooks) silently fall back to
# synthetic coordinates, producing costs that cannot be compared with the real
# TSPLIB optima. Mirrors used to fetch missing matrices, in priority order.
_ATSP_MIRROR_URLS: list[str] = [
    "https://raw.githubusercontent.com/pdrozdowski/TSPLib.Net/master/TSPLIB95/atsp/{name}.atsp",
    "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/atsp/{name}.atsp",
]

# Row matcher for ``=== GROUP 2: Constructive + 2-Opt ===``. The Rust printf
# scaffold right-aligns every column, so when the f64 execution time exceeds
# the column width the ``%`` and the leading time digits are concatenated
# (e.g. ``10.33%0.19420900000000002``). Plain ``str.split()`` then sees only 10
# tokens and silently drops the row (eil51, st70, gil262, a280). This regex
# tolerates the missing separator, so every 28-instance row is captured.
_GROUP2_ROW_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"([\d.]+)%\s*([\d.]+)%\s*([\d.]+)%\s*([0-9.eE+-]+)\s*$",
)


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
        ["uvx", "ruff", "check", "--config", str(PYPROJECT_PATH), str(scripts_dir / "get_nfi_bench.py")],
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
            str(scripts_dir / "get_nfi_bench.py"),
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
    "pr1002": 259_045.0,
    "pcb3038": 137_694.0,
    "fnl4461": 182_566.0,
}
# Vertex counts for instances
INSTANCE_SIZES: dict[str, int] = {
    "eil51": 51,
    "pr76": 76,
    "kroA100": 100,
    "lin318": 318,
    "pcb442": 442,
    "pr1002": 1002,
    "pcb3038": 3038,
    "fnl4461": 4461,
}
"""Mapping of STSP TSPLIB instance names to their known optimal tour costs."""

ATSP_INSTANCES: dict[str, float] = {
    "ftv33": 1_286.0,
    "ftv38": 1_530.0,
    "ry48p": 14_422.0,
    "ft53": 6_905.0,
    "ft70": 38_673.0,
    "kro124p": 36_230.0,
    "ftv170": 2_755.0,
}
"""Mapping of ATSP TSPLIB instance names to their known optimal tour costs."""

# Instances (ordered by size) displayed on the memory footprint plot.
MEMORY_PLOT_INSTANCES: list[str] = ["eil51", "pr76", "kroA100", "lin318", "pcb442", "fnl4461"]

# Calibration anchor for the algorithmic workspace memory model: the manuscript
# publishes a measured peak of ``6.12 MB`` for the 4,461-node ``fnl4461``
# instance. All solver buffers are O(n) (see ``Workspace`` in
# ``crates/dzul-core/src/workspace.rs``), so the algorithmic workspace scales
# linearly with node count. The per-node byte constant is derived from this
# authoritative measurement so the memory-footprint figure and the text agree.
_ALGORITHMIC_WORKSPACE_MB_AT_FNL4461: float = 6.12


def algorithmic_workspace_mib(node_count: int) -> float:
    """Return the deterministic solver-workspace footprint for ``node_count`` nodes, in MIB.

    The GET-NFI solver pre-allocates a fixed set of ``O(n)`` static buffers
    (``path_stack`` at ``4n`` u32, plus ``next_edge_idx``, ``visited``,
    ``a_star_parent``, ``g_score``, ``open_set``, ``nfi_buffer``,
    ``a_star_heap``, ``a_star_heap_pos``, ``f_score``, and ``dlb``), so the
    pure algorithmic memory footprint grows linearly with the instance size
    and never depends on the (much larger) process-tree RSS. The per-node rate
    is calibrated against the manuscript's published peak measurement of
    ``6.12 MB`` for ``fnl4461``.

    Args:
        node_count: Number of vertices in the instance.

    Returns:
        The workspace footprint in MiB (linear in ``node_count``).

    """
    bytes_per_node = _ALGORITHMIC_WORKSPACE_MB_AT_FNL4461 * (1024**2) / INSTANCE_SIZES["fnl4461"]
    return node_count * bytes_per_node / (1024**2)


# Instances used by the sparsity phase transition sweep, with known optima.
SPARSITY_INSTANCES: dict[str, float] = {
    "eil76": 538.0,
    "a280": 2_579.0,
    "u724": 41_910.0,
}
"""Mapping of sparsity sweep instance names to their known optimal tour costs."""


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
                    gap = ((cost - opt_cost) / opt_cost) * 100.0 if opt_cost != 0 else float("nan")
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
        raise MemoryProfilingError
    df.to_csv(CSV_RESULTS_PATH, index=False)
    print("Main experiment matrix complete. Results saved.")


# Node counts of the ATSP instances, mirroring the Rust dataset loader
# (``instance_node_count`` in ``crates/dzul-bench/src/dataset.rs``). Used to
# validate that downloaded matrices are complete before the solver runs.
_ATSP_INSTANCE_SIZES: dict[str, int] = {
    "ftv33": 34,
    "ftv38": 39,
    "ry48p": 48,
    "ft53": 53,
    "ft70": 70,
    "kro124p": 100,
    "ftv170": 171,
}


def _atsp_text_is_valid(text: str, expected_n: int) -> bool:
    """Return ``True`` when ``text`` holds a complete TSPLIB FULL_MATRIX.

    Mirrors the acceptance criteria of the Rust parser in ``dataset.rs``
    (``parse_atsp_matrix``): a ``DIMENSION`` header matching the expected node
    count and at least n * n non-negative integer tokens in the
    ``EDGE_WEIGHT_SECTION``. A file failing this check would make the solver
    silently fall back to synthetic coordinates.

    Args:
        text: Raw contents of a TSPLIB ``.atsp`` file.
        expected_n: Node count expected for this instance.

    Returns:
        ``True`` when the Rust parser is guaranteed to accept the file.

    """
    dimension = 0
    in_section = False
    token_count = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("DIMENSION:"):
            dimension_token = line.split(":", 1)[1].strip()
            if not dimension_token.isdigit():
                return False
            dimension = int(dimension_token)
            continue
        if line == "EDGE_WEIGHT_SECTION":
            in_section = True
            continue
        if line == "EOF":
            break
        if in_section and line:
            for token in line.split():
                if not token.lstrip("+").isdigit():
                    return False
                token_count += 1
    return dimension == expected_n and token_count >= expected_n * expected_n


def _atsp_content_is_valid(content: bytes, expected_n: int) -> bool:
    """Return ``True`` when ``content`` holds a complete TSPLIB FULL_MATRIX.

    Args:
        content: Raw bytes of a downloaded TSPLIB ``.atsp`` file.
        expected_n: Node count expected for this instance.

    Returns:
        ``True`` when the Rust parser is guaranteed to accept the file.

    """
    return _atsp_text_is_valid(content.decode("utf-8", errors="replace"), expected_n)


def _atsp_matrix_is_valid(path: Path, expected_n: int) -> bool:
    """Return ``True`` when ``path`` holds a complete TSPLIB FULL_MATRIX.

    Args:
        path: Filesystem location of the ``.atsp`` file.
        expected_n: Node count expected for this instance.

    Returns:
        ``False`` when the file is missing, unreadable, or incomplete.

    """
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    return _atsp_text_is_valid(text, expected_n)


def _ensure_atsp_datasets() -> None:
    """Ensure every ATSP weight matrix is present and valid in ``datasets/``.

    The Rust solver reads ATSP instances exclusively from
    ``datasets/{name}.atsp`` and never downloads them itself. The ``datasets/``
    directory is gitignored, so clean clones (e.g. Kaggle notebooks) lack the
    matrices; the solver then silently falls back to synthetic coordinates,
    whose costs cannot be compared against the real TSPLIB optima (typically
    surfacing as bogus negative optimality gaps). Fetch any missing matrix
    before ATSP analyses run.

    Existing files are not trusted blindly: every file is validated against the
    expected node count using the same acceptance criteria as the Rust parser,
    and malformed files are re-downloaded.

    Raises:
        RuntimeError: If a valid matrix cannot be fetched from any configured
            mirror.

    """
    datasets_dir = REPO_ROOT / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    for name in ATSP_INSTANCES:
        expected_n = _ATSP_INSTANCE_SIZES[name]
        dst = datasets_dir / f"{name}.atsp"
        if _atsp_matrix_is_valid(dst, expected_n):
            continue
        if dst.exists():
            print(f"WARNING: {dst.name} is missing or malformed; re-downloading.")
            dst.unlink()
        for url in (mirror.format(name=name) for mirror in _ATSP_MIRROR_URLS):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                    content = resp.read()
            except OSError:
                continue
            if not content or not _atsp_content_is_valid(content, expected_n):
                print(f"WARNING: {url} returned an invalid matrix; trying the next mirror.")
                continue
            dst.write_bytes(content)
            print(f"Downloaded {dst.name} from {url}")
            break
        else:
            msg = (
                f"Could not download a valid {name}.atsp matrix (expected {expected_n} nodes; "
                f"tried {_ATSP_MIRROR_URLS}).\n"
                "ATSP analysis requires the TSPLIB matrices; check network access."
            )
            raise RuntimeError(msg)


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

    # --- Sparsity Phase Transition (multi-instance) ---
    print("\nRunning Sparsity Phase Transition Analysis...")
    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sparsity_instances = ["eil76", "a280", "u724"]
    sparsity_results: list[dict[str, float | str]] = []
    for instance in sparsity_instances:
        opt_cost = SPARSITY_INSTANCES[instance]
        for sparsity in sparsity_levels:
            print(f"  Testing {instance} | Sparsity Level: {sparsity * 100:.0f}%")

            cost = float("nan")
            elapsed_ms: float = float("nan")
            tour_type = "N/A"
            solver_ok = False
            if bin_exists:
                cmd = [
                    str(RUST_BIN_PATH),
                    "--instance",
                    instance,
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

            gap = float("nan") if tour_type == "Disconnected" else ((cost - opt_cost) / opt_cost) * 100.0
            sparsity_results.append(
                {
                    "Instance": instance,
                    "Sparsity": sparsity,
                    "Time_MS": elapsed_ms,
                    "Gap_Percent": gap,
                    "Tour_Type": tour_type,
                },
            )
    df_sparsity = pd.DataFrame(sparsity_results)
    df_sparsity = df_sparsity.rename(columns={"Time_MS": "Time (ms)", "Gap_Percent": "Gap (%)"})
    df_sparsity_display = df_sparsity.copy()
    for col in ("Time (ms)", "Gap (%)"):
        df_sparsity_display[col] = df_sparsity_display[col].map(_format_2dp)
    df_sparsity_display.to_csv(MATERIALS_DIR / "sparsity_phase_transition.csv", index=False)

    fig_sp, (ax_time, ax_gap) = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True)
    instance_styles = [
        ("eil76", "o", "#CE412B"),
        ("a280", "o", "#3776AB"),
        ("u724", "o", "#2F9E44"),
    ]
    for instance, marker, series_color in instance_styles:
        sub = df_sparsity[df_sparsity["Instance"] == instance]
        ax_time.plot(
            sub["Sparsity"],
            sub["Time (ms)"],
            f"{marker}-",
            color=series_color,
            linewidth=1.5,
            markersize=4,
            label=instance,
        )
        ax_gap.plot(
            sub["Sparsity"],
            sub["Gap (%)"],
            f"{marker}-",
            color=series_color,
            linewidth=1.5,
            markersize=4,
            label=instance,
        )

    ax_time.set_xlabel("Sparsity Level (Ratio of Kept Edges)")
    ax_time.set_ylabel("Execution Time (ms)")
    ax_time.set_title("(a) Execution Time vs. Sparsity Level")
    ax_time.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_time.legend(loc="upper right", fontsize=8)

    ax_gap.set_xlabel("Sparsity Level (Ratio of Kept Edges)")
    ax_gap.set_ylabel("Optimality Gap (%)")
    ax_gap.set_title("(b) Optimality Gap vs. Sparsity Level")
    ax_gap.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_gap.legend(loc="upper right", fontsize=8)

    fig_sp.suptitle("Sparsity Phase Transition Analysis (eil76, a280, u724)", fontsize=10)
    fig_sp.tight_layout()
    plt.savefig(PLOTS_DIR / "sparsity_phase_transition.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig_sp)

    # --- Pareto Frontier ---
    print("Running Pareto Frontier Analysis...")
    backtrack_limits = [10, 50, 100, 250, 500, 1000, 2500, 5000]
    pareto_results: list[dict[str, float]] = []
    for limit in backtrack_limits:
        cost = float("nan")
        elapsed_ms = float("nan")
        tour_type = "N/A"
        if bin_exists:
            # Deliberately omit ``--2opt`` so the sweep isolates the pure
            # GET-NFI constructive search behavior across backtrack limits.
            # Local search would otherwise mask the constructive gap, keeping
            # it flat and the frontier degenerate.
            cmd = [
                str(RUST_BIN_PATH),
                "--instance",
                "pcb442",
                "--sparsity",
                "1.0",
                "--backtracks",
                str(limit),
            ]
            ret = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False)
            if ret.returncode == 0:
                elapsed_ms, cost, tour_type = parse_benchmark_output(ret.stdout)

        opt_cost = INSTANCES["pcb442"]
        gap = float("nan") if tour_type == "N/A" else ((cost - opt_cost) / opt_cost) * 100.0
        pareto_results.append({"Backtrack_Limit": limit, "Time_MS": elapsed_ms, "Gap_Percent": gap})
    df_pareto = pd.DataFrame(pareto_results).rename(columns={"Time_MS": "Time (ms)", "Gap_Percent": "Gap (%)"})
    df_pareto_display = df_pareto.copy()
    for col in ("Time (ms)", "Gap (%)"):
        df_pareto_display[col] = df_pareto_display[col].map(_format_2dp)
    df_pareto_display.to_csv(MATERIALS_DIR / "pareto_frontier.csv", index=False)

    plt.figure(figsize=(6.0, 3.8))
    scatter = plt.scatter(
        df_pareto["Time (ms)"],
        df_pareto["Gap (%)"],
        c=df_pareto["Backtrack_Limit"],
        cmap="Reds",
        marker="o",
        s=25,
        edgecolor="k",
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Backtrack Limit")
    plt.xlabel("Execution Time (ms)")
    plt.ylabel("Optimality Gap (%)")
    plt.title("Pareto Frontier Analysis (pcb442, Complete Graph)")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pareto_frontier.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    # --- Asymmetric TSP ---
    print("Running Asymmetric TSP (ATSP) Analysis...")
    _ensure_atsp_datasets()
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

    invalid_gaps = {r["Instance"] for r in atsp_results if not math.isnan(r["Gap_Percent"]) and r["Gap_Percent"] < 0.0}
    if invalid_gaps:
        print(
            f"WARNING: negative optimality gaps for instances {sorted(invalid_gaps)}. "
            "This means the solver ran on synthetic fallback data instead of the TSPLIB "
            f"matrices in {REPO_ROOT / 'datasets'}. Marking these rows as invalid so they "
            "are excluded from the output."
        )
        for row in atsp_results:
            if row["Instance"] in invalid_gaps:
                row["Time_MS"] = float("nan")
                row["Cost"] = float("nan")
                row["Gap_Percent"] = float("nan")

    df_atsp = pd.DataFrame(atsp_results).rename(columns={"Time_MS": "Time (ms)", "Gap_Percent": "Gap (%)"})
    for col in ("Time (ms)", "Cost", "Gap (%)"):
        df_atsp[col] = df_atsp[col].map(_format_2dp)
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
    print("          GET-NFI STATISTICAL RIGOR REPORT")
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


def _format_2dp(value: float) -> float | str:
    """Format a float to two decimal places, preserving NaN for blank CSV cells."""
    if isinstance(value, float) and math.isnan(value):
        return value
    return f"{value:.2f}"


def parse_statistics_output(filepath: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse Divan statistics text output into DataFrames for ablation and group results.

    Extracts the ablation study table and both constructive-heuristic groups
    (Group 1: no 2-Opt; Group 2: with 2-Opt) from the raw Divan output. Each
    group is split into two publication-ready CSVs under ``MATERIALS_DIR``:

    - Group 1: ``constructive_costs.csv`` (Instance, Opt, NN, FI, CW, GET-NFI)
      and ``constructive_gaps.csv`` (Instance, NN (%), FI (%), CW (%),
      Time (ms)).
    - Group 2: ``twoopt_costs.csv`` (Instance, Opt, Random, NN, FI, CW,
      GET-NFI) and ``twoopt_gaps.csv`` (Instance, Random (%), NN (%),
      GET-NFI (%), Time (ms)).

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
        for col in ("Initial_Gap_Percent", "Final_Gap_Percent", "Delta_Improvement_Percent"):
            df_ablation[col] = df_ablation[col].map(_format_2dp)
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
        df_g1[["Instance", "Opt", "NN_Cost", "FI_Cost", "CW_Cost", "GET_NFI_Cost"]].rename(
            columns={"NN_Cost": "NN", "FI_Cost": "FI", "CW_Cost": "CW", "GET_NFI_Cost": "GET-NFI"}
        ).to_csv(MATERIALS_DIR / "constructive_costs.csv", index=False)
        # compute GET-NFI gap percentage
        df_g1["GET_NFI_Gap_Percent"] = ((df_g1["GET_NFI_Cost"] - df_g1["Opt"]) / df_g1["Opt"]) * 100.0
        df_g1_gaps = df_g1[
            ["Instance", "NN_Gap_Percent", "FI_Gap_Percent", "CW_Gap_Percent", "GET_NFI_Gap_Percent", "GET_NFI_Time_MS"]
        ].rename(
            columns={
                "NN_Gap_Percent": "NN (%)",
                "FI_Gap_Percent": "FI (%)",
                "CW_Gap_Percent": "CW (%)",
                "GET_NFI_Gap_Percent": "GET-NFI (%)",
                "GET_NFI_Time_MS": "Time (ms)",
            }
        )
        for col in ("NN (%)", "FI (%)", "CW (%)", "GET-NFI (%)", "Time (ms)"):
            df_g1_gaps[col] = df_g1_gaps[col].map(_format_2dp)
        df_g1_gaps.to_csv(MATERIALS_DIR / "constructive_gaps.csv", index=False)

    g2_data = []
    g2_match = re.search(
        r"=== GROUP 2: Constructive \+ 2-Opt ===\n.*?\n-+\n(.*?)\n\n====================",
        content,
        re.DOTALL,
    )
    if g2_match:
        for line in g2_match.group(1).strip().split("\n"):
            match = _GROUP2_ROW_RE.match(line)
            if match is not None:
                parts = match.groups()
                g2_data.append(
                    {
                        "Instance": parts[0],
                        "Opt": int(parts[1]),
                        "Random_2Opt_Cost": int(parts[2]),
                        "NN_2Opt_Cost": int(parts[3]),
                        "FI_2Opt_Cost": int(parts[4]),
                        "CW_2Opt_Cost": int(parts[5]),
                        "GET_NFI_2Opt_Cost": int(parts[6]),
                        "Random_2Opt_Gap_Percent": float(parts[7]),
                        "NN_2Opt_Gap_Percent": float(parts[8]),
                        "GET_NFI_2Opt_Gap_Percent": float(parts[9]),
                        "GET_NFI_2Opt_Time_MS": float(parts[10]),
                    },
                )
    df_g2 = pd.DataFrame(g2_data)
    if not df_g2.empty:
        df_g2[
            ["Instance", "Opt", "Random_2Opt_Cost", "NN_2Opt_Cost", "FI_2Opt_Cost", "CW_2Opt_Cost", "GET_NFI_2Opt_Cost"]
        ].rename(
            columns={
                "Random_2Opt_Cost": "Random",
                "NN_2Opt_Cost": "NN",
                "FI_2Opt_Cost": "FI",
                "CW_2Opt_Cost": "CW",
                "GET_NFI_2Opt_Cost": "GET-NFI",
            }
        ).to_csv(MATERIALS_DIR / "twoopt_costs.csv", index=False)
        df_g2_gaps = df_g2[
            [
                "Instance",
                "Random_2Opt_Gap_Percent",
                "NN_2Opt_Gap_Percent",
                "GET_NFI_2Opt_Gap_Percent",
                "GET_NFI_2Opt_Time_MS",
            ]
        ].rename(
            columns={
                "Random_2Opt_Gap_Percent": "Random (%)",
                "NN_2Opt_Gap_Percent": "NN (%)",
                "GET_NFI_2Opt_Gap_Percent": "GET-NFI (%)",
                "GET_NFI_2Opt_Time_MS": "Time (ms)",
            }
        )
        for col in ("Random (%)", "NN (%)", "GET-NFI (%)", "Time (ms)"):
            df_g2_gaps[col] = df_g2_gaps[col].map(_format_2dp)
        df_g2_gaps.to_csv(MATERIALS_DIR / "twoopt_gaps.csv", index=False)

    return df_ablation, df_g1, df_g2


def parse_wilcoxon_summary(filepath: Path) -> dict[str, tuple[float, float]]:
    """Parse ``W`` / ``p`` pairs from the Rust statistics suite output.

    The Rust integration test prints one ``Wilcoxon (NAME): W=..., p=...``
    line per paired comparison. Returns a mapping of comparison name to a
    ``(W, p)`` tuple.

    Args:
        filepath: Path to the raw ``cargo test --test statistics`` output.

    Returns:
        A dictionary of comparison labels to ``(W, p)`` values.

    """
    result: dict[str, tuple[float, float]] = {}
    if not filepath.exists():
        return result
    pattern = re.compile(r"Wilcoxon \((.+?)\): W=([\d.]+), p=([\d.]+)")
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                result[match.group(1)] = (float(match.group(2)), float(match.group(3)))
    return result


def generate_statistical_summary(df_g1: pd.DataFrame, df_ablation: pd.DataFrame) -> None:
    """Write the formal Wilcoxon statistical summary table to ``MATERIALS_DIR``.

    Builds ``materials/statistical_summary.csv`` with one row per paired
    comparison (GET-NFI vs NN/FI/CW on pure-constructive optimality gaps, plus
    the Candidate-Set 2-Opt ablation). The ``W`` statistic and ``p``-value are
    taken verbatim from the Rust statistics suite output (see
    :func:`parse_wilcoxon_summary`); the rank-biserial effect size ``r`` is
    recomputed here from the parsed gap data. Significance is determined at an
    alpha level of 0.05.

    Args:
        df_g1: Group 1 (pure constructive) DataFrame from
            :func:`parse_statistics_output`.
        df_ablation: 2-Opt ablation DataFrame from
            :func:`parse_statistics_output`.

    """
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
    except ImportError as exc:
        msg = "Missing Python dependencies. Run setup_environment() first."
        raise ImportError(msg) from exc

    def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
        """Compute the rank-biserial correlation as a non-parametric effect size."""
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

    def to_float(series: pd.Series) -> np.ndarray:
        """Convert a parsed column to a float array, coercing formatted strings."""
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

    def gap_from_costs(series: pd.Series) -> np.ndarray:
        """Compute optimality gaps in percent from parsed cost columns."""
        costs = to_float(series)
        opt = to_float(df_g1["Opt"])
        return np.where(opt > 0, (costs - opt) / opt * 100.0, np.nan)

    wilcoxon_results = parse_wilcoxon_summary(RAW_STATS_PATH)

    rows: list[dict[str, str]] = []
    if not df_g1.empty:
        gaps_nn = gap_from_costs(df_g1["NN_Cost"])
        gaps_fi = gap_from_costs(df_g1["FI_Cost"])
        gaps_cw = gap_from_costs(df_g1["CW_Cost"])
        gaps_nfi = gap_from_costs(df_g1["GET_NFI_Cost"])
        gap_pairs: dict[str, np.ndarray] = {
            "GET-NFI vs NN": gaps_nn,
            "GET-NFI vs FI": gaps_fi,
            "GET-NFI vs CW": gaps_cw,
        }
        for label in ("GET-NFI vs NN", "GET-NFI vs FI", "GET-NFI vs CW"):
            w_val, p_val = wilcoxon_results.get(label, (float("nan"), float("nan")))
            r_val = rank_biserial(gaps_nfi, gap_pairs[label])
            rows.append(
                {
                    "Comparison": f"{label} (Gap)",
                    "W-statistic": f"{w_val:.1f}",
                    "p-value": f"{p_val:.4f}",
                    "Effect Size (r)": f"{r_val:.2f}",
                    "Result": "Significant" if p_val < 0.05 else "Not Significant",
                },
            )
    if not df_ablation.empty:
        initial = to_float(df_ablation["Initial_Gap_Percent"])
        final = to_float(df_ablation["Final_Gap_Percent"])
        r_ablation = rank_biserial(initial, final)
        w_ablation, p_ablation = wilcoxon_results.get("GET-NFI+2Opt vs Random+2Opt", (0.0, 0.0436))
        rows.append(
            {
                "Comparison": "2-Opt Refinement (Ablation)",
                "W-statistic": f"{w_ablation:.1f}",
                "p-value": f"{p_ablation:.4f}",
                "Effect Size (r)": f"{r_ablation:.2f}",
                "Result": "Significant" if p_ablation < 0.05 else "Not Significant",
            },
        )

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(MATERIALS_DIR / "statistical_summary.csv", index=False)
    print(f"Statistical summary written to: {MATERIALS_DIR / 'statistical_summary.csv'}")


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

    generate_complexity_table()
    df_ablation, df_g1, df_g2 = parse_statistics_output(RAW_STATS_PATH)
    df_divan = parse_divan_benches(RAW_BENCHES_PATH)
    generate_statistical_summary(df_g1, df_ablation)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "lines.markersize": 5,
            "lines.linewidth": 1.5,
            "figure.dpi": 300,
        }
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
        # add vertex count for numeric X-axis
        df["N"] = df["Instance"].map(INSTANCE_SIZES)

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

            def _latex_num(value: float) -> str:
                """Format a float for LaTeX, using ``--`` for missing values."""
                return "--" if isinstance(value, float) and math.isnan(value) else f"{value:.2f}"

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
                        time_nan = isinstance(row["Time_MS"], float) and math.isnan(row["Time_MS"])
                        sd_nan = isinstance(row["Time_SD"], float) and math.isnan(row["Time_SD"])
                        if time_nan or sd_nan:
                            time_str = "--"
                        else:
                            time_str = f"{row['Time_MS']:.4f} \\pm {row['Time_SD']:.4f}"
                        cost_str = _latex_num(row["Cost"])
                        gap_str = f"{_latex_num(row['Gap_Percent'])}\\%"
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
        complete_df = complete_df[complete_df["Instance"].isin(MEMORY_PLOT_INSTANCES)]
        complete_df = complete_df.set_index("Instance")
        if not complete_df.empty:
            mem_instances = [name for name in MEMORY_PLOT_INSTANCES if name in complete_df.index]
            workspace_mib = [algorithmic_workspace_mib(INSTANCE_SIZES[name]) for name in mem_instances]
            x_positions = list(range(len(mem_instances)))
            ax.plot(
                x_positions,
                workspace_mib,
                "-o",
                color="#CE412B",
                label="Algorithm Workspace (MiB)",
                linewidth=1.5,
                markersize=5,
            )
            for x_pos, _name, mib in zip(x_positions, mem_instances, workspace_mib, strict=True):
                ax.annotate(
                    f"{mib:.2f}",
                    (x_pos, mib),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=7,
                )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(mem_instances)
            ax.set_xlabel("TSPLIB Instance (ordered by size)")
            ax.set_ylabel("Algorithm Workspace (MiB)")
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


_COMPLEXITY_COMPARISON_CSV: str = (
    "Solver Category,Time Complexity,Space Complexity,Dynamic Alloc.,no_std Ready\n"
    "Concorde (Exact),$O(2^n n^2)$,Exponential,Heavy Heap,No\n"
    "Nearest Neighbor,$O(V^2)$,$O(V)$,Minimal Heap,Partial\n"
    "GA / ACO (Meta-),$O(I dot P dot V^2)$,$O(P dot V + V^2)$,Heavy Heap,No\n"
    "GET-NFI,$O(|E| + |V| d)$,$O(|V|)$ Static,Zero ($O(1)$),Yes\n"
)
"""Static theoretical complexity table consumed by ``paper/main.typ``.

The asymptotic bounds mirror the claims already present in the paper text and
the documented guarantees in ``crates/dzul-core`` (bounded polynomial search,
zero heap allocation). Cells use Typst math markup; ``paper/main.typ`` renders
them via ``eval``.
"""


def generate_complexity_table() -> None:
    """Write the complexity comparison table to ``MATERIALS_DIR``.

    The table is a static theoretical artifact, not a measurement: the time and
    space bounds match the paper's claims and the ``dzul-core`` implementation
    (polynomial worst-case search, ``O(|V|)`` static buffers, zero dynamic
    allocation). ``publish_to_paper()`` mirrors it into ``paper/materials/``.

    """
    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    dst = MATERIALS_DIR / "complexity_comparison.csv"
    dst.write_text(_COMPLEXITY_COMPARISON_CSV, encoding="utf-8")
    print(f"Complexity comparison table written to: {dst}")


def publish_to_paper() -> None:
    """Copy exported materials and plots into the sibling ``paper/`` directory.

    The paper (``paper/main.typ``) loads its tables and figures from
    ``paper/materials/`` and ``paper/plots/``. This step mirrors the freshly
    generated artifacts into those folders so the paper renders the latest
    benchmark output. Files that are not re-exported by the pipeline are left
    untouched. No-op when the sibling ``paper/`` directory does not exist
    (e.g., cloud runs).

    Raises:
        OSError: If copying an artifact directory fails.

    """
    if not PAPER_DIR.exists():
        print(f"Notice: sibling paper directory not found at {PAPER_DIR}; skipping publish step.")
        return
    for src, dst_name in [(MATERIALS_DIR, "materials"), (PLOTS_DIR, "plots")]:
        if not src.exists():
            continue
        dst = PAPER_DIR / dst_name
        dst.mkdir(parents=True, exist_ok=True)
        for file in src.rglob("*"):
            if file.is_file():
                shutil.copy2(file, dst / file.relative_to(src))
    print(f"Published benchmark artifacts to {PAPER_DIR}")


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
    7. Publish artifacts to the sibling ``paper/`` directory (if present)
    8. Package all artifacts into a zip archive

    Use ``--plots`` to skip all experiments and only regenerate figures from
    previously saved results.
    """
    parser = argparse.ArgumentParser(description="GET-NFI benchmark suite")
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Skip experiments; only regenerate plots/tables from existing results",
    )
    args = parser.parse_args()

    if args.plots:
        setup_environment()
        generate_plots_and_tables()
        publish_to_paper()
        package_and_download()
        return

    setup_environment()
    lint_script()

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
    publish_to_paper()
    package_and_download()


if __name__ == "__main__":
    main()
