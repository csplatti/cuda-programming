#!/usr/bin/env python3

import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


RESULT_PATTERN = re.compile(
    r"N=(?P<size>\d+)\s+"
    r"serial=(?P<serial>[0-9.eE+-]+)ms\s+"
    r"kernel=(?P<kernel>[0-9.eE+-]+)ms\s+"
    r"correct=(?P<correct>[01])"
)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    plots_dir = project_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for plot_path in plots_dir.iterdir():
        if plot_path.is_file() or plot_path.is_symlink():
            plot_path.unlink()

    completed = subprocess.run(
        ["mcr", "matmul.cu"],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )

    output = f"{completed.stdout}\n{completed.stderr}"

    if completed.returncode != 0:
        print(output)
        raise SystemExit(f"mcr failed with exit code {completed.returncode}")

    results = [match.groupdict() for match in RESULT_PATTERN.finditer(output)]

    if not results:
        print(output)
        raise SystemExit("No benchmark results were found in mcr output")

    sizes = [int(result["size"]) for result in results]
    serial_times = [float(result["serial"]) for result in results]
    kernel_times = [float(result["kernel"]) for result in results]

    incorrect = [result for result in results if result["correct"] != "1"]
    if incorrect:
        raise SystemExit("The benchmark produced an incorrect result")

    plt.plot(sizes, serial_times, marker="o", label="Serial")
    plt.plot(sizes, kernel_times, marker="o", label="CUDA kernel")
    plt.xlabel("Matrix size (N × N)")
    plt.ylabel("Runtime (ms)")
    plt.title("Serial vs CUDA Matrix Multiplication")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = plots_dir / "matmul_benchmark.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
