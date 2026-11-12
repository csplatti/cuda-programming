# CUDA Programming

## Python benchmark plotter

`plot_matmul.py` runs the matrix-multiplication benchmark and plots its timing
results.

### Dependencies

- Python 3.12 in the project's `uv` environment
- `matplotlib`, installed with `uv pip install matplotlib`
- `mcr`, an external command-line program that runs `matmul.cu`

`mcr` is not a Python package and cannot be installed from
`requirements.txt`. It must be installed separately and available on the
shell's `PATH`. Verify that the script can find it with:

```bash
command -v mcr
```

### Run

```bash
source .venv/bin/activate
python plot_matmul.py
```

On every run, the script clears the existing files in `plots/` and saves the
new graph as `plots/matmul_benchmark.png`.
