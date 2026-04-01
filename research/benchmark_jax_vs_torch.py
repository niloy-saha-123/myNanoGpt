#!/usr/bin/env python3
"""
Compare wall-clock training step time: PyTorch `model.GPT` vs JAX `jax_lm` (same architecture
as research/model.py: Pre-LN blocks, final LayerNorm, lm_head).

How to run (from your terminal; long runs are OK):

  conda activate comp560
  cd /path/to/myNanoGpt/research
  pip install -r requirements_jax_benchmark.txt   # jax + optax (use env-specific pip)

  # needs ../input.txt (Tiny Shakespeare) unless you pass --data
  python benchmark_jax_vs_torch.py --preset fast
  python benchmark_jax_vs_torch.py --preset full --data ../input.txt

  # optional
  python benchmark_jax_vs_torch.py --preset full --steps 40 --warmup 5   # shorter full-model run
  python benchmark_jax_vs_torch.py --torch-compile
  PYTHONUNBUFFERED=1 python benchmark_jax_vs_torch.py --preset full   # see lines as they print

Printed timing:
  data_prep_s     — load + encode + batch construction
  wall_total_s    — entire training loop for that framework (all steps)
  first step      — step 0 (often includes compile / lazy init)
  step min/max/mean/median/stdev — over steps after warmup only
  sum timed steps — sum of per-step times after warmup (≈ timed_steps * mean)
  tokens/s        — (batch * block) / median step time

Interpretation:
  - Compare median (or mean) step time after warmup for steady-state throughput.
  - On Apple Silicon, PyTorch often uses MPS while JAX may use CPU — not an apples-to-apples
    accelerator comparison unless both use the same kind of device.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from model import GPT  # noqa: E402


def default_data_path() -> Path:
    return SCRIPT_DIR.parent / "input.txt"


def load_text(path: Path) -> str:
    if not path.exists():
        print(f"Missing data file: {path}")
        print("Place Tiny Shakespeare as myNanoGpt/input.txt or pass --data PATH")
        sys.exit(1)
    return path.read_text(encoding="utf-8")


def encode_text(text: str) -> tuple[np.ndarray, int]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    return data, len(chars)


def numpy_batches(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    n_batches: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute random batches (shared by Torch/JAX for fair timing)."""
    rng = np.random.RandomState(seed)
    max_i = len(data) - block_size
    xs = np.zeros((n_batches, batch_size, block_size), dtype=np.int64)
    ys = np.zeros((n_batches, batch_size, block_size), dtype=np.int64)
    for b in range(n_batches):
        ix = rng.randint(0, max_i, size=(batch_size,))
        for j, i in enumerate(ix):
            xs[b, j] = data[i : i + block_size]
            ys[b, j] = data[i + 1 : i + block_size + 1]
    return xs, ys


def torch_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def summarize_step_times(times: list[float]) -> dict:
    """Stats over timed steps (after warmup). Empty list -> NaNs."""
    if not times:
        return {
            "n": 0,
            "min_s": float("nan"),
            "max_s": float("nan"),
            "mean_s": float("nan"),
            "median_s": float("nan"),
            "stdev_s": float("nan"),
            "sum_s": float("nan"),
        }
    n = len(times)
    return {
        "n": n,
        "min_s": min(times),
        "max_s": max(times),
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "stdev_s": statistics.stdev(times) if n > 1 else 0.0,
        "sum_s": sum(times),
    }


def run_torch(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    vocab_size: int,
    batch_size: int,
    block_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    dropout: float,
    lr: float,
    seed: int,
    steps: int,
    warmup: int,
    torch_compile: bool,
) -> dict:
    device = (
        torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    torch.manual_seed(seed)
    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    step_fn = model
    if torch_compile:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            step_fn = model
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    times: list[float] = []
    first_step: float | None = None
    t_section0 = time.perf_counter()

    for s in range(steps):
        x = torch.from_numpy(xs[s]).to(device=device, dtype=torch.long)
        y = torch.from_numpy(ys[s]).to(device=device, dtype=torch.long)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        _, loss = step_fn(x, y)
        loss.backward()
        opt.step()
        torch_sync()
        dt = time.perf_counter() - t0
        if s == 0:
            first_step = dt
        if s >= warmup:
            times.append(dt)

    wall_total_s = time.perf_counter() - t_section0
    st = summarize_step_times(times)
    med = st["median_s"]
    return {
        "device": str(device),
        "first_step_s": first_step,
        "wall_total_s": wall_total_s,
        "warmup_steps": warmup,
        "timed_step_count": st["n"],
        "step_min_s": st["min_s"],
        "step_max_s": st["max_s"],
        "step_mean_s": st["mean_s"],
        "step_median_s": st["median_s"],
        "step_stdev_s": st["stdev_s"],
        "step_sum_s": st["sum_s"],
        "median_step_s": med,
        "tokens_per_s": (batch_size * block_size) / med if med == med and med > 0 else float("nan"),
    }


def run_jax(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    vocab_size: int,
    batch_size: int,
    block_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    dropout: float,
    lr: float,
    seed: int,
    steps: int,
    warmup: int,
) -> dict:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        print("JAX not installed. pip install -U jax optax")
        raise SystemExit(1) from e

    from jax_lm import GPTConfig, init_params, train_step_factory

    key = jax.random.PRNGKey(seed)
    cfg = GPTConfig(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
    )
    key, k_init = jax.random.split(key)
    params = init_params(k_init, cfg)
    step, tx = train_step_factory(cfg, lr)
    opt_state = tx.init(params)

    times: list[float] = []
    first_step: float | None = None
    t_section0 = time.perf_counter()

    for s in range(steps):
        x = jnp.asarray(xs[s], dtype=jnp.int32)
        y = jnp.asarray(ys[s], dtype=jnp.int32)
        key, k_step = jax.random.split(key)
        t0 = time.perf_counter()
        params, opt_state, loss = step(params, opt_state, k_step, x, y)
        jax.block_until_ready(loss)
        jax.block_until_ready(params["wte"])
        dt = time.perf_counter() - t0
        if s == 0:
            first_step = dt
        if s >= warmup:
            times.append(dt)

    wall_total_s = time.perf_counter() - t_section0
    st = summarize_step_times(times)
    med = st["median_s"]
    devs = jax.devices()
    return {
        "device": str(devs[0]) if devs else "unknown",
        "first_step_s": first_step,
        "wall_total_s": wall_total_s,
        "warmup_steps": warmup,
        "timed_step_count": st["n"],
        "step_min_s": st["min_s"],
        "step_max_s": st["max_s"],
        "step_mean_s": st["mean_s"],
        "step_median_s": st["median_s"],
        "step_stdev_s": st["stdev_s"],
        "step_sum_s": st["sum_s"],
        "median_step_s": med,
        "tokens_per_s": (batch_size * block_size) / med if med == med and med > 0 else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="JAX vs PyTorch small-GPT step benchmark")
    p.add_argument("--data", type=Path, default=default_data_path(), help="Path to corpus text")
    p.add_argument("--preset", choices=("fast", "full"), default="fast")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--n-embd", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--torch-only", action="store_true")
    p.add_argument("--jax-only", action="store_true")
    args = p.parse_args()

    if args.preset == "fast":
        batch_size = args.batch_size or 16
        block_size = args.block_size or 64
        n_layer = args.n_layer or 2
        n_head = args.n_head or 4
        n_embd = args.n_embd or 128
        dropout = args.dropout if args.dropout is not None else 0.0
        steps = args.steps or 80
        warmup = args.warmup or 5
    else:
        batch_size = args.batch_size or 32
        block_size = args.block_size or 256
        n_layer = args.n_layer or 6
        n_head = args.n_head or 6
        n_embd = args.n_embd or 384
        dropout = args.dropout if args.dropout is not None else 0.2
        steps = args.steps or 120
        warmup = args.warmup or 10

    t0 = time.perf_counter()
    text = load_text(args.data)
    data, vocab_size = encode_text(text)

    xs, ys = numpy_batches(data, batch_size, block_size, steps, args.seed)
    data_prep_s = time.perf_counter() - t0

    print("=== benchmark_jax_vs_torch ===")
    print(f"preset={args.preset}  batch={batch_size}  block={block_size}  "
          f"layers={n_layer} heads={n_head} n_embd={n_embd} dropout={dropout}")
    print(f"steps={steps} warmup={warmup} seed={args.seed}")
    print(f"data_prep_s={data_prep_s:.4f}  (load + encode + batch tensors)")

    if not args.jax_only:
        tr = run_torch(
            xs=xs,
            ys=ys,
            vocab_size=vocab_size,
            batch_size=batch_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            lr=args.lr,
            seed=args.seed,
            steps=steps,
            warmup=warmup,
            torch_compile=args.torch_compile,
        )
        print("\n[PyTorch]")
        print(f"  device:              {tr['device']}")
        print(f"  wall_total_s:        {tr['wall_total_s']:.4f}  (all {steps} steps incl. warmup)")
        print(f"  first step (s):      {tr['first_step_s']:.6f}")
        print(f"  warmup_steps:        {tr['warmup_steps']}  |  timed_steps: {tr['timed_step_count']}")
        print(f"  step time (s): min={tr['step_min_s']:.6f}  max={tr['step_max_s']:.6f}  "
              f"mean={tr['step_mean_s']:.6f}  median={tr['step_median_s']:.6f}  stdev={tr['step_stdev_s']:.6f}")
        print(f"  sum timed steps (s): {tr['step_sum_s']:.4f}")
        print(f"  tokens/s (median):   {tr['tokens_per_s']:.1f}")

    if not args.torch_only:
        jr = run_jax(
            xs=xs,
            ys=ys,
            vocab_size=vocab_size,
            batch_size=batch_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            lr=args.lr,
            seed=args.seed,
            steps=steps,
            warmup=warmup,
        )
        print("\n[JAX]")
        print(f"  device:              {jr['device']}")
        print(f"  wall_total_s:        {jr['wall_total_s']:.4f}  (all {steps} steps incl. warmup)")
        print(f"  first step (s):      {jr['first_step_s']:.6f}")
        print(f"  warmup_steps:        {jr['warmup_steps']}  |  timed_steps: {jr['timed_step_count']}")
        print(f"  step time (s): min={jr['step_min_s']:.6f}  max={jr['step_max_s']:.6f}  "
              f"mean={jr['step_mean_s']:.6f}  median={jr['step_median_s']:.6f}  stdev={jr['step_stdev_s']:.6f}")
        print(f"  sum timed steps (s): {jr['step_sum_s']:.4f}")
        print(f"  tokens/s (median):   {jr['tokens_per_s']:.1f}")

    print("\nNote: JAX first step usually includes XLA compile; compare median/mean step for steady-state.")
    print("      If PyTorch uses MPS/CUDA and JAX uses CPU, results are not a fair accelerator comparison.")


if __name__ == "__main__":
    main()
