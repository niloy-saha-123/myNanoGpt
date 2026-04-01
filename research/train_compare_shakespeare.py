#!/usr/bin/env python3
"""
Full Tiny Shakespeare training: PyTorch (`model.GPT`) vs JAX (`jax_lm`).

This script is meant to answer the fairness question first:
are the two framework runs actually starting from the same model, seeing the
same batches, and using the same optimizer settings?

Defaults therefore favor a controlled comparison:
  - same data split
  - same precomputed batch order
  - shared initial weights (PyTorch init exported into JAX)
  - explicit AdamW hyperparameters
  - dropout off by default, because different dropout RNG streams make
    "same training run" claims much weaker

Usage (from `myNanoGpt/research`, `comp560` env):

  python train_compare_shakespeare.py --data ../input.txt
  python train_compare_shakespeare.py --data ../input.txt --max-iters 500
  python train_compare_shakespeare.py --dropout 0.2 --weight-decay 0.01
  python train_compare_shakespeare.py --no-shared-init    # deliberately looser comparison
"""

from __future__ import annotations

import argparse
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


def load_encode(path: Path) -> tuple[np.ndarray, int]:
    if not path.exists():
        print(f"Missing: {path}")
        sys.exit(1)
    text = path.read_text(encoding="utf-8")
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    return data, len(chars)


def make_batches(
    data: np.ndarray,
    n_batches: int,
    batch_size: int,
    block_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def build_shared_torch_state(
    *,
    vocab_size: int,
    block_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    dropout: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """
    Build one canonical PyTorch initialization and reuse it for both frameworks.

    Using the same initial weights removes the largest hidden source of mismatch
    in PT-vs-JAX training comparisons.
    """
    torch.manual_seed(seed)
    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


@torch.no_grad()
def torch_estimate_loss(
    model: GPT,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: torch.device,
    eval_iters: int,
) -> tuple[float, float]:
    model.eval()
    tl = torch.zeros(eval_iters, device=device)
    vl = torch.zeros(eval_iters, device=device)
    for k in range(eval_iters):
        x = torch.from_numpy(train_x[k]).to(device=device, dtype=torch.long)
        y = torch.from_numpy(train_y[k]).to(device=device, dtype=torch.long)
        _, loss = model(x, y)
        tl[k] = loss
    for k in range(eval_iters):
        x = torch.from_numpy(val_x[k]).to(device=device, dtype=torch.long)
        y = torch.from_numpy(val_y[k]).to(device=device, dtype=torch.long)
        _, loss = model(x, y)
        vl[k] = loss
    model.train()
    return float(tl.mean().item()), float(vl.mean().item())


def run_pytorch(
    *,
    train_xs: np.ndarray,
    train_ys: np.ndarray,
    train_eval_x: np.ndarray,
    train_eval_y: np.ndarray,
    val_eval_x: np.ndarray,
    val_eval_y: np.ndarray,
    vocab_size: int,
    batch_size: int,
    block_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    dropout: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    seed: int,
    max_iters: int,
    eval_interval: int,
    eval_iters: int,
    initial_state_dict: dict[str, torch.Tensor] | None,
) -> dict:
    device = (
        torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if initial_state_dict is None:
        torch.manual_seed(seed)
    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )

    init_train, init_val = torch_estimate_loss(
        model, train_eval_x, train_eval_y, val_eval_x, val_eval_y, device, eval_iters
    )

    t0 = time.perf_counter()
    last_train, last_val = float("nan"), float("nan")

    for it in range(max_iters):
        x = torch.from_numpy(train_xs[it]).to(device=device, dtype=torch.long)
        y = torch.from_numpy(train_ys[it]).to(device=device, dtype=torch.long)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()

        if it % eval_interval == 0 or it == max_iters - 1:
            tr, va = torch_estimate_loss(
                model, train_eval_x, train_eval_y, val_eval_x, val_eval_y, device, eval_iters
            )
            last_train, last_val = tr, va
            elapsed = time.perf_counter() - t0
            print(f"  [pt] step {it:5d} | train {tr:.4f} | val {va:.4f} | elapsed {elapsed:.1f}s", flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

    wall = time.perf_counter() - t0
    return {
        "device": str(device),
        "wall_s": wall,
        "initial_train": init_train,
        "initial_val": init_val,
        "final_train": last_train,
        "final_val": last_val,
    }


def run_jax(
    *,
    train_xs: np.ndarray,
    train_ys: np.ndarray,
    train_eval_x: np.ndarray,
    train_eval_y: np.ndarray,
    val_eval_x: np.ndarray,
    val_eval_y: np.ndarray,
    vocab_size: int,
    batch_size: int,
    block_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    dropout: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    seed: int,
    max_iters: int,
    eval_interval: int,
    eval_iters: int,
    initial_params: dict[str, object] | None,
) -> dict:
    import jax
    import jax.numpy as jnp
    from jax_lm import GPTConfig, init_params, loss_fn, train_step_factory

    cfg = GPTConfig(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
    )
    if initial_params is None:
        key_init = jax.random.PRNGKey(seed)
        params = init_params(key_init, cfg)
    else:
        params = initial_params
    step, tx = train_step_factory(
        cfg,
        lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )
    opt_state = tx.init(params)

    @jax.jit
    def eval_batch_loss(p, rng, xb, yb):
        return loss_fn(p, rng, xb, yb, train=False, cfg=cfg)

    def jax_mean_eval(p, key, x_stack: np.ndarray, y_stack: np.ndarray) -> float:
        s = 0.0
        k = key
        for i in range(eval_iters):
            k, sk = jax.random.split(k)
            xi = jnp.asarray(x_stack[i], dtype=jnp.int32)
            yi = jnp.asarray(y_stack[i], dtype=jnp.int32)
            li = eval_batch_loss(p, sk, xi, yi)
            jax.block_until_ready(li)
            s += float(li)
        return s / eval_iters

    init_train = jax_mean_eval(params, jax.random.PRNGKey(seed + 11), train_eval_x, train_eval_y)
    init_val = jax_mean_eval(params, jax.random.PRNGKey(seed + 12), val_eval_x, val_eval_y)

    t0 = time.perf_counter()
    last_train, last_val = float("nan"), float("nan")
    key_loop = jax.random.PRNGKey(seed)

    for it in range(max_iters):
        x = jnp.asarray(train_xs[it], dtype=jnp.int32)
        y = jnp.asarray(train_ys[it], dtype=jnp.int32)
        key_loop, k_step = jax.random.split(key_loop)
        params, opt_state, _ = step(params, opt_state, k_step, x, y)

        if it % eval_interval == 0 or it == max_iters - 1:
            key_loop, ke = jax.random.split(key_loop)
            tr = jax_mean_eval(params, ke, train_eval_x, train_eval_y)
            key_loop, ke2 = jax.random.split(key_loop)
            va = jax_mean_eval(params, ke2, val_eval_x, val_eval_y)
            last_train, last_val = tr, va
            elapsed = time.perf_counter() - t0
            print(f"  [jx] step {it:5d} | train {last_train:.4f} | val {last_val:.4f} | elapsed {elapsed:.1f}s", flush=True)

    wall = time.perf_counter() - t0
    devs = jax.devices()
    return {
        "device": str(devs[0]) if devs else "unknown",
        "wall_s": wall,
        "initial_train": init_train,
        "initial_val": init_val,
        "final_train": last_train,
        "final_val": last_val,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Full Shakespeare training: PyTorch vs JAX")
    p.add_argument("--data", type=Path, default=default_data_path())
    p.add_argument("--max-iters", type=int, default=5000, help="Optimizer steps (v2.py default 5000)")
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=200, help="Batches to average for train/val loss (v2 uses 200)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-shared-init", action="store_true")
    p.add_argument("--torch-only", action="store_true")
    p.add_argument("--jax-only", action="store_true")
    args = p.parse_args()

    data, vocab_size = load_encode(args.data)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    bs, T = args.batch_size, args.block_size
    ei = args.eval_iters
    seed = args.seed

    print("=== train_compare_shakespeare (fair cross-framework defaults) ===")
    print(f"data={args.data}  chars={vocab_size}  train_tokens={n}  val_tokens={len(val_data)}")
    print(
        f"batch={bs} block={T} layers={args.n_layer} heads={args.n_head} n_embd={args.n_embd} "
        f"dropout={args.dropout} lr={args.lr}"
    )
    print(
        f"max_iters={args.max_iters} eval_interval={args.eval_interval} eval_iters={ei} seed={seed}"
    )
    print(
        f"optimizer=AdamW beta1={args.beta1} beta2={args.beta2} eps={args.eps} "
        f"weight_decay={args.weight_decay}"
    )
    print(
        f"shared_init={'no' if args.no_shared_init else 'yes'}  "
        f"same_batch_order=yes"
    )
    if args.dropout != 0.0:
        print(
            "warning: dropout > 0 means the two runs will use different dropout masks, "
            "so they are no longer the exact same optimization path."
        )
    print("Precomputing training batches (same sequence for PT & JAX)...", flush=True)
    t_prep = time.perf_counter()
    train_xs, train_ys = make_batches(train_data, args.max_iters, bs, T, seed)
    train_eval_x, train_eval_y = make_batches(train_data, ei, bs, T, seed + 1)
    val_eval_x, val_eval_y = make_batches(val_data, ei, bs, T, seed + 2)
    print(f"  done in {time.perf_counter() - t_prep:.2f}s", flush=True)

    shared_state = None
    shared_jax_params = None
    if not args.no_shared_init:
        shared_state = build_shared_torch_state(
            vocab_size=vocab_size,
            block_size=T,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            seed=seed,
        )
        shared_param_count = sum(
            tensor.numel() for name, tensor in shared_state.items() if not name.endswith("tril")
        )
        print(f"Shared initialization built once: params={shared_param_count:,}", flush=True)
        if not args.torch_only:
            from jax_lm import GPTConfig, pytorch_state_dict_to_jax_params

            shared_cfg = GPTConfig(
                vocab_size=vocab_size,
                n_embd=args.n_embd,
                n_head=args.n_head,
                n_layer=args.n_layer,
                block_size=T,
                dropout=args.dropout,
            )
            shared_jax_params = pytorch_state_dict_to_jax_params(shared_state, shared_cfg)

    rp = None
    if not args.jax_only:
        print("\n[PyTorch]", flush=True)
        rp = run_pytorch(
            train_xs=train_xs,
            train_ys=train_ys,
            train_eval_x=train_eval_x,
            train_eval_y=train_eval_y,
            val_eval_x=val_eval_x,
            val_eval_y=val_eval_y,
            vocab_size=vocab_size,
            batch_size=bs,
            block_size=T,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            weight_decay=args.weight_decay,
            seed=seed,
            max_iters=args.max_iters,
            eval_interval=args.eval_interval,
            eval_iters=ei,
            initial_state_dict=shared_state,
        )
        print(
            f"  initial train/val={rp['initial_train']:.4f} / {rp['initial_val']:.4f}"
        )
        print(
            f"  device={rp['device']}  wall_s={rp['wall_s']:.1f}  "
            f"final train/val={rp['final_train']:.4f} / {rp['final_val']:.4f}"
        )

    rj = None
    if not args.torch_only:
        print("\n[JAX]", flush=True)
        rj = run_jax(
            train_xs=train_xs,
            train_ys=train_ys,
            train_eval_x=train_eval_x,
            train_eval_y=train_eval_y,
            val_eval_x=val_eval_x,
            val_eval_y=val_eval_y,
            vocab_size=vocab_size,
            batch_size=bs,
            block_size=T,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            weight_decay=args.weight_decay,
            seed=seed,
            max_iters=args.max_iters,
            eval_interval=args.eval_interval,
            eval_iters=ei,
            initial_params=shared_jax_params,
        )
        print(
            f"  initial train/val={rj['initial_train']:.4f} / {rj['initial_val']:.4f}"
        )
        print(
            f"  device={rj['device']}  wall_s={rj['wall_s']:.1f}  "
            f"final train/val={rj['final_train']:.4f} / {rj['final_val']:.4f}"
        )

    if rp is not None and rj is not None:
        train_gap = abs(rp["initial_train"] - rj["initial_train"])
        val_gap = abs(rp["initial_val"] - rj["initial_val"])
        print("\n[Fairness check]")
        print(f"  initial train loss gap: {train_gap:.6e}")
        print(f"  initial val loss gap:   {val_gap:.6e}")
        if not args.no_shared_init and args.dropout == 0.0:
            print("  expectation: these gaps should be near zero; otherwise the setups still differ.")
        print(f"  JAX/PyTorch wall-time speedup: {rp['wall_s'] / rj['wall_s']:.2f}x")

    print("\nDone. Compare wall_s only when both use comparable hardware (e.g. both GPU).")


if __name__ == "__main__":
    main()
