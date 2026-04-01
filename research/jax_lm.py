"""
JAX implementation of the same decoder-only GPT as `model.py` (research).

Used for timing / throughput comparisons against PyTorch. Not wired into the
generalization experiments (those stay PyTorch + `model.py`).

Dependencies: pip install jax optax
  - GPU: follow https://github.com/jax-ml/jax (e.g. jax[cuda12] on Linux CUDA)
  - Apple Silicon: JAX often runs on CPU; see JAX install docs for experimental GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    n_embd: int
    n_head: int
    n_layer: int
    block_size: int
    dropout: float = 0.0


def layer_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return scale * x_norm + bias


def dropout(rng: jax.Array, x: jnp.ndarray, rate: float, train: bool) -> jnp.ndarray:
    if not train or rate <= 0.0:
        return x
    keep = 1.0 - rate
    mask = jax.random.bernoulli(rng, p=keep, shape=x.shape)
    return jnp.where(mask, x / keep, 0.0)


def linear(x: jnp.ndarray, kernel: jnp.ndarray, bias: jnp.ndarray | None) -> jnp.ndarray:
    # kernel: (in_dim, out_dim)  ->  x @ kernel + bias
    y = jnp.dot(x, kernel)
    if bias is not None:
        y = y + bias
    return y


def causal_mask(T: int) -> jnp.ndarray:
    return jnp.tril(jnp.ones((T, T), dtype=jnp.float32))


def head_forward(
    rng: jax.Array,
    x: jnp.ndarray,
    Wq: jnp.ndarray,
    Wk: jnp.ndarray,
    Wv: jnp.ndarray,
    tril_T: jnp.ndarray,
    dropout_rate: float,
    train: bool,
) -> jnp.ndarray:
    """Single attention head. Shapes: W* are (n_embd, head_size). tril_T is (T, T)."""
    hs = Wq.shape[1]
    q = jnp.dot(x, Wq)
    k = jnp.dot(x, Wk)
    v = jnp.dot(x, Wv)
    wei = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * (hs ** -0.5)
    wei = jnp.where(tril_T[None, :, :] == 0, -1e9, wei)
    wei = jax.nn.softmax(wei, axis=-1)
    rng, dr = jax.random.split(rng)
    wei = dropout(dr, wei, dropout_rate, train)
    return jnp.matmul(wei, v)


def block_forward(
    rng: jax.Array,
    x: jnp.ndarray,
    block: Dict[str, Any],
    tril_T: jnp.ndarray,
    n_head: int,
    dropout_rate: float,
    train: bool,
) -> jnp.ndarray:
    ln1s, ln1b = block["ln1_scale"], block["ln1_bias"]
    ln2s, ln2b = block["ln2_scale"], block["ln2_bias"]

    h = layer_norm(x, ln1s, ln1b)
    head_outs = []
    for i in range(n_head):
        rng, hr = jax.random.split(rng)
        ho = head_forward(
            hr,
            h,
            block["Wq"][i],
            block["Wk"][i],
            block["Wv"][i],
            tril_T,
            dropout_rate,
            train,
        )
        head_outs.append(ho)
    sa = jnp.concatenate(head_outs, axis=-1)
    rng, dr = jax.random.split(rng)
    sa = linear(sa, block["proj_kernel"], block["proj_bias"])
    sa = dropout(dr, sa, dropout_rate, train)
    x = x + sa

    h2 = layer_norm(x, ln2s, ln2b)
    rng, dr = jax.random.split(rng)
    ff = linear(h2, block["ff1_kernel"], block["ff1_bias"])
    ff = jax.nn.relu(ff)
    ff = linear(ff, block["ff2_kernel"], block["ff2_bias"])
    ff = dropout(dr, ff, dropout_rate, train)
    x = x + ff
    return x


def gpt_forward(
    rng: jax.Array,
    params: Dict[str, Any],
    idx: jnp.ndarray,
    train: bool,
    cfg: GPTConfig,
) -> jnp.ndarray:
    """Returns logits (B, T, vocab). `params` is array-only (no config nested)."""
    B, T = idx.shape
    te = params["wte"][idx]
    pos = jnp.arange(T)[None, :]
    pe = params["wpe"][pos]
    x = te + pe
    tril_T = causal_mask(T)
    for i in range(cfg.n_layer):
        rng, lr = jax.random.split(rng)
        x = block_forward(lr, x, params["blocks"][i], tril_T, cfg.n_head, cfg.dropout, train)
    x = layer_norm(x, params["lnf_scale"], params["lnf_bias"])
    logits = linear(x, params["lm_head_kernel"], params["lm_head_bias"])
    return logits


def loss_fn(
    params: Dict[str, Any],
    rng: jax.Array,
    idx: jnp.ndarray,
    targets: jnp.ndarray,
    train: bool,
    cfg: GPTConfig,
) -> jnp.ndarray:
    logits = gpt_forward(rng, params, idx, train=train, cfg=cfg)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    mask = targets != -100
    safe_targets = jnp.where(mask, targets, 0)
    token_loss = -jnp.take_along_axis(log_probs, safe_targets[..., None], axis=-1).squeeze(-1)
    return jnp.sum(token_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def init_params(key: jax.Array, cfg: GPTConfig) -> Dict[str, Any]:
    """Initialize parameters (similar scale to default PyTorch nn.Module init)."""
    keys = jax.random.split(key, 256)
    ki = 0
    scale = 0.02

    def randn(shape):
        nonlocal ki
        k = keys[ki]
        ki += 1
        return jax.random.normal(k, shape, dtype=jnp.float32) * scale

    d = cfg.n_embd
    hs = d // cfg.n_head
    T = cfg.block_size
    V = cfg.vocab_size

    wte = randn((V, d))
    wpe = randn((T, d))

    blocks = []
    for _ in range(cfg.n_layer):
        Wq = jnp.stack([randn((d, hs)) for _ in range(cfg.n_head)], axis=0)
        Wk = jnp.stack([randn((d, hs)) for _ in range(cfg.n_head)], axis=0)
        Wv = jnp.stack([randn((d, hs)) for _ in range(cfg.n_head)], axis=0)
        proj_kernel = randn((d, d))
        proj_bias = jnp.zeros((d,), dtype=jnp.float32)
        ff1_kernel = randn((d, 4 * d))
        ff1_bias = jnp.zeros((4 * d,), dtype=jnp.float32)
        ff2_kernel = randn((4 * d, d))
        ff2_bias = jnp.zeros((d,), dtype=jnp.float32)
        ln1_scale = jnp.ones((d,), dtype=jnp.float32)
        ln1_bias = jnp.zeros((d,), dtype=jnp.float32)
        ln2_scale = jnp.ones((d,), dtype=jnp.float32)
        ln2_bias = jnp.zeros((d,), dtype=jnp.float32)
        blocks.append(
            {
                "Wq": Wq,
                "Wk": Wk,
                "Wv": Wv,
                "proj_kernel": proj_kernel,
                "proj_bias": proj_bias,
                "ff1_kernel": ff1_kernel,
                "ff1_bias": ff1_bias,
                "ff2_kernel": ff2_kernel,
                "ff2_bias": ff2_bias,
                "ln1_scale": ln1_scale,
                "ln1_bias": ln1_bias,
                "ln2_scale": ln2_scale,
                "ln2_bias": ln2_bias,
            }
        )

    lnf_scale = jnp.ones((d,), dtype=jnp.float32)
    lnf_bias = jnp.zeros((d,), dtype=jnp.float32)
    lm_head_kernel = randn((d, V))
    lm_head_bias = jnp.zeros((V,), dtype=jnp.float32)

    return {
        "wte": wte,
        "wpe": wpe,
        "blocks": tuple(blocks),
        "lnf_scale": lnf_scale,
        "lnf_bias": lnf_bias,
        "lm_head_kernel": lm_head_kernel,
        "lm_head_bias": lm_head_bias,
    }


def pytorch_state_dict_to_jax_params(
    state_dict: Mapping[str, Any],
    cfg: GPTConfig,
) -> Dict[str, Any]:
    """
    Convert a PyTorch GPT `state_dict()` into the array layout used by this file.

    This lets PyTorch and JAX start from the exact same weights, which is the
    cleanest way to compare training behavior across frameworks.
    """

    def arr(name: str) -> jnp.ndarray:
        x = state_dict[name]
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
        return jnp.asarray(x, dtype=jnp.float32)

    blocks = []
    for layer in range(cfg.n_layer):
        Wq = jnp.stack(
            [arr(f"blocks.{layer}.sa.heads.{head}.query.weight").T for head in range(cfg.n_head)],
            axis=0,
        )
        Wk = jnp.stack(
            [arr(f"blocks.{layer}.sa.heads.{head}.key.weight").T for head in range(cfg.n_head)],
            axis=0,
        )
        Wv = jnp.stack(
            [arr(f"blocks.{layer}.sa.heads.{head}.value.weight").T for head in range(cfg.n_head)],
            axis=0,
        )
        blocks.append(
            {
                "Wq": Wq,
                "Wk": Wk,
                "Wv": Wv,
                "proj_kernel": arr(f"blocks.{layer}.sa.proj.weight").T,
                "proj_bias": arr(f"blocks.{layer}.sa.proj.bias"),
                "ff1_kernel": arr(f"blocks.{layer}.ffwd.net.0.weight").T,
                "ff1_bias": arr(f"blocks.{layer}.ffwd.net.0.bias"),
                "ff2_kernel": arr(f"blocks.{layer}.ffwd.net.2.weight").T,
                "ff2_bias": arr(f"blocks.{layer}.ffwd.net.2.bias"),
                "ln1_scale": arr(f"blocks.{layer}.ln1.weight"),
                "ln1_bias": arr(f"blocks.{layer}.ln1.bias"),
                "ln2_scale": arr(f"blocks.{layer}.ln2.weight"),
                "ln2_bias": arr(f"blocks.{layer}.ln2.bias"),
            }
        )

    return {
        "wte": arr("token_embedding_table.weight"),
        "wpe": arr("position_embedding_table.weight"),
        "blocks": tuple(blocks),
        "lnf_scale": arr("ln_f.weight"),
        "lnf_bias": arr("ln_f.bias"),
        "lm_head_kernel": arr("lm_head.weight").T,
        "lm_head_bias": arr("lm_head.bias"),
    }


def build_optimizer(
    learning_rate: float,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
):
    return optax.adamw(
        learning_rate,
        b1=beta1,
        b2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )


def train_step_factory(
    cfg: GPTConfig,
    learning_rate: float,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
):
    optimizer = build_optimizer(
        learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )

    def loss_train(p, r, xi, yi):
        return loss_fn(p, r, xi, yi, train=True, cfg=cfg)

    @jax.jit
    def _step(params, opt_state, rng, x, y):
        loss, grads = jax.value_and_grad(loss_train)(params, rng, x, y)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss

    return _step, optimizer


def numpy_batches(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    n_batches: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute random batches shared by Torch/JAX for fair timing."""
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
