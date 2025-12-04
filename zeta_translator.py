# Create a single-file scaffold for NEP's ζ-compatible curvature monitor.
# The file contains:
# - ZetaTranslator: computes κ_ζ from eigenvalues (scale/basis invariant)
# - FisherPullbackApprox: top-k Fisher pullback via JVPs (torch)
# - NEPMonitor: streaming detector with cheap proxy -> expensive check -> block decision
# - Example hook for HuggingFace GPT-2 (optional, guarded import)
#
# Notes:
# - Internet is disabled here; user should run in their own env with transformers installed.
# - This is a minimal, readable scaffold with explicit complexity notes.

# nep_monitor.py
# Copyright (c) 2025
# NEP ζ-Compatible Curvature Monitor — Minimal Scaffold
# Author: Sar Hamam (Noetic Eidos Project)
# License: MIT
from __future__ import annotations

import math
import torch
import numpy as np
from torch import Tensor
from mpmath import zeta, diff
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# -----------------------------
# Zeta-compatible translation
# -----------------------------

class ZetaTranslator:
    """
    Computes κ_ζ (kappa_zeta) from a set of positive eigenvalues λ_i of the pullback metric G_l.
    Scale- and basis-invariant via log-centering. Uses ζ baseline on the critical line s=1/2 by default.
    """

    def __init__(self, s: float = 0.5, dynamics_alpha=0.01):
        self.s = s
        if zeta is None or diff is None:
            raise ImportError("mpmath is required for ZetaTranslator (pip install mpmath).")

        # Cache ζ baseline second derivative at s
        self._zeta_d2 = float(diff(lambda u: math.log(abs(zeta(u))), self.s, 2))
        self.dynamics_alpha = dynamics_alpha
        self.offset = 0

    def kappa(self, eigs: np.ndarray, centered=True, return_tuple=False):
        """
        Compute κζ from eigenvalues with optional global offset calibration.

        Args:
            eigs: Positive eigenvalues of metric tensor
            centered: If True, subtract dynamic offset (default: True)
            return_tuple: If True, return (raw, calibrated, offset) tuple (default: False)

        Returns:
            If return_tuple=False: float (calibrated κζ if centered=True, else raw κζ)
            If return_tuple=True: (kappa_raw, kappa_calibrated, offset) tuple
        """
        eigs = np.asarray(eigs, dtype=float)
        eigs = np.maximum(eigs, 1e-30)  # guard
        y = np.log(eigs)
        y -= y.mean()

        w = np.exp(-self.s * y)
        ws = w.sum()
        if ws <= 0:
            if return_tuple:
                return (0.0, 0.0, self.offset)
            return 0.0

        mu = (y * w).sum() / ws
        var_tilt = ((y - mu) ** 2 * w).sum() / ws  # Φ''(s)

        kappa_raw = float(var_tilt - self._zeta_d2)

        if not centered:
            # Update offset but don't subtract (used for initialization)
            self.offset = kappa_raw
            if return_tuple:
                return (kappa_raw, kappa_raw, self.offset)
            return kappa_raw

        # Update offset via EMA (global structure smoothing, α=0.01)
        if self.offset:
            self.offset = (1 - self.dynamics_alpha) * self.offset + self.dynamics_alpha * kappa_raw
        else:
            self.offset = kappa_raw

        kappa_calibrated = kappa_raw - self.offset

        if return_tuple:
            return (kappa_raw, kappa_calibrated, self.offset)
        return kappa_calibrated


# --------------------------------
# Fisher pullback (top-k, JVP/JTM)
# --------------------------------

@dataclass
class FisherPullbackApprox:
    """
    Top-k Fisher pullback for a single layer mapping h -> z (logits).
    We avoid constructing J in full (V x d). Instead we compute J^T F J via vector-Jacobian products.
    Assumptions:
      - We can run forward to get logits z and softmax p.
      - We can compute v -> J v (JVP) and u -> J^T u (VJP) using autograd.
    Complexity (per token, per layer):
      - k target logits: O(k * d) for JVP/VJP (vectorized on GPU).
      - Low-rank eigenspectrum on r x r (r<=k) ~ O(r^3) negligible for small r.
    """
    top_k: int = 128
    eps: float = 1e-6
    device: Optional[str] = None

    def _softmax(self, z: Tensor) -> Tensor:
        return z.log_softmax(dim=-1).exp()

    def pullback_spectrum(
        self,
        h: Tensor,
        logits_fn: Callable[[Tensor], Tensor],
    ) -> np.ndarray:
        """
        Computes eigenvalues of G ≈ J^T F_topk J for a single vector h (shape: [d]).
        logits_fn: maps hidden state h -> logits z (shape: [V]).
        Returns: np.ndarray of eigenvalues (low-rank approx, size <= top_k).
        """
        if torch is None:
            raise ImportError("PyTorch required for FisherPullbackApprox.")

        assert h.dim() == 1, "h must be a 1-D tensor (hidden activations for one token at one layer)."
        h = h.detach().requires_grad_(True)
        z = logits_fn(h)  # [V]
        V = z.shape[-1]
        p = self._softmax(z)  # [V]

        # top-k indices by probability mass
        k = min(self.top_k, V)
        top_vals, top_idx = torch.topk(p, k)

        # Build an orthonormal basis U for the top-k subspace in ℝ^V as unit vectors e_i at top_idx.
        # We'll compute G = J^T F J with F restricted to top-k => F_topk = diag(p_top) - p_top p_top^T
        p_top = p[top_idx]  # [k]
        F_top = torch.diag(p_top) - p_top[:, None] * p_top[None, :]  # [k,k]

        # Compute J_top: rows are ∂z_i/∂h for i in top_idx
        # Each row can be obtained by VJP: ∂z_i/∂h = grad(z_i, h)
        J_rows = []
        for i in range(k):
            grad_vec = torch.zeros_like(z)
            grad_vec[top_idx[i]] = 1.0
            (g_h,) = torch.autograd.grad(z, h, grad_outputs=grad_vec, retain_graph=True, create_graph=False)
            J_rows.append(g_h.detach())
        J = torch.stack(J_rows, dim=0)  # [k, d]

        # G = J^T F_top J (d x d, but rank <= k). We only need its non-zero eigenvalues which equal
        # eigenvalues of the small matrix S = (F_top)^{1/2} J J^T (F_top)^{1/2}  in ℝ^{k x k}
        # Compute S efficiently:
        # A = J J^T [k,k]
        A = J @ J.t()
        # S = F_top^{1/2} A F_top^{1/2}
        # Use symmetric PSD sqrt via eig on kxk
        eigvals_F, eigvecs_F = torch.linalg.eigh(F_top + self.eps * torch.eye(k, device=F_top.device))
        sqrtF = eigvecs_F @ torch.diag(torch.clamp(eigvals_F, min=0).sqrt()) @ eigvecs_F.t()
        S = sqrtF @ A @ sqrtF.t()  # [k,k], PSD

        # Non-zero eigenvalues of G equal eigenvalues of S (see Sylvester's inertia for products).
        eigvals_S = torch.linalg.eigvalsh(S + self.eps * torch.eye(k, device=S.device))
        eigvals = eigvals_S.detach().cpu().numpy()
        # Filter small numerical noise
        eigvals = eigvals[eigvals > self.eps]
        return eigvals


# --------------------
# Streaming monitor
# --------------------

@dataclass
class EWMAStat:
    mean: float = 0.0
    mad: float = 0.0
    inited: bool = False
    alpha: float = 0.02  # slow drift; adjust per model

    def update(self, x: float) -> None:
        if not self.inited:
            self.mean = x
            self.mad = 0.0
            self.inited = True
            return
        # EWMA
        delta = x - self.mean
        self.mean += self.alpha * delta
        # Robust MAD update (approx)
        self.mad += self.alpha * (abs(delta) - self.mad)

    def threshold(self, k: float = 6.0) -> float:
        return self.mean + k * max(self.mad, 1e-8)


@dataclass
class NEPMonitor:
    zeta: ZetaTranslator
    fisher: FisherPullbackApprox
    k_sigma: float = 6.0
    cheap_rank: int = 16  # rank for cheap proxy (activation covariance approx)
    layer_stats: Dict[int, EWMAStat] = field(default_factory=dict)

    def _stat(self, layer: int) -> EWMAStat:
        if layer not in self.layer_stats:
            self.layer_stats[layer] = EWMAStat()
        return self.layer_stats[layer]

    def cheap_proxy(self, h: np.ndarray) -> float:
        """
        Cheap curvature proxy from activations only (no JVP):
          - project to rank-r via random Gaussian matrix and compute spectral spread
        Complexity: O(d * r)
        """
        d = h.shape[-1]
        r = min(self.cheap_rank, d)
        R = np.random.default_rng(0).standard_normal((d, r))
        proj = h @ R  # [r]
        # spectral spread proxy: log(var) as simple scalar
        v = np.var(proj) + 1e-12
        return float(math.log(v))

    def step(
        self,
        layer: int,
        h_torch: Tensor,
        logits_fn: Callable[[Tensor], Tensor],
        expensive_check: bool = True,
    ) -> Tuple[float, bool, Optional[float]]:
        """
        Process a single layer's hidden vector (one token).
        Returns: (cheap_score, blocked, kappa_zeta_if_computed)
        """
        # Cheap proxy
        h_np = h_torch.detach().cpu().numpy()
        cheap = self.cheap_proxy(h_np)
        stat = self._stat(layer)
        stat.update(cheap)
        blocked = False
        kappa_val = None

        if expensive_check and cheap > stat.threshold(self.k_sigma):
            # Compute Fisher pullback spectrum (top-k), then κ_ζ
            eigs = self.fisher.pullback_spectrum(h_torch, logits_fn)
            if eigs.size > 0:
                kappa_val = self.zeta.kappa(eigs)
                # Block if κ_ζ also exceeds a robust threshold (reuse same EWMA for simplicity,
                # or maintain a dedicated κ-stat per layer in production)
                if kappa_val > stat.threshold(self.k_sigma):
                    blocked = True
        return cheap, blocked, kappa_val


# ---------------------------
# Example model integration
# ---------------------------

def hook_gpt2_layer_hidden_states(model) -> List[int]:
    """
    Returns list of layer indices that expose hidden states in the forward outputs.
    For HF GPT-2, hidden_states can be enabled and layers are 0..n_layer.
    """
    return list(range(getattr(model.config, "n_layer", 12)))


class GPT2LayerLogitsAdapter:
    """
    Given a HuggingFace GPT-2 model, provide a per-layer logits_fn: h -> z (logits).
    We approximate z by routing h through the remaining blocks + lm_head.
    For performance, you may cache partial blocks or use a tiny head mapping.
    """

    def __init__(self, model, layer_index: int):
        self.model = model
        self.layer_index = layer_index
        self.n_layers = model.config.n_layer
        # basic sanity
        assert 0 <= layer_index < self.n_layers

    def __call__(self, h: Tensor) -> Tensor:
        """
        Map a single hidden vector at layer_index to logits.
        NOTE: This is a slow but portable path. In practice you should
        write a partial forward pass from this layer to output.
        """
        # h: [d] -> [1,1,d]
        h2 = h.view(1, 1, -1)
        x = h2
        # pass through remaining transformer blocks
        for j in range(self.layer_index + 1, self.n_layers):
            x = self.model.transformer.h[j](x)[0]
        # layer norm + lm_head
        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x)[0, 0, :]  # [V]
        return logits