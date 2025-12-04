import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Loss Functions for Zeta Block ---------- #

class ZetaLosses(nn.Module):
    def __init__(self, lambda_subm=1.0, eta_zero=1.0):
        super().__init__()
        self.lambda_subm = lambda_subm
        self.eta_zero = eta_zero

    def task_loss(self, logits, targets):
        """
        Standard cross-entropy task loss.
        logits: (B,N,C)
        targets: (B,N) with class indices
        """
        return F.cross_entropy(logits.transpose(1, 2), targets)

    def zero_set_loss(self, f_out, mask=None):
        """
        Enforces f(h) ≈ 0 on 'axiomatic' states.
        f_out: (B,N,D) = combined Gaussian+Poisson output
        mask: (B,N) boolean, True where constraint applies
        """
        if mask is None:
            # Apply everywhere
            return (f_out ** 2).mean()
        else:
            masked = f_out[mask]
            return (masked ** 2).mean() if masked.numel() > 0 else torch.tensor(0.0, device=f_out.device)


    def forward(self, logits, targets, f_out, tau, sigma, mask=None):
        L_task = self.task_loss(logits, targets)
        L_zero = self.zero_set_loss(f_out, mask)
        return (L_task +
                self.eta_zero * L_zero,
                {"task": L_task.item(),
                 "zero": L_zero.item()})
