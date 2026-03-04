"""BatchRenorm normalization layer for tiling-artifact-free inference.

Standard BatchNorm causes train/eval disparity when used with sliding-window
inference on large images: running-average statistics accumulated during
training diverge from actual batch statistics, especially with small batch
sizes typical in microscopy segmentation.

BatchRenorm (Ioffe, 2017) fixes this by using running-average statistics
consistently in both training and inference, with learnable correction
parameters (r, d) that gradually transition from BatchNorm-like behavior
to full global statistics during training.

Reference:
    Ioffe, "Batch Renormalization: Towards Reducing Minibatch Dependence
    in Batch-Normalized Models", NeurIPS 2017.

    Buglakova et al., "Tiling artifacts and trade-offs of feature
    normalization in the segmentation of large biological images",
    ICCV 2025. arXiv:2503.19545

Based on: https://github.com/ludvb/batchrenorm
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BatchRenorm2d(nn.Module):
    """Batch Renormalization for 2D inputs (4D tensor: N, C, H, W).

    During training, uses batch statistics but clips the correction factors
    r and d to prevent instability. During eval, uses running statistics
    (same as BatchNorm eval mode), so train and eval behave consistently
    once the running stats have converged.

    The key difference from BatchNorm: during training, the output is
    computed as::

        r = clip(std_batch / std_running, 1/rmax, rmax)
        d = clip((mean_batch - mean_running) / std_running, -dmax, dmax)
        x_hat = (x - mean_batch) / std_batch * r + d
        y = gamma * x_hat + beta

    When rmax=1 and dmax=0, this is equivalent to standard BatchNorm.

    Args:
        num_features: Number of channels (C dimension).
        eps: Small constant for numerical stability.
        momentum: Factor for running mean/var update (default 0.01).
        affine: If True, learnable gamma/beta parameters.
        rmax: Maximum allowed ratio of batch std to running std.
        dmax: Maximum allowed shift between batch mean and running mean.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.01,
        affine: bool = True,
        rmax: float = 3.0,
        dmax: float = 5.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # Clipping bounds for r and d correction factors
        self.register_buffer("rmax", torch.tensor(rmax))
        self.register_buffer("dmax", torch.tensor(dmax))

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # Running statistics (same as BatchNorm)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (N, C, H, W), got {x.dim()}D"
            )

        if self.training:
            # Compute batch statistics over (N, H, W) dimensions
            dims = (0, 2, 3)
            batch_mean = x.mean(dims)
            batch_var = x.var(dims, unbiased=False)
            batch_std = (batch_var + self.eps).sqrt()

            running_std = (self.running_var + self.eps).sqrt()

            # Compute correction factors with clipping
            r = (batch_std.detach() / running_std).clamp(
                1.0 / self.rmax.item(), self.rmax.item()
            )
            d = ((batch_mean.detach() - self.running_mean) / running_std).clamp(
                -self.dmax.item(), self.dmax.item()
            )

            # Normalize using batch stats, then apply correction
            x_hat = (x - batch_mean[None, :, None, None]) / batch_std[None, :, None, None]
            x_hat = x_hat * r[None, :, None, None] + d[None, :, None, None]

            # Update running statistics
            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * batch_mean.detach()
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * batch_var.detach()
            )
            self.num_batches_tracked += 1
        else:
            # Eval mode: use running statistics (identical to BatchNorm eval)
            running_std = (self.running_var + self.eps).sqrt()
            x_hat = (x - self.running_mean[None, :, None, None]) / running_std[None, :, None, None]

        # Apply affine transformation
        if self.affine:
            x_hat = x_hat * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return x_hat

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, rmax={self.rmax.item()}, dmax={self.dmax.item()}"
        )


def replace_bn_with_batchrenorm(model: nn.Module) -> nn.Module:
    """Replace all BatchNorm2d layers in a model with BatchRenorm2d.

    Copies the running statistics and affine parameters from the original
    BatchNorm layers so that pretrained weights are preserved.

    Args:
        model: PyTorch model (modified in-place).

    Returns:
        The same model with BatchNorm2d layers replaced.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # Create BatchRenorm2d with matching configuration
            br = BatchRenorm2d(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum if module.momentum is not None else 0.01,
                affine=module.affine,
            )

            # Copy running statistics
            if module.running_mean is not None:
                br.running_mean.copy_(module.running_mean)
            if module.running_var is not None:
                br.running_var.copy_(module.running_var)
            if module.num_batches_tracked is not None:
                br.num_batches_tracked.copy_(module.num_batches_tracked)

            # Copy affine parameters
            if module.affine:
                br.weight.data.copy_(module.weight.data)
                br.bias.data.copy_(module.bias.data)

            # Replace the module in the parent
            _set_module(model, name, br)
            count += 1

    if count > 0:
        logger.info(f"Replaced {count} BatchNorm2d layers with BatchRenorm2d")
    else:
        logger.warning("No BatchNorm2d layers found to replace")

    return model


def set_batchrenorm_limits(model: nn.Module, rmax: float, dmax: float):
    """Update rmax/dmax clipping bounds on all BatchRenorm2d layers.

    Used during warmup to gradually relax the clipping from strict
    BatchNorm-equivalent (rmax=1, dmax=0) to full BatchRenorm (rmax=3, dmax=5).

    Args:
        model: Model containing BatchRenorm2d layers.
        rmax: New maximum for r correction factor.
        dmax: New maximum for d correction factor.
    """
    for module in model.modules():
        if isinstance(module, BatchRenorm2d):
            module.rmax.fill_(rmax)
            module.dmax.fill_(dmax)


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a named submodule in a model.

    Handles dotted names like 'encoder.layer1.0.bn1' by walking
    the module hierarchy.
    """
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    setattr(parent, parts[-1], new_module)
