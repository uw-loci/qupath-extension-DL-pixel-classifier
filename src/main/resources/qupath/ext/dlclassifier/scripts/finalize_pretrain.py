"""
Finalize a paused pretraining run via Appose.

Loads pause_checkpoint.pt from a paused MAE/SSL pretraining run and writes
the encoder out as model.pt + a minimal metadata.json so the user can use it
without resuming.

Inputs:
    checkpoint_path: str - path to pause_checkpoint.pt
    output_dir: str - directory to write model.pt and metadata.json

Outputs:
    encoder_path: str - path to saved model.pt
    best_loss: float - best loss recorded in the checkpoint
"""
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger("dlclassifier.appose.finalize_pretrain")

# Appose 0.10.0+: inputs are injected directly into script scope.
# Required: checkpoint_path, output_dir

ckpt_path = Path(checkpoint_path)
out_dir = Path(output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

if not ckpt_path.exists():
    raise FileNotFoundError("Pause checkpoint not found: %s" % ckpt_path)

logger.info("Finalizing pretraining from %s -> %s", ckpt_path, out_dir)
ckpt = torch.load(str(ckpt_path), map_location="cpu")

# Prefer the best_state captured during training; fall back to current weights.
best_state = ckpt.get("best_state")
state = best_state if best_state is not None else ckpt.get("model_state_dict")
if state is None:
    raise RuntimeError(
        "Checkpoint has neither best_state nor model_state_dict: %s" % ckpt_path)

best_loss = float(ckpt.get("best_loss", 0.0))
epoch = int(ckpt.get("epoch", 0))
config = ckpt.get("config", {}) or {}

encoder_path = out_dir / "model.pt"
torch.save({"model_state_dict": state}, str(encoder_path))
logger.info(
    "Saved finalized encoder to %s (best_loss=%.6f, epoch=%d)",
    encoder_path, best_loss, epoch)

# Minimal metadata. The original run's metadata.json (if any) is left in place.
metadata = {
    "model_type": "pretrained_finalized",
    "run_name": config.get("run_name", ""),
    "finalized": {
        "best_loss": best_loss,
        "epoch": epoch,
        "source_checkpoint": str(ckpt_path),
    },
}
metadata_path = out_dir / "metadata_finalized.json"
with open(str(metadata_path), "w") as f:
    json.dump(metadata, f, indent=2)

task.outputs["encoder_path"] = str(encoder_path)
task.outputs["best_loss"] = best_loss
