"""Export trained SB3 PPO policy to ONNX for onnxruntime-web (PRD §6.4)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "build", _ROOT / "build" / "Release"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dll_windows import ensure_mingw_dll_dirs

ensure_mingw_dll_dirs(_ROOT)


class PolicyMeanExport(nn.Module):
    """Maps observation → deterministic action mean (continuous PPO)."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(obs)
        latent_pi, latent_vf = self.policy.mlp_extractor(features)
        return self.policy.action_net(latent_pi)


def export_to_onnx(
    model_path: Path,
    output_path: Path,
    obs_dim: int,
    opset: int = 17,
) -> None:
    model = PPO.load(str(model_path))
    model.policy.set_training_mode(False)
    model.policy.eval()
    model.policy.to("cpu")

    wrapper = PolicyMeanExport(model.policy).cpu()
    wrapper.eval()

    dummy = torch.zeros((1, obs_dim), dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        opset_version=opset,
    )
    print(f"Exported ONNX to {output_path} (obs_dim={obs_dim})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export PPO policy to ONNX")
    ap.add_argument(
        "--model",
        type=Path,
        default=_ROOT / "models" / "mars_ppo_latest.zip",
        help="SB3 PPO .zip checkpoint",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "frontend" / "public" / "mars_policy.onnx",
        help="Output .onnx path",
    )
    ap.add_argument(
        "--obs-dim",
        type=int,
        default=15 * 15 + 5,
        help="MarsRoverEnv flat observation size",
    )
    args = ap.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")

    export_to_onnx(args.model, args.out, args.obs_dim)


if __name__ == "__main__":
    main()
