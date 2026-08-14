"""Export a trained PPO policy to ONNX for inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO


class DeterministicPolicyWrapper(torch.nn.Module):
    """Torch wrapper that exposes deterministic PPO policy actions."""

    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.policy._predict(observation, deterministic=True)


def export_to_onnx(model_path: Path, output_path: Path, opset: int = 17) -> None:
    model = PPO.load(str(model_path), device="cpu")
    obs_shape = model.observation_space.shape
    if not obs_shape:
        raise ValueError("Model observation space shape is undefined.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = DeterministicPolicyWrapper(model.policy).eval()
    dummy_input = torch.zeros((1, *obs_shape), dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
        opset_version=opset,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a trained PPO model to ONNX")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    model_path = args.model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {model_path}")

    export_to_onnx(model_path=model_path, output_path=args.output_path.resolve(), opset=args.opset)
    print(f"Exported ONNX policy: {args.output_path.resolve()}")


if __name__ == "__main__":
    main()
