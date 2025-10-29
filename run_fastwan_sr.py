#!/usr/bin/env python3
"""
Minimal launcher for FastWan sketch-render inference with xDiT.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable

import torch

from xfuser import xFuserFastWanSRPipeline
from xfuser.config.args import FlexibleArgumentParser, xFuserArgs


def _normalize_prompt(value: Any) -> Any:
    """
    xFuserArgs parses `--prompt`/`--negative_prompt` with nargs="*".
    Collapse single-entry lists back to strings so downstream pipelines
    get the shape they expect.
    """
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        return value
    return value


def _save_output(result: Any, output_path: str) -> None:
    """
    Best-effort saver that handles common diffusers pipeline outputs.
    For videos we log the tensor path for post-processing by the user.
    """
    if result is None:
        print("No output returned from pipeline.")
        return

    # diffusers PipelineOutput is dict-like
    if isinstance(result, dict):
        data = result
    else:
        data = getattr(result, "__dict__", {})

    if "images" in data and data["images"]:
        image = data["images"][0]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        image.save(output_path)
        print(f"Saved image to {output_path}")
        return

    if "frames" in data and data["frames"]:
        tensor_path = f"{os.path.splitext(output_path)[0]}_frames.pt"
        torch.save(data["frames"][0], tensor_path)
        print(f"Video frames tensor saved to {tensor_path} (convert to MP4 as needed).")
        return

    if "videos" in data and data["videos"]:
        tensor_path = f"{os.path.splitext(output_path)[0]}_video.pt"
        torch.save(data["videos"][0], tensor_path)
        print(f"Video tensor saved to {tensor_path} (convert to MP4 as needed).")
        return

    print("Pipeline completed, but no recognised media field was returned.")


def parse_args() -> argparse.Namespace:
    parser = FlexibleArgumentParser()
    xFuserArgs.add_cli_args(parser)
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.png",
        help="Where to save the first generated sample (image results only).",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    xfuser_args = xFuserArgs.from_cli_args(cli_args)

    # Normalise prompt inputs for pipelines that expect strings.
    xfuser_args.prompt = _normalize_prompt(xfuser_args.prompt)
    xfuser_args.negative_prompt = _normalize_prompt(xfuser_args.negative_prompt)

    engine_config, input_config = xfuser_args.create_config()

    pipeline = xFuserFastWanSRPipeline.from_pretrained(
        pretrained_model_name_or_path=xfuser_args.model,
        engine_config=engine_config,
        rendering_model_path=xfuser_args.rendering_model_path,
        self_diff_threshold=xfuser_args.sr_diff_threshold,
        min_switch_step=xfuser_args.sr_min_switch_step,
        max_switch_step=xfuser_args.sr_max_switch_step,
        render_offload=xfuser_args.sr_render_offload,
        lazy_move_render=xfuser_args.sr_lazy_move_render,
        torch_dtype=torch.float16,
    ).to("cuda")

    pipeline.prepare_run(input_config)

    result = pipeline(
        prompt=xfuser_args.prompt,
        height=xfuser_args.height,
        width=xfuser_args.width,
        num_frames=xfuser_args.num_frames,
        num_inference_steps=xfuser_args.num_inference_steps,
        guidance_scale=xfuser_args.guidance_scale,
        negative_prompt=xfuser_args.negative_prompt,
        output_type=xfuser_args.output_type,
        return_dict=True,
    )

    if pipeline.is_dp_last_group():
        _save_output(result, cli_args.output_path)


if __name__ == "__main__":
    main()
