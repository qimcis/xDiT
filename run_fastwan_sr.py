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


def _add_suffix(path: str, suffix: str) -> str:
    """Append a suffix right before the file extension."""
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def _ensure_iterable(value: Any) -> Iterable[Any]:
    """Wrap singleton outputs (like PIL.Image) into a sequence."""
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _has_media(data: Dict[str, Any], key: str) -> bool:
    """Return True when the pipeline returned a non-empty value for key."""
    if key not in data:
        return False
    value = data[key]
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    return True


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

    if _has_media(data, "images"):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        images = _ensure_iterable(data["images"])
        for idx, image in enumerate(images):
            suffix = "" if idx == 0 else f"_{idx}"
            target_path = _add_suffix(output_path, suffix)
            if hasattr(image, "save"):
                image.save(target_path)
                print(f"Saved image to {target_path}")
            elif isinstance(image, torch.Tensor):
                tensor_path = f"{os.path.splitext(target_path)[0]}_image.pt"
                torch.save(image, tensor_path)
                print(f"Image tensor saved to {tensor_path} (convert or decode manually).")
            else:
                print("Image output type not recognised; nothing saved.")
        return

    if _has_media(data, "frames"):
        frames_list = _ensure_iterable(data["frames"])
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        for idx, frames in enumerate(frames_list):
            suffix = "" if idx == 0 else f"_{idx}"
            tensor_path = f"{os.path.splitext(_add_suffix(output_path, suffix))[0]}_frames.pt"
            torch.save(frames, tensor_path)
            print(f"Video frames tensor saved to {tensor_path} (convert to MP4 as needed).")
        return

    if _has_media(data, "videos"):
        videos = _ensure_iterable(data["videos"])
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        for idx, video in enumerate(videos):
            suffix = "" if idx == 0 else f"_{idx}"
            tensor_path = f"{os.path.splitext(_add_suffix(output_path, suffix))[0]}_video.pt"
            torch.save(video, tensor_path)
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
    parser.add_argument(
        "--save_all_ranks",
        action="store_true",
        help="Save outputs on every data-parallel rank instead of only the last group.",
    )
    parser.add_argument(
        "--offset_seed_by_rank",
        action="store_true",
        help="Add the LOCAL_RANK to the configured seed so each rank gets different noise.",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    xfuser_args = xFuserArgs.from_cli_args(cli_args)

    # Normalise prompt inputs for pipelines that expect strings.
    xfuser_args.prompt = _normalize_prompt(xfuser_args.prompt)
    xfuser_args.negative_prompt = _normalize_prompt(xfuser_args.negative_prompt)

    if cli_args.offset_seed_by_rank:
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        except ValueError:
            local_rank = 0
        xfuser_args.seed = xfuser_args.seed + local_rank

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

    should_save = cli_args.save_all_ranks or pipeline.is_dp_last_group()
    if not should_save:
        return

    output_path = cli_args.output_path
    if cli_args.save_all_ranks:
        rank_token = (
            os.environ.get("LOCAL_RANK")
            or os.environ.get("RANK")
            or os.environ.get("SLURM_PROCID")
            or "0"
        )
        output_path = _add_suffix(output_path, f"_rank{rank_token}")

    _save_output(result, output_path)


if __name__ == "__main__":
    main()
