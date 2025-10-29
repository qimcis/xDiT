import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers import DiffusionPipeline

from xfuser.config import EngineConfig, InputConfig
from xfuser.logger import init_logger
from xfuser.model_executor.pipelines import xFuserPipelineBaseWrapper
from .register import xFuserPipelineWrapperRegister

logger = init_logger(__name__)

try:  # FastWan diffusers pipeline was introduced after diffusers 0.30
    from diffusers import WanT2VPipeline  # type: ignore
except Exception:  # pragma: no cover - if missing we fall back to base type
    WanT2VPipeline = None  # type: ignore


@dataclass
class SketchRenderConfig:
    rendering_model_path: str
    threshold: float = 0.01
    min_switch_step: int = 5
    max_switch_step: int = 30
    render_offload: bool = False
    lazy_move_render: bool = True


def _gather_mean(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value


class _SketchRenderTransformer(nn.Module):
    """
    Wrapper around two DiT backbones that handles the sketching->rendering switch
    based on the self-diff heuristic from SRDiffusion.
    """

    def __init__(
        self,
        sketch_model: nn.Module,
        render_model: nn.Module,
        config: SketchRenderConfig,
    ):
        super().__init__()
        self.sketch_model = sketch_model
        self.render_model = render_model
        self.config = config

        self.active_model: nn.Module = self.sketch_model
        self.switched = False
        self.step_idx = 0
        self.prev_output: Optional[torch.Tensor] = None
        self.prev_diff: Optional[torch.Tensor] = None

        # expose common attributes used by diffusers pipelines
        for attr in ["config", "dtype", "device"]:
            if hasattr(self.sketch_model, attr):
                setattr(self, attr, getattr(self.sketch_model, attr))

    def reset_state(self):
        self.active_model = self.sketch_model
        self.switched = False
        self.step_idx = 0
        self.prev_output = None
        self.prev_diff = None

    def to(self, *args, **kwargs):  # noqa: D401 (follow torch semantics)
        self.sketch_model.to(*args, **kwargs)
        if not self.config.lazy_move_render or self.switched:
            self.render_model.to(*args, **kwargs)
        super().to(*args, **kwargs)
        return self

    def _compute_self_diff(self, current: torch.Tensor) -> Optional[torch.Tensor]:
        if self.prev_output is None:
            return None

        denom = torch.clamp(self.prev_output.abs().mean(), min=1e-6)
        diff = torch.tanh((current - self.prev_output).abs().mean() / denom)
        diff = _gather_mean(diff.unsqueeze(0)).squeeze(0)

        if self.prev_diff is None:
            self.prev_diff = diff
            return None

        self_diff = self.prev_diff - diff
        self.prev_diff = diff
        return self_diff

    def _switch_to_rendering(self, device: torch.device):
        if self.switched:
            return
        logger.info("Switching to FastWan rendering model at step %d", self.step_idx)

        if self.config.lazy_move_render:
            self.render_model.to(device)
        self.active_model = self.render_model
        self.switched = True

        if self.config.render_offload:
            self.sketch_model.to("cpu")

    def forward(self, *args, **kwargs):
        outputs = self.active_model(*args, **kwargs)
        # the DiT forward usually returns either a tensor or a tuple, we always take the tensor
        current = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        if not self.switched and self.step_idx >= self.config.min_switch_step:
            maybe_self_diff = self._compute_self_diff(current.detach())

            if maybe_self_diff is not None:
                should_switch = (
                    0 < maybe_self_diff.item() < self.config.threshold
                    or self.step_idx >= self.config.max_switch_step
                )
                if should_switch:
                    device = current.device if hasattr(current, "device") else torch.device("cuda")
                    self._switch_to_rendering(device)
        else:
            # still collect diffs for early steps
            _ = self._compute_self_diff(current.detach())

        self.prev_output = current.detach()
        self.step_idx += 1
        return outputs

    def reset_activation_cache(self):
        for model in (self.sketch_model, self.render_model):
            if hasattr(model, "reset_activation_cache"):
                model.reset_activation_cache()


class xFuserFastWanSRPipeline(xFuserPipelineBaseWrapper):
    """
    xDiT wrapper for FastWan 2.1 sketch-render inference (14B -> 1.3B).
    """

    def __init__(
        self,
        pipeline: DiffusionPipeline,
        engine_config: EngineConfig,
        *,
        render_transformer: nn.Module,
        sr_config: SketchRenderConfig,
        cache_args: Optional[Dict] = None,
    ):
        self._sr_config = sr_config
        self._raw_render_transformer = render_transformer
        self._cache_args = cache_args
        super().__init__(pipeline=pipeline, engine_config=engine_config, cache_args=cache_args)
        self._init_sketch_render_models(engine_config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        *,
        rendering_model_path: str,
        self_diff_threshold: float = 0.01,
        min_switch_step: int = 5,
        max_switch_step: int = 30,
        render_offload: bool = False,
        lazy_move_render: bool = True,
        cache_args: Optional[Dict] = None,
        return_org_pipeline: bool = False,
        **kwargs,
    ):
        if rendering_model_path is None:
            raise ValueError("`rendering_model_path` must be provided for sketch-render inference.")

        pipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=engine_config.model_config.trust_remote_code,
            **kwargs,
        )
        if return_org_pipeline:
            return pipeline

        torch_dtype = kwargs.get("torch_dtype", None)

        if not os.path.isdir(rendering_model_path):
            raise ValueError(f"Rendering model path {rendering_model_path} does not exist.")

        render_transformer = pipeline.transformer.__class__.from_pretrained(
            rendering_model_path,
            subfolder="transformer",
            torch_dtype=torch_dtype,
        )

        sr_config = SketchRenderConfig(
            rendering_model_path=rendering_model_path,
            threshold=self_diff_threshold,
            min_switch_step=min_switch_step,
            max_switch_step=max_switch_step,
            render_offload=render_offload,
            lazy_move_render=lazy_move_render,
        )

        return cls(
            pipeline=pipeline,
            engine_config=engine_config,
            render_transformer=render_transformer,
            sr_config=sr_config,
            cache_args=cache_args,
        )

    def _init_sketch_render_models(self, engine_config: EngineConfig):
        sketch_model = self.module.transformer

        render_transformer = self._raw_render_transformer
        target_dtype = next(sketch_model.parameters()).dtype
        render_transformer.to(dtype=target_dtype)

        render_transformer = self._convert_transformer_backbone(
            render_transformer,
            enable_torch_compile=engine_config.runtime_config.use_torch_compile,
            enable_onediff=engine_config.runtime_config.use_onediff,
            cache_args=self._cache_args,
        )

        # restore original transformer reference overwritten by the above conversion
        self.original_transformer = sketch_model

        switcher = _SketchRenderTransformer(
            sketch_model=sketch_model,
            render_model=render_transformer,
            config=self._sr_config,
        )

        self.module.transformer = switcher
        self.switcher = switcher

    def prepare_run(
        self,
        input_config: InputConfig,
        steps: int = 4,
        sync_steps: int = 1,
    ):
        if hasattr(self.module.transformer, "reset_state"):
            self.module.transformer.reset_state()
        super().prepare_run(input_config=input_config, steps=steps, sync_steps=sync_steps)

    @property
    def switcher_state(self) -> Dict[str, Union[bool, int]]:
        return {
            "switched": getattr(self.switcher, "switched", False),
            "step_idx": getattr(self.switcher, "step_idx", 0),
        }

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        output_type: Optional[str] = "latent",
        return_dict: bool = True,
        **kwargs,
    ):
        if hasattr(self.module.transformer, "reset_state"):
            self.module.transformer.reset_state()

        output = self.module(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            num_videos_per_prompt=num_videos_per_prompt,
            output_type=output_type,
            return_dict=return_dict,
            **kwargs,
        )

        if hasattr(self.module.transformer, "reset_state"):
            self.module.transformer.reset_state()

        return output


if WanT2VPipeline is not None:
    xFuserPipelineWrapperRegister.register(WanT2VPipeline)(xFuserFastWanSRPipeline)
