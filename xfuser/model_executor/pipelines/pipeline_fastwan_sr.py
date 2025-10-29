import os
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers import DiffusionPipeline

from xfuser.config import EngineConfig, InputConfig
try:
    from accelerate.utils import set_module_tensor_to_device
    from accelerate.utils.modeling import load_checkpoint_in_model
    from accelerate.hooks import remove_hook_from_module
except ImportError:  # pragma: no cover - accelerate is expected in runtime but guard anyway
    set_module_tensor_to_device = None  # type: ignore[assignment]
    load_checkpoint_in_model = None  # type: ignore[assignment]
    remove_hook_from_module = None  # type: ignore[assignment]

from xfuser.core.distributed import (
    get_dit_group,
    get_dit_world_size,
    get_pipeline_parallel_world_size,
    get_runtime_state,
    model_parallel_is_initialized,
)
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
    group = dist.group.WORLD
    world_size = dist.get_world_size()
    if model_parallel_is_initialized():
        try:
            group = get_dit_group()
            world_size = get_dit_world_size()
        except AssertionError:
            group = dist.group.WORLD
            world_size = dist.get_world_size()
    dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
    value /= world_size
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
        *,
        default_dtype: Optional[torch.dtype] = None,
        sketch_checkpoint_path: Optional[str] = None,
        render_checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self.sketch_model = sketch_model
        self.render_model = render_model
        self.sr_config = config
        self._default_dtype = default_dtype or torch.float32
        self._sketch_checkpoint_path = sketch_checkpoint_path
        self._render_checkpoint_path = render_checkpoint_path
        self._rehydrated_modules: set[int] = set()

        self.active_model: nn.Module = self.sketch_model
        self.switched = False
        self.step_idx = 0
        self.prev_output: Optional[torch.Tensor] = None
        self.prev_diff: Optional[torch.Tensor] = None
        self._last_device: Optional[torch.device] = None

        # expose common attributes used by diffusers pipelines
        for attr in ["config", "dtype", "device"]:
            if hasattr(self.sketch_model, attr):
                setattr(self, attr, getattr(self.sketch_model, attr))

        if not hasattr(self, "dtype") or getattr(self, "dtype", None) is None:
            self.dtype = self._default_dtype

        self._materialize_meta_tensors(self.sketch_model)
        self._materialize_meta_tensors(self.render_model)

    def reset_state(self):
        if self.sr_config.render_offload:
            device = self._last_device or self._get_module_device(self.render_model)
            if device is not None and device.type != "cpu":
                self.sketch_model.to(device)
        self.active_model = self.sketch_model
        self.switched = False
        self.step_idx = 0
        self.prev_output = None
        self.prev_diff = None

    def cache_context(self, name: str):
        """
        Mirror WanTransformer cache management across both sketch and render transformers.
        """
        sketch_ctx = getattr(self.sketch_model, "cache_context", None)
        render_ctx = getattr(self.render_model, "cache_context", None)
        if sketch_ctx is None and render_ctx is None:
            # fallback no-op context manager
            class _NullContext:
                def __enter__(self_inner):
                    return None

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _NullContext()

        stack = ExitStack()
        if sketch_ctx is not None:
            stack.enter_context(sketch_ctx(name))
        if render_ctx is not None:
            stack.enter_context(render_ctx(name))
        return stack

    def to(self, *args, **kwargs):  # noqa: D401 (follow torch semantics)
        target_device = self._extract_device_from_args(*args, **kwargs)
        target_dtype = self._extract_dtype_from_args(*args, **kwargs)

        self._materialize_meta_tensors(self.sketch_model)
        if not self.sr_config.lazy_move_render or self.switched:
            self._materialize_meta_tensors(self.render_model)

        self.sketch_model.to(*args, **kwargs)
        if not self.sr_config.lazy_move_render or self.switched:
            self.render_model.to(*args, **kwargs)
        super().to(*args, **kwargs)

        if isinstance(target_device, torch.device):
            self._last_device = target_device
        else:
            self._last_device = self._get_module_device(self.sketch_model)

        if self.sr_config.lazy_move_render and not self.switched:
            # keep rendering transformer on CPU until the switch happens
            self.render_model.to("cpu")
            self._materialize_meta_tensors(self.render_model, device=torch.device("cpu"))

        if target_dtype is not None:
            self.dtype = target_dtype

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

        if self.sr_config.lazy_move_render:
            self._materialize_meta_tensors(self.render_model, device=device)
            self.render_model.to(device)
        self.active_model = self.render_model
        self.switched = True
        self._last_device = device

        if self.sr_config.render_offload:
            self.sketch_model.to("cpu")

    def forward(self, *args, **kwargs):
        outputs = self.active_model(*args, **kwargs)
        # the DiT forward usually returns either a tensor or a tuple, we always take the tensor
        current = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        if not self.switched and self.step_idx >= self.sr_config.min_switch_step:
            maybe_self_diff = self._compute_self_diff(current.detach())

            if maybe_self_diff is not None:
                should_switch = (
                    0 < maybe_self_diff.item() < self.sr_config.threshold
                    or self.step_idx >= self.sr_config.max_switch_step
                )
                if should_switch:
                    device = current.device if hasattr(current, "device") else torch.device("cuda")
                    self._switch_to_rendering(device)
        else:
            # still collect diffs for early steps
            _ = self._compute_self_diff(current.detach())

        self.prev_output = current.detach()
        if hasattr(current, "device"):
            self._last_device = current.device
        self.step_idx += 1
        return outputs

    def reset_activation_cache(self):
        for model in (self.sketch_model, self.render_model):
            if hasattr(model, "reset_activation_cache"):
                model.reset_activation_cache()

    @staticmethod
    def _get_module_device(module: nn.Module) -> Optional[torch.device]:
        try:
            return next(module.parameters()).device
        except StopIteration:
            pass
        for buffer in module.buffers():
            return buffer.device
        return getattr(module, "device", None)

    def _ensure_rehydrated(self, module: nn.Module):
        if load_checkpoint_in_model is None:
            return

        module_id = id(module)
        if module_id in self._rehydrated_modules:
            return

        has_meta_params = any(getattr(param, "is_meta", False) for param in module.parameters())
        if not has_meta_params:
            self._rehydrated_modules.add(module_id)
            return

        checkpoint_path = self._resolve_checkpoint_path(module)
        if checkpoint_path and os.path.isdir(checkpoint_path):
            dtype = self.dtype if isinstance(self.dtype, torch.dtype) else self._default_dtype
            try:
                load_checkpoint_in_model(
                    module,
                    checkpoint_path,
                    device_map={"": "cpu"},
                    dtype=dtype,
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.warning(
                    "Failed to load checkpoint weights from %s: %s", checkpoint_path, exc
                )
        else:
            logger.debug(
                "No checkpoint path available to materialize module %s",
                module.__class__.__name__,
            )

        self._rehydrated_modules.add(module_id)

    def _resolve_checkpoint_path(self, module: nn.Module) -> Optional[str]:
        if module is self.sketch_model:
            return self._sketch_checkpoint_path
        if module is self.render_model:
            return self._render_checkpoint_path
        return None

    @staticmethod
    def _extract_device_from_args(*args, **kwargs) -> Optional[torch.device]:
        device = kwargs.get("device", None)
        if device is None:
            for arg in args:
                if isinstance(arg, torch.device):
                    device = arg
                    break
                if isinstance(arg, str):
                    try:
                        device = torch.device(arg)
                        break
                    except (RuntimeError, ValueError):
                        continue
        return device

    @staticmethod
    def _extract_dtype_from_args(*args, **kwargs) -> Optional[torch.dtype]:
        dtype = kwargs.get("dtype", None)
        if dtype is None:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        return dtype

    def _materialize_meta_tensors(self, module: nn.Module, device: Optional[torch.device] = None):
        self._ensure_rehydrated(module)

        if set_module_tensor_to_device is None:
            if any(getattr(param, "is_meta", False) for param in module.parameters()):
                logger.debug(
                    "Accelerate not available to materialize meta tensors for module %s; continuing with meta weights.",
                    module.__class__.__name__,
                )
            return
        hook = getattr(module, "_hf_hook", None)
        weights_map: Optional[Dict[str, torch.Tensor]] = None
        if hook is not None:
            weights_map = getattr(hook, "weights_map", None)

        target_device = device or torch.device("cpu")
        zero_fallback_params: List[str] = []
        zero_fallback_buffers: List[str] = []
        for name, param in module.named_parameters(recurse=True):
            if getattr(param, "is_meta", False):
                value = weights_map.get(name) if weights_map is not None else None
                if value is not None:
                    set_module_tensor_to_device(module, name, target_device, value=value)
                else:
                    fallback_tensor = torch.zeros(
                        param.shape,
                        dtype=getattr(param, "dtype", self._default_dtype),
                    )
                    set_module_tensor_to_device(module, name, target_device, value=fallback_tensor)
                    zero_fallback_params.append(name)
        for name, buffer in module.named_buffers(recurse=True):
            if getattr(buffer, "is_meta", False):
                value = weights_map.get(name) if weights_map is not None else None
                if value is None:
                    zero_fallback_buffers.append(name)
                    value = torch.zeros(
                        buffer.shape,
                        dtype=getattr(buffer, "dtype", self._default_dtype),
                    )
                set_module_tensor_to_device(module, name, target_device, value=value)

        if hook is not None and remove_hook_from_module is not None:
            remove_hook_from_module(module, recurse=True)

        remaining_meta: List[str] = [
            name for name, param in module.named_parameters(recurse=True) if getattr(param, "is_meta", False)
        ]
        remaining_meta += [
            name for name, buffer in module.named_buffers(recurse=True) if getattr(buffer, "is_meta", False)
        ]
        if remaining_meta:
            logger.debug(
                "Module %s retains %d meta tensors after materialisation: %s",
                module.__class__.__name__,
                len(remaining_meta),
                remaining_meta[:5],
            )
        if zero_fallback_params or zero_fallback_buffers:
            logger.debug(
                "Module %s initialised %d parameters and %d buffers with zeros due to missing checkpoint entries. "
                "Examples: params=%s buffers=%s",
                module.__class__.__name__,
                len(zero_fallback_params),
                len(zero_fallback_buffers),
                zero_fallback_params[:5],
                zero_fallback_buffers[:5],
            )


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

        sketch_checkpoint_dir = getattr(sketch_model, "config", None)
        if sketch_checkpoint_dir is not None:
            sketch_checkpoint_dir = getattr(sketch_model.config, "_name_or_path", None)
        if sketch_checkpoint_dir and os.path.isdir(sketch_checkpoint_dir):
            candidate = os.path.join(sketch_checkpoint_dir, "transformer")
            if os.path.isdir(candidate):
                sketch_checkpoint_dir = candidate
        else:
            sketch_checkpoint_dir = None

        render_checkpoint_dir = None
        if self._sr_config.rendering_model_path and os.path.isdir(self._sr_config.rendering_model_path):
            candidate = os.path.join(self._sr_config.rendering_model_path, "transformer")
            render_checkpoint_dir = candidate if os.path.isdir(candidate) else self._sr_config.rendering_model_path

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
            default_dtype=getattr(engine_config.runtime_config, "dtype", torch.float16),
            sketch_checkpoint_path=sketch_checkpoint_dir,
            render_checkpoint_path=render_checkpoint_dir,
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

    def use_naive_forward(self) -> bool:  # type: ignore[override]
        # WanPipeline does not support extra kwargs like resolution binning, so always route through wrapper.
        return False

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
        # WanPipeline does not support resolution binning flag; strip it if present.
        kwargs.pop("use_resolution_binning", None)

        if hasattr(self.module.transformer, "reset_state"):
            self.module.transformer.reset_state()

        height = height or getattr(self.module.transformer.config, "sample_height", 480)
        width = width or getattr(self.module.transformer.config, "sample_width", 832)
        num_frames = num_frames or getattr(self.module.transformer.config, "sample_frames", 81)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            prompt_embeds = kwargs.get("prompt_embeds")
            batch_size = prompt_embeds.shape[0] if isinstance(prompt_embeds, torch.Tensor) else 1

        if num_videos_per_prompt:
            batch_size *= num_videos_per_prompt

        get_runtime_state().set_video_input_parameters(
            height=height,
            width=width,
            num_frames=num_frames,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )

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
        )

        if hasattr(self.module.transformer, "reset_state"):
            self.module.transformer.reset_state()

        return output


if WanT2VPipeline is not None:
    xFuserPipelineWrapperRegister.register(WanT2VPipeline)(xFuserFastWanSRPipeline)
