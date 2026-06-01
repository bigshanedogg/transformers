# coding=utf-8
# Copyright 2026 NAVER Cloud Corp. and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
HyperCLOVA X SEED Image Processor (Fast)

Supports two resize modes:
- Smart resize (default): adjusts image to fit within min_pixels and max_pixels,
  preserving aspect ratio. Activated when size has both 'shortest_edge' and 'longest_edge'.
- CLIP mode: standard Resize(shortest_edge) + CenterCrop, identical to open_clip's
  preprocess_val pipeline. Activated when size has only 'shortest_edge'.

Based on BaseImageProcessorFast with torchvision resize.
"""

import math
import os
import PIL
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
from torchvision.transforms.v2 import functional as F
try:
    from transformers.image_processing_utils import BatchFeature
except ImportError:
    from transformers import BatchFeature
try:
    from transformers.image_processing_backends import BaseImageProcessorFast
except ImportError:
    # transformers < v5.3.0
    from transformers.image_processing_utils_fast import BaseImageProcessorFast
try:
    from transformers.image_processing_utils_fast import DefaultFastImageProcessorKwargs
except ImportError:
    from transformers.processing_utils import ImagesKwargs as DefaultFastImageProcessorKwargs  # transformers < v5.3.0
try:
    from PIL.Image import Resampling as PILResampling
except (ImportError, AttributeError):
    # Pillow < 9.1.0
    class PILResampling:
        NEAREST = 0
        LANCZOS = 1
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
try:
    from transformers.image_utils import SizeDict
except ImportError:
    SizeDict = dict  # transformers < 4.46

# OpenAI CLIP normalization constants
# Source: transformers.image_utils.OPENAI_CLIP_MEAN / OPENAI_CLIP_STD
_OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class ResizeMode(str, Enum):
    """Resize strategy for HyperCLOVAXVisionV2ImageProcessor.

    SMART_RESIZE:
        Pixel-count-constrained resize (Qwen2-VL style). Preserves aspect ratio
        and ensures both dimensions are divisible by patch_size * merge_size.
        Requires size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}.

    CLIP:
        Standard CLIP preprocess_val pipeline. Resizes the shorter edge to
        size["shortest_edge"], then center-crops to a square of the same size.
        Requires do_center_crop=True and crop_size = {"height": N, "width": N}.
    """

    SMART_RESIZE = "smart_resize"
    CLIP = "clip"


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    """Smart resize for dynamic resolution.

    Adjusts image dimensions to satisfy:
    1. Both dimensions are divisible by factor.
    2. Total pixel count is between min_pixels and max_pixels.

    Adapted from the Qwen2.5-VL image processing implementation.
    Reference: https://github.com/QwenLM/Qwen2.5-VL (Apache 2.0 License)

    Args:
        height: Original image height.
        width: Original image width.
        factor: Rounding unit (default: 28 = patch_size * merge_size).
        min_pixels: Minimum pixel count (default: 3136).
        max_pixels: Maximum pixel count (default: 1003520).

    Returns:
        Tuple of (new_height, new_width).
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def patchify(
    patches: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> Tuple[torch.Tensor, int, int, int]:
    """Convert a spatial or spatio-temporal image tensor into patch tokens.

    Args:
        patches: Image tensor of shape (B, C, H, W) or video tensor of shape
            (B, T, C, H, W). If 4-D, a temporal dimension of size 1 is added.
        patch_size: Spatial patch size (pixels per patch side).
        temporal_patch_size: Number of frames per temporal patch.
        merge_size: Token reduction factor applied spatially.

    Returns:
        Tuple of (flatten_patches, grid_t, grid_h, grid_w):
            - flatten_patches: (B, grid_t * grid_h * grid_w,
                                C * temporal_patch_size * patch_size²)
            - grid_t, grid_h, grid_w: patch grid dimensions
    """
    if patches.ndim == 4:
        patches = patches.unsqueeze(1)

    # Pad temporal dimension to be divisible by temporal_patch_size
    if patches.shape[1] % temporal_patch_size != 0:
        repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
        patches = torch.cat([patches, repeats], dim=1)

    batch_size, grid_t, channel, height, width = patches.shape
    grid_t = grid_t // temporal_patch_size
    grid_h, grid_w = height // patch_size, width // patch_size

    patches = patches.view(
        batch_size,
        grid_t, temporal_patch_size,
        channel,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flatten_patches = patches.reshape(
        batch_size,
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )

    return flatten_patches, grid_t, grid_h, grid_w


class HyperCLOVAXVisionV2FastImageProcessorKwargs(DefaultFastImageProcessorKwargs, total=False):
    resize_mode: Optional[str]
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]
    # Token parameters
    image_token: Optional[str]
    image_start_token: Optional[str]
    image_end_token: Optional[str]
    # Discrete image parameters
    discrete_image_size: Optional[int]
    discrete_token_size: Optional[int]
    discrete_image_ratios: Optional[List]
    discrete_image_token: Optional[str]
    discrete_image_start_token: Optional[str]
    discrete_image_end_token: Optional[str]
    use_discrete_token: Optional[bool]
    vision_eol_token: Optional[str]
    vision_eof_token: Optional[str]


class HyperCLOVAXVisionV2ImageProcessor(BaseImageProcessorFast):
    """Fast image processor for HyperCLOVA X SEED.

    Supports two resize modes selected by the 'resize_mode' config field
    (see ResizeMode enum for details):

    ResizeMode.SMART_RESIZE (default):
      Pixel-count-constrained resize preserving aspect ratio. Both dimensions
      are kept divisible by patch_size * merge_size. Requires size to contain
      both 'shortest_edge' (min_pixels) and 'longest_edge' (max_pixels).

    ResizeMode.CLIP:
      Standard CLIP preprocess_val pipeline — resize shorter edge then
      center-crop to a square. Requires do_center_crop=True and crop_size set.
    """

    # Class-level defaults
    resample = PILResampling.BICUBIC
    resize_mode: ResizeMode = ResizeMode.SMART_RESIZE
    image_mean = _OPENAI_CLIP_MEAN
    image_std = _OPENAI_CLIP_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_center_crop = False
    crop_size = None
    size = {"shortest_edge": 3136, "longest_edge": 2073600}
    default_to_square = False
    min_pixels = 3136
    max_pixels = 2073600
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    image_token = "<|IMAGE_PAD|>"
    image_start_token = "<|image_start|>"
    image_end_token = "<|image_end|>"
    discrete_image_size = 384
    discrete_token_size = 27
    discrete_image_ratios = []
    discrete_image_token = "<|DISCRETE_IMAGE_PAD|>"
    discrete_image_start_token = "<|discrete_image_start|>"
    discrete_image_end_token = "<|discrete_image_end|>"
    use_discrete_token = False
    vision_eol_token = "<|vision_eol|>"
    vision_eof_token = "<|vision_eof|>"
    model_input_names = ["pixel_values"]
    valid_kwargs = HyperCLOVAXVisionV2FastImageProcessorKwargs

    def __init__(self, **kwargs):
        # Normalize resize_mode: str → ResizeMode enum
        resize_mode = kwargs.pop("resize_mode", None)
        if resize_mode is None:
            resize_mode = self.__class__.resize_mode
        if isinstance(resize_mode, str):
            resize_mode = ResizeMode(resize_mode)
        kwargs["resize_mode"] = resize_mode

        # Handle size <-> min_pixels/max_pixels
        size = kwargs.pop("size", None)
        min_pixels = kwargs.pop("min_pixels", None)
        max_pixels = kwargs.pop("max_pixels", None)

        size = {**self.size} if size is None else size
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
            size.pop("min_pixels", None)
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
            size.pop("max_pixels", None)
        if "shortest_edge" not in size:
            raise ValueError("size must contain 'shortest_edge' key.")

        # Normalize discrete_image_ratios: None → []
        if kwargs.get("discrete_image_ratios") is None:
            kwargs["discrete_image_ratios"] = []

        super().__init__(size=size, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs)

        # Ensure min_pixels/max_pixels are set from size in smart_resize mode
        if self.min_pixels is None and "shortest_edge" in self.size:
            self.min_pixels = self.size["shortest_edge"]
        if self.max_pixels is None and "longest_edge" in self.size:
            self.max_pixels = self.size["longest_edge"]

        # Build ratio -> token mapping from discrete_image_ratios
        ratios = self.discrete_image_ratios if self.discrete_image_ratios is not None else []
        self.discrete_image_ratio_tokens = {
            f"{r[0]}:{r[1]}": f"<|vision_ratio_{r[0]}:{r[1]}|>"
            for r in ratios
        }

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resize_mode: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Synchronize size <-> min_pixels/max_pixels based on resize_mode."""
        # Normalize resize_mode
        if resize_mode is None:
            resize_mode = self.resize_mode
        if isinstance(resize_mode, str):
            resize_mode = ResizeMode(resize_mode)

        if resize_mode == ResizeMode.SMART_RESIZE:
            if min_pixels is not None and max_pixels is not None:
                size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
            elif size is not None:
                if "shortest_edge" not in size or "longest_edge" not in size:
                    raise ValueError(
                        "smart_resize mode requires size with both 'shortest_edge' and 'longest_edge'."
                    )
                min_pixels = size["shortest_edge"]
                max_pixels = size["longest_edge"]
            else:
                size = {**self.size}
        else:  # ResizeMode.CLIP
            if size is None:
                size = {**self.size}
            if "shortest_edge" not in size:
                raise ValueError("clip mode requires size with 'shortest_edge'.")

        return super()._further_process_kwargs(
            size=size, min_pixels=min_pixels, max_pixels=max_pixels,
            resize_mode=resize_mode, **kwargs,
        )

    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        interpolation: Optional[F.InterpolationMode] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Resize hook called by BaseImageProcessorFast._preprocess.

        Dispatches based on self.resize_mode (ResizeMode enum):
        - SMART_RESIZE: pixel-count-constrained smart_resize preserving aspect ratio.
        - CLIP: standard shortest-edge resize delegated to base class.
        """
        if interpolation is None:
            interpolation = F.InterpolationMode.BICUBIC
        if self.resize_mode == ResizeMode.SMART_RESIZE:
            h, w = image.shape[-2:]
            new_h, new_w = smart_resize(
                h, w,
                factor=self.patch_size * self.merge_size,
                min_pixels=size["shortest_edge"],
                max_pixels=size["longest_edge"],
            )
            return F.resize(image, [new_h, new_w], interpolation=interpolation, antialias=True)
        else:
            # ResizeMode.CLIP
            # BaseImageProcessorFast's resize() will handle the standard CLIP resizing (shortest edge + center crop)
            return super().resize(image, size, interpolation=interpolation, **kwargs)

    def _find_best_ratio_token(
        self,
        original_size: List[int],
        discrete_image_ratios: Optional[List[List[int]]] = None,
    ) -> List[int]:
        """Find the best ratio token based on the original image aspect ratio.

        Args:
            original_size: Original [height, width] of the image.
            discrete_image_ratios: List of [h, w] ratio pairs. Defaults to self.discrete_image_ratios.

        Returns:
            Best matching [h_ratio, w_ratio] list element from discrete_image_ratios.
        """
        discrete_image_ratios = discrete_image_ratios if discrete_image_ratios is not None else self.discrete_image_ratios

        if not discrete_image_ratios:
            return (1, 1)

        h, w = original_size
        if h == 0 or w == 0:
            return (1, 1)

        ratios = [i / j for i, j in discrete_image_ratios]
        diffs = [abs(w / h - r) for r in ratios]
        best_size_idx = diffs.index(min(diffs))

        return discrete_image_ratios[best_size_idx]

    def _preprocess_discrete_image(
        self,
        images: List[PIL.Image.Image],
        original_sizes: List[Tuple[int, int]],
        interpolation: Optional[F.InterpolationMode],
    ) -> dict:
        """Preprocess images for discrete vision tokens.

        Resizes each image to a fixed size (discrete_image_size) and finds
        the closest aspect ratio token.

        Args:
            images: List of image tensors to preprocess.
            original_sizes: List of (height, width) tuples for each image.
            interpolation: Interpolation method.

        Returns:
            Dictionary with:
                - "discrete_pixel_values": Tensor of shape (N, C, discrete_image_size, discrete_image_size).
                - "discrete_image_ratios": Tensor of shape (N, 2).
                - "num_discrete_image_tokens": Tensor of shape (N,) with per-image discrete token counts.
        """
        discrete_pixel_values_list = []
        discrete_image_ratios_list = []

        for i, img in enumerate(images):
            orig_h, orig_w = original_sizes[i]
            best_ratio = self._find_best_ratio_token([orig_h, orig_w])

            # Resize to fixed discrete_image_size x discrete_image_size (torchvision)
            discrete_img = F.resize(
                img.unsqueeze(0),
                [self.discrete_image_size, self.discrete_image_size],
                interpolation=interpolation,
                antialias=True,
            )
            discrete_img = discrete_img.squeeze(0)

            # Match torchvision to_tensor: float32 / 255.0 (no normalize)
            discrete_img = discrete_img.to(torch.float32) / 255.0

            discrete_pixel_values_list.append(discrete_img)
            discrete_image_ratios_list.append(best_ratio)

        n = len(images)
        discrete_token_size = self.discrete_token_size
        # ratio_token(1) + discrete_token_size rows * (discrete_token_size tokens + vision_eol(1)) + vision_eof(1)
        num_discrete_per_image = 1 + discrete_token_size * (discrete_token_size + 1) + 1

        return {
            "discrete_pixel_values": torch.stack(discrete_pixel_values_list),
            "discrete_image_ratios": torch.tensor(discrete_image_ratios_list),
            "num_discrete_image_tokens": torch.full((n,), num_discrete_per_image, dtype=torch.long),
        }

    def _preprocess(
        self,
        images: List[PIL.Image.Image],
        **kwargs,
    ) -> BatchFeature:
        """Main preprocessing entry point called by BaseImageProcessorFast.

        Delegates resize/center_crop/rescale/normalize to the base class
        (using our resize() override for smart_resize dispatch), then
        patchifies the normalized tensors.

        Returns:
            BatchFeature containing pixel_values, image_grid_thw, num_image_tokens,
            and optionally discrete processing results.
        """
        patch_size = kwargs.get("patch_size", self.patch_size)
        temporal_patch_size = kwargs.get("temporal_patch_size", self.temporal_patch_size)
        merge_size = kwargs.get("merge_size", self.merge_size)
        return_tensors = kwargs.get("return_tensors", None)

        # Ensure interpolation is resolved: BaseImageProcessorFast may pass 'resample' (int)
        # in older transformers versions. Normalize to InterpolationMode for our resize() override.
        if not kwargs.get("interpolation"):
            resample = kwargs.get("resample", self.resample)
            if resample is not None and isinstance(resample, int):
                _pil_to_torch = {
                    0: F.InterpolationMode.NEAREST,
                    1: F.InterpolationMode.LANCZOS,
                    2: F.InterpolationMode.BILINEAR,
                    3: F.InterpolationMode.BICUBIC,
                    4: F.InterpolationMode.BOX,
                    5: F.InterpolationMode.HAMMING,
                }
                kwargs["interpolation"] = _pil_to_torch.get(int(resample), F.InterpolationMode.BICUBIC)
            elif resample is not None:
                kwargs["interpolation"] = resample
            else:
                kwargs["interpolation"] = F.InterpolationMode.BICUBIC

        # Record original sizes before any transforms (needed for discrete processing)
        if self.use_discrete_token:
            original_sizes = [(img.shape[-2], img.shape[-1]) for img in images]

        # Delegate resize (our override) + [center_crop] + rescale + normalize to base class.
        # return_tensors=None → returns BatchFeature({"pixel_values": list[Tensor(C, H, W)]})
        base_result = super()._preprocess(images, **{**kwargs, "return_tensors": None})
        normalized_images = base_result["pixel_values"]  # list[Tensor(C, H, W)]

        # Patchify each normalized image independently
        all_patches, all_grids = [], []
        for img in normalized_images:
            flatten_patches, grid_t, grid_h, grid_w = patchify(
                img.unsqueeze(0).to(torch.float32),
                patch_size, temporal_patch_size, merge_size,
            )
            all_patches.append(flatten_patches.squeeze(0))
            all_grids.append([grid_t, grid_h, grid_w])

        pixel_values = torch.cat(all_patches, dim=0)
        image_grid_thw = torch.tensor(all_grids)
        num_image_tokens = image_grid_thw.prod(dim=1) // (merge_size ** 2)

        data = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "num_image_tokens": num_image_tokens,
        }

        if self.use_discrete_token:
            discrete_result = self._preprocess_discrete_image(
                images,
                original_sizes=original_sizes,
                interpolation=kwargs.get("interpolation"),
            )
            data.update(discrete_result)

        return BatchFeature(data=data, tensor_type=return_tensors)

    def get_num_image_tokens(
        self,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        pixel_values: Optional[torch.Tensor] = None,
        include_boundary_tokens: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tuple: Optional[bool] = None,
    ) -> Union[int, Tuple[int, int]]:
        """Compute the number of image tokens for the given input.

        Args:
            image_width: Image width (used when pixel_values is None).
            image_height: Image height (used when pixel_values is None).
            pixel_values: Pre-computed pixel values tensor.
            include_boundary_tokens: Whether to include start/end boundary tokens.
            min_pixels: Minimum pixel count. Defaults to self.min_pixels.
            max_pixels: Maximum pixel count. Defaults to self.max_pixels.
            patch_size: ViT patch size. Defaults to self.patch_size.
            merge_size: Token reduction merge size. Defaults to self.merge_size.
            return_tuple: If True, return (continuous, discrete) tuple.
                Otherwise return the sum.

        Returns:
            Token count as int, or (continuous, discrete) tuple if return_tuple is True.
        """
        patch_size = patch_size if patch_size is not None else self.patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        num_continuous_tokens, num_discrete_tokens = 0, 0
        if pixel_values is None:
            factor = patch_size * merge_size
            resized_height, resized_width = smart_resize(
                image_height, image_width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
            )
            grid_h = resized_height // patch_size
            grid_w = resized_width // patch_size
            num_continuous_tokens = (grid_h // merge_size) * (grid_w // merge_size)
        elif len(pixel_values.shape) == 2:
            num_continuous_tokens = pixel_values.shape[0] // (merge_size ** 2)
        else:
            num_continuous_tokens = sum([
                _pixel_values.shape[0] // (merge_size ** 2)
                for _pixel_values in pixel_values
            ])
        if include_boundary_tokens:
            num_continuous_tokens += 2

        if self.use_discrete_token:
            discrete_token_size = self.discrete_token_size
            num_discrete_tokens = discrete_token_size ** 2
            if include_boundary_tokens:
                num_discrete_tokens += 2

        if return_tuple:
            return (num_continuous_tokens, num_discrete_tokens)
        else:
            return num_continuous_tokens + num_discrete_tokens

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        *args,
        **kwargs,
    ) -> None:
        """Save the processor to a directory.

        Registers for auto class before saving.

        Args:
            save_directory: Directory path to save the processor.
        """
        self.register_for_auto_class()
        super().save_pretrained(save_directory, *args, **kwargs)


__all__ = ["ResizeMode", "HyperCLOVAXVisionV2FastImageProcessorKwargs", "HyperCLOVAXVisionV2ImageProcessor"]
