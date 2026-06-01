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
HyperCLOVA X SEED Video Processor

Implements dynamic resolution video processing:
- Smart resize: adjusts video frames to fit within min_pixels and max_pixels
- Temporal patch: frame grouping by temporal_patch_size
- Patch flattening: token reduction using merge_size

Based on BaseVideoProcessor with torchvision resize.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
try:
    from torchvision.transforms.v2 import functional as F
except ImportError:
    from torchvision.transforms import functional as F  # torchvision < 0.15
try:
    from transformers.image_processing_utils import BatchFeature
except ImportError:
    from transformers import BatchFeature
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
try:
    from transformers.processing_utils import VideosKwargs
except ImportError:
    from typing import TypedDict as VideosKwargs  # transformers < 4.46
try:
    from transformers.video_processing_utils import BaseVideoProcessor
    from transformers.video_utils import group_videos_by_shape, reorder_videos
except ImportError:
    from transformers.image_processing_utils_fast import BaseImageProcessorFast as BaseVideoProcessor
    from transformers.image_processing_utils_fast import group_images_by_shape as group_videos_by_shape
    from transformers.image_processing_utils_fast import reorder_images as reorder_videos
from .image_processing_hyperclovax_vision_v2 import patchify
try:
    from torchvision.transforms.v2 import InterpolationMode as _InterpolationMode
except ImportError:
    from torchvision.transforms import InterpolationMode as _InterpolationMode  # torchvision < 0.15
_pil_to_torch_interpolation = {
    0: _InterpolationMode.NEAREST,
    1: _InterpolationMode.LANCZOS,
    2: _InterpolationMode.BILINEAR,
    3: _InterpolationMode.BICUBIC,
    4: _InterpolationMode.BOX,
    5: _InterpolationMode.HAMMING,
}


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    """Smart resize for dynamic resolution.

    Adjusts dimensions so that both sides are divisible by factor
    and total pixel count is between min_pixels and max_pixels.

    Adapted from the Qwen2.5-VL image processing implementation.
    Reference: https://github.com/QwenLM/Qwen2.5-VL (Apache 2.0 License)

    Args:
        height: Original height.
        width: Original width.
        factor: Rounding unit (default: 28 = patch_size * merge_size).
        min_pixels: Minimum pixel count.
        max_pixels: Maximum pixel count.

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


class HyperCLOVAXVisionV2VideosKwargs(VideosKwargs, total=False):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    max_frames: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]
    video_token: Optional[str]
    video_start_token: Optional[str]
    video_end_token: Optional[str]
    use_audio_in_video: Optional[bool]
    use_discrete_token: Optional[bool]
    vision_eol_token: Optional[str]
    vision_eof_token: Optional[str]


class HyperCLOVAXVisionV2VideoProcessor(BaseVideoProcessor):
    """Video processor for HyperCLOVA X SEED.

    Uses torchvision for resize and inline torch ops for rescale/normalize,
    with dynamic resolution video processing.
    """

    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        min_pixels: int = 128 * 28 * 28,
        max_pixels: int = 28 * 28 * 768,
        max_frames: int = 120,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        resample: int = PILResampling.BICUBIC,
        use_audio_in_video: bool = False,
        # Token parameters
        video_token: str = "<|VIDEO_PAD|>",
        video_start_token: str = "<|video_start|>",
        video_end_token: str = "<|video_end|>",
        video_audio_token: str = "<|VIDEO_AUDIO_PAD|>",
        # Discrete video parameters
        use_discrete_token: bool = False,
        vision_eol_token: str = "<|vision_eol|>",
        vision_eof_token: str = "<|vision_eof|>",
        **kwargs,
    ):
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}

        super().__init__(
            size=size,
            do_resize=do_resize,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean if image_mean is not None else _OPENAI_CLIP_MEAN,
            image_std=image_std if image_std is not None else _OPENAI_CLIP_STD,
            do_convert_rgb=do_convert_rgb,
            resample=resample,
            # Custom fields
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_frames=max_frames,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            use_audio_in_video=use_audio_in_video,
            # Token parameters
            video_token=video_token,
            video_start_token=video_start_token,
            video_end_token=video_end_token,
            video_audio_token=video_audio_token,
            # Discrete video parameters
            use_discrete_token=use_discrete_token,
            vision_eol_token=vision_eol_token,
            vision_eof_token=vision_eof_token,
        )

    def _preprocess_continuous_video(
        self,
        videos: List[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        interpolation: _InterpolationMode,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, tuple]],
        image_std: Optional[Union[float, tuple]],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
    ) -> dict:
        """Preprocess a single video for continuous vision features.

        Performs group_videos_by_shape -> resize -> rescale/normalize -> patchify.

        Args:
            videos: List of channel-first torch tensors, each of shape (num_frames, C, H, W).
            do_resize: Whether to perform resizing.
            size: SizeDict with shortest_edge/longest_edge (smart_resize min/max pixels).
            interpolation: torchvision InterpolationMode.
            do_rescale: Whether to perform rescaling.
            rescale_factor: Rescale factor.
            do_normalize: Whether to perform normalization.
            image_mean: Normalization mean (tuple).
            image_std: Normalization standard deviation (tuple).
            patch_size: ViT patch size.
            temporal_patch_size: Temporal patch size.
            merge_size: Token merge size.

        Returns:
            Dictionary with:
                - "pixel_values_videos": Tensor of shape (grid_t * grid_h * grid_w, feat_dim).
                - "video_grid_thw": List of [grid_t, grid_h, grid_w].
                - "num_video_tokens": Number of continuous tokens (int).
        """
        # 1. Group & smart resize
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)

        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            height, width = stacked_videos[0].shape[-2], stacked_videos[0].shape[-1]
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height, width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_videos = F.resize(
                    stacked_videos,
                    [resized_height, resized_width],
                    interpolation=interpolation,
                    antialias=True,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # 2. Group again -> rescale/normalize -> patchify
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = stacked_videos[0].shape[-2], stacked_videos[0].shape[-1]

            if do_rescale or do_normalize:
                stacked_videos = stacked_videos.to(torch.float32)
                if do_rescale:
                    stacked_videos = stacked_videos * rescale_factor
                if do_normalize:
                    mean_t = torch.tensor(list(image_mean), dtype=stacked_videos.dtype, device=stacked_videos.device).reshape(1, 1, 3, 1, 1)
                    std_t = torch.tensor(list(image_std), dtype=stacked_videos.dtype, device=stacked_videos.device).reshape(1, 1, 3, 1, 1)
                    stacked_videos = (stacked_videos - mean_t) / std_t
            patches = stacked_videos

            flatten_patches, grid_t, grid_h, grid_w = patchify(
                patches, patch_size, temporal_patch_size, merge_size
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * flatten_patches.shape[0]

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)

        num_video_tokens = (video_grid_thw.prod(dim=1) // (merge_size ** 2)).item()

        return {
            "pixel_values_videos": pixel_values_videos.squeeze(0),
            "video_grid_thw": video_grid_thw[0].tolist(),
            "num_video_tokens": num_video_tokens,
        }

    def _preprocess_discrete_video(self, video: torch.Tensor) -> dict:
        """Preprocess a single video for discrete vision tokens.

        Args:
            video: Video tensor.

        Raises:
            NotImplementedError: Discrete video tokenization is not yet supported.
        """
        raise NotImplementedError("Discrete video tokenization is not yet supported.")

    def preprocess(
        self,
        videos: Union[List[List[np.ndarray]], List[np.ndarray]],
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        """Preprocess a batch of videos.

        Resolves all kwargs at the entry point, then routes each video to
        ``_preprocess_continuous_video`` or ``_preprocess_discrete_video``.

        Args:
            videos: Video input. Either:
                - np.ndarray: Single video of shape (num_frames, H, W, C).
                - List[np.ndarray]: Batch of videos, each 4D.
            return_tensors: Desired tensor type for outputs.

        Returns:
            BatchFeature with:
                - pixel_values_videos: Tensor of shape (total_patches, feat_dim).
                - video_grid_thw: Tensor of shape (num_videos, 3).
                - num_video_tokens: Tensor of shape (num_videos,).

        Note:
            Discrete video tokenization (``use_discrete_token=True``) is not yet
            implemented and will raise ``NotImplementedError``.
        """
        if isinstance(videos, np.ndarray) and videos.ndim == 4:
            videos = [videos]

        # 1. Resolve kwargs from self attributes
        do_resize = kwargs.pop("do_resize", None)
        if do_resize is None:
            do_resize = self.do_resize

        do_rescale = kwargs.pop("do_rescale", None)
        if do_rescale is None:
            do_rescale = self.do_rescale

        rescale_factor = kwargs.pop("rescale_factor", None)
        if rescale_factor is None:
            rescale_factor = self.rescale_factor

        do_normalize = kwargs.pop("do_normalize", None)
        if do_normalize is None:
            do_normalize = self.do_normalize

        do_convert_rgb = kwargs.pop("do_convert_rgb", None)
        if do_convert_rgb is None:
            do_convert_rgb = self.do_convert_rgb

        resample = kwargs.pop("resample", None)
        if resample is None:
            resample = self.resample

        image_mean = kwargs.pop("image_mean", None)
        if image_mean is None:
            image_mean = self.image_mean

        image_std = kwargs.pop("image_std", None)
        if image_std is None:
            image_std = self.image_std

        patch_size = kwargs.pop("patch_size", None)
        if patch_size is None:
            patch_size = self.patch_size

        temporal_patch_size = kwargs.pop("temporal_patch_size", None)
        if temporal_patch_size is None:
            temporal_patch_size = self.temporal_patch_size

        merge_size = kwargs.pop("merge_size", None)
        if merge_size is None:
            merge_size = self.merge_size

        min_pixels = kwargs.pop("min_pixels", None)
        if min_pixels is None:
            min_pixels = self.size["shortest_edge"]

        max_pixels = kwargs.pop("max_pixels", None)
        if max_pixels is None:
            max_pixels = self.size["longest_edge"]

        size = SizeDict(shortest_edge=min_pixels, longest_edge=max_pixels)

        use_discrete_token = kwargs.pop("use_discrete_token", None)
        if use_discrete_token is None:
            use_discrete_token = self.use_discrete_token

        # 2. Convert resample -> interpolation, mean/std -> tuple
        if isinstance(resample, int):
            interpolation = _pil_to_torch_interpolation.get(int(resample), _InterpolationMode.BICUBIC)
        else:
            interpolation = resample

        if isinstance(image_mean, list):
            image_mean = tuple(image_mean)
        if isinstance(image_std, list):
            image_std = tuple(image_std)

        # 3. Per-video processing: route to continuous or discrete sub-processor
        pixel_values_list = []
        grid_thw_list = []
        num_video_tokens_list = []

        for video in videos:
            if isinstance(video, np.ndarray):
                # NHWC -> NCHW
                video = torch.from_numpy(np.ascontiguousarray(video.transpose(0, 3, 1, 2)))

            if do_convert_rgb:
                c = video.shape[1]  # (N, C, H, W)
                if c == 1:
                    video = video.expand(-1, 3, -1, -1).contiguous()
                elif c == 4:
                    video = video[:, :3].contiguous()

            if use_discrete_token:
                result = self._preprocess_discrete_video(video)
            else:
                result = self._preprocess_continuous_video(
                    videos=[video],
                    do_resize=do_resize,
                    size=size,
                    interpolation=interpolation,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                )
            pixel_values_list.append(result["pixel_values_videos"])
            grid_thw_list.append(result["video_grid_thw"])
            num_video_tokens_list.append(result["num_video_tokens"])

        data = {
            "pixel_values_videos": torch.cat(pixel_values_list, dim=0),
            "video_grid_thw": torch.tensor(grid_thw_list),
            "num_video_tokens": torch.tensor(num_video_tokens_list, dtype=torch.long),
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def get_num_video_tokens(
        self,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        num_frames: Optional[int] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        include_boundary_tokens: bool = False,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        return_tuple: Optional[bool] = None,
    ) -> Union[int, Tuple[int, int]]:
        """Compute the number of video tokens for the given input.

        Args:
            image_width: Frame width (used when pixel_values_videos is None).
            image_height: Frame height (used when pixel_values_videos is None).
            num_frames: Number of frames (used when pixel_values_videos is None).
            pixel_values_videos: Pre-computed pixel values tensor.
            include_boundary_tokens: Whether to include start/end boundary tokens.
            patch_size: ViT patch size. Defaults to self.patch_size.
            temporal_patch_size: Temporal patch size. Defaults to self.temporal_patch_size.
            merge_size: Token reduction merge size. Defaults to self.merge_size.
            min_pixels: Minimum pixel count. Defaults to self.size["shortest_edge"].
            max_pixels: Maximum pixel count. Defaults to self.size["longest_edge"].
            return_tuple: If True, return (continuous, discrete) tuple.
                Otherwise return the sum.

        Returns:
            Token count as int, or (continuous, discrete) tuple if return_tuple is True.
        """
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        min_pixels = min_pixels if min_pixels is not None else self.size["shortest_edge"]
        max_pixels = max_pixels if max_pixels is not None else self.size["longest_edge"]

        num_continuous_tokens, num_discrete_tokens = 0, 0
        if pixel_values_videos is None:
            factor = patch_size * merge_size
            resized_height, resized_width = smart_resize(
                image_height, image_width, factor, min_pixels=min_pixels, max_pixels=max_pixels
            )
            grid_t = num_frames // temporal_patch_size
            grid_h = resized_height // patch_size
            grid_w = resized_width // patch_size
            num_continuous_tokens = (grid_t * grid_h * grid_w) // (merge_size ** 2)
        elif len(pixel_values_videos.shape) == 2:
            num_continuous_tokens = pixel_values_videos.shape[0] // (merge_size ** 2)
        else:
            num_continuous_tokens = sum(
                pv.shape[0] // (merge_size ** 2) for pv in pixel_values_videos
            )
        if include_boundary_tokens:
            num_continuous_tokens += 2

        if return_tuple:
            return (num_continuous_tokens, num_discrete_tokens)
        else:
            return num_continuous_tokens + num_discrete_tokens


__all__ = ["HyperCLOVAXVisionV2VideosKwargs", "HyperCLOVAXVisionV2VideoProcessor"]
