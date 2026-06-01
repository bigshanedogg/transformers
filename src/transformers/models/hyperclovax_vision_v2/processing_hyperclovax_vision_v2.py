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
"""HyperCLOVA X SEED multimodal processor"""

import base64
import copy
import io
import json
import mimetypes
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
import PIL
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoVideoProcessor,
)
from transformers.audio_utils import AudioInput
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import (
    AudioKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
    VideosKwargs,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.video_utils import VideoInput

logger = logging.get_logger(__name__)

# Default timeout for HTTP requests (connect, read) in seconds
_DEFAULT_REQUEST_TIMEOUT = (5, 60)

# Per-sample token counts used only for placeholder expansion (and optionally by
# external tooling). They are NOT model forward inputs, so they are excluded from
# the returned BatchFeature. If a downstream consumer needs the counts, derive them
# from `image_grid_thw` / `video_grid_thw`.
_PLACEHOLDER_COUNT_KEYS = frozenset(
    {
        "num_image_tokens",
        "num_discrete_image_tokens",
        "num_video_tokens",
        "num_discrete_video_tokens",
        "num_audio_tokens",
        "num_discrete_audio_tokens",
        "num_video_audio_tokens",
        "num_discrete_video_audio_tokens",
    }
)


def _detect_audio_suffix(audio_bytes: bytes) -> str:
    """Return a file-extension suffix (e.g. '.wav') from magic bytes, defaulting to '.wav'."""
    header = audio_bytes[:12]
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return ".wav"
    if header[:4] == b"fLaC":
        return ".flac"
    if header[:4] == b"OggS":
        return ".ogg"
    if header[:3] == b"ID3" or header[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    if header[4:8] == b"ftyp":
        return ".m4a"
    return ".wav"


class HyperCLOVAXVisionV2AudioKwargs(AudioKwargs, total=False):
    sample_rate: int
    chunk_unit: int
    min_chunk_size: int


class HyperCLOVAXVisionV2TextKwargs(TextKwargs, total=False):
    return_mm_token_type_ids: bool


class HyperCLOVAXVisionV2VideosKwargs(VideosKwargs, total=False):
    max_num_frames: int


class HyperCLOVAXVisionV2ProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: HyperCLOVAXVisionV2AudioKwargs
    text_kwargs: HyperCLOVAXVisionV2TextKwargs
    videos_kwargs: HyperCLOVAXVisionV2VideosKwargs
    _defaults = {
        "audio_kwargs": {
            "sample_rate": 16_000,
            "chunk_unit": 80,
            "min_chunk_size": 1_600,
        },
        "images_kwargs": {},
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {
            "max_num_frames": 120,
        },
    }


class HyperCLOVAXVisionV2Processor(ProcessorMixin):
    r"""
    Processor for HyperCLOVA X SEED multimodal model.

    Combines a tokenizer, image processor, and video processor into a single processor
    that handles text, image, and video inputs. Supports both continuous and discrete
    representations for each modality. (Audio is added by the omni subclass.)

    Args:
        image_processor ([`HyperCLOVAXVisionV2ImageProcessor`], *optional*):
            Image processor for continuous and discrete image processing.
        video_processor ([`HyperCLOVAXVisionV2VideoProcessor`], *optional*):
            Video processor for continuous video processing.
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            Tokenizer for text encoding and special token management.
        chat_template (`str`, *optional*):
            Jinja2 chat template string. Falls back to the tokenizer's chat template if not provided.

    ```python
    >>> from transformers import AutoProcessor

    >>> processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")
    ```
    """

    def __init__(
        self,
        image_processor: Optional[AutoImageProcessor] = None,
        video_processor: Optional[AutoVideoProcessor] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        # Prefer explicit chat_template; fall back to tokenizer's if available
        if chat_template is None and hasattr(tokenizer, "chat_template"):
            chat_template = tokenizer.chat_template

        ProcessorMixin.__init__(
            self,
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )

        # Audio is contributed by the omni subclass; vision-only processors have no
        # audio sub-processor. Kept as None so the audio-aware code paths below
        # (all guarded by `self.audio_processor is not None`) stay dormant.
        self.audio_processor = None

        self.modalities = list()
        if self.audio_processor is not None:
            self.modalities.append("audio")
            self.audio_token = self.audio_processor.audio_token
            self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_processor.audio_token)
            self.audio_start_token_id = tokenizer.convert_tokens_to_ids(self.audio_processor.audio_start_token)
            self.audio_end_token_id = tokenizer.convert_tokens_to_ids(self.audio_processor.audio_end_token)

            self.discrete_audio_token_id = None
            self.discrete_audio_start_token_id = None
            self.discrete_audio_end_token_id = None
            if self.audio_processor.use_discrete_token:
                self.discrete_audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_processor.discrete_audio_token)
                self.discrete_audio_start_token_id = tokenizer.convert_tokens_to_ids(self.audio_processor.discrete_audio_start_token)
                self.discrete_audio_end_token_id = tokenizer.convert_tokens_to_ids(self.audio_processor.discrete_audio_end_token)

        if self.image_processor is not None:
            self.modalities.append("image")
            self.image_token = self.image_processor.image_token
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_processor.image_token)
            self.image_start_token_id = tokenizer.convert_tokens_to_ids(self.image_processor.image_start_token)
            self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_processor.image_end_token)

            self.discrete_image_token_id = None
            self.discrete_image_start_token_id = None
            self.discrete_image_end_token_id = None
            if self.image_processor.use_discrete_token:
                self.discrete_image_token_id = tokenizer.convert_tokens_to_ids(self.image_processor.discrete_image_token)
                self.discrete_image_start_token_id = tokenizer.convert_tokens_to_ids(self.image_processor.discrete_image_start_token)
                self.discrete_image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_processor.discrete_image_end_token)

        if self.video_processor is not None:
            self.modalities.append("video")
            self.video_token = self.video_processor.video_token
            self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_processor.video_token)
            self.video_audio_token = self.video_processor.video_audio_token
            self.video_start_token_id = tokenizer.convert_tokens_to_ids(self.video_processor.video_start_token)
            self.video_end_token_id = tokenizer.convert_tokens_to_ids(self.video_processor.video_end_token)
            self.video_audio_token_id = tokenizer.convert_tokens_to_ids(self.video_processor.video_audio_token)

    def load_multimodal_inputs(
        self,
        conversation: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        use_audio_in_video: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Load audio, image, and video data referenced in conversations.

        Extracts media references from conversation messages and loads the
        actual data. Each message may reference media either via top-level
        keys (``audio_files``, ``image_files``, ``video_files``) or via
        structured content blocks with ``type`` set to ``"audio"``,
        ``"image"``, or ``"video"``.

        Supported input formats for each media type:
            - Local file path (e.g., ``"/path/to/file.wav"``)
            - HTTP/HTTPS URL (e.g., ``"https://example.com/image.jpg"``)
            - Base64 data URI (e.g., ``"data:audio/wav;base64,..."``)

        Args:
            conversation: A single conversation (list of message dicts) or
                a batch of conversations (list of lists). Each message dict
                should have a ``"role"`` and ``"content"`` key following the
                chat format, matching the input accepted by
                ``processor.tokenizer.apply_chat_template``.
            use_audio_in_video: If ``True``, extract audio tracks from
                video files and include them in the returned audio list.

        Returns:
            Plain dict with flat parallel lists::

                {
                    "audios": List[np.ndarray] | None,
                    "sampling_rates": List[int] | None,
                    "images": List[PIL.Image.Image] | None,
                    "videos": List[List[PIL.Image.Image]] | None,
                    "video_audios": List[np.ndarray | None] | None,
                    "video_sampling_rates": List[int | None] | None,
                    "video_fps_list": List[float] | None,
                }

            Pass directly to the processor via ``processor(**mm)``.
        """
        if use_audio_in_video is None:
            use_audio_in_video = bool(
                self.video_processor is not None
                and getattr(self.video_processor, "use_audio_in_video", False)
            )

        conversations = conversation if isinstance(conversation[0], list) else [conversation]

        audios: List[np.ndarray] = []
        sampling_rates: List[int] = []
        images: List[Image.Image] = []
        videos: List[List[Image.Image]] = []
        video_audios: List[Optional[np.ndarray]] = []
        video_sampling_rates: List[Optional[int]] = []
        video_fps_list: List[float] = []

        target_sr = 16_000
        if (
            self.audio_processor is not None
            and hasattr(self.audio_processor, "sampling_rate")
        ):
            target_sr = self.audio_processor.sampling_rate

        for conv in conversations:
            for message in conv:
                # Handle media files at message level (when content is a string)
                if message.get("audio_files"):
                    for audio_path in message["audio_files"]:
                        info = self._load_audio(audio_path, sr=target_sr)
                        audios.append(info["waveform"])
                        sampling_rates.append(info["sampling_rate"])

                if message.get("image_files"):
                    for image_path in message["image_files"]:
                        info = self._load_image(image_path)
                        images.append(info["image"])

                if message.get("video_files"):
                    for video_path in message["video_files"]:
                        info = self._load_video(
                            video_path,
                            sr=target_sr,
                            use_audio_in_video=use_audio_in_video,
                        )
                        videos.append(info["frames"])
                        video_audios.append(info["audio"])
                        video_sampling_rates.append(info["sampling_rate"])
                        video_fps_list.append(info["fps"])

                content = message.get("content", [])
                if not isinstance(content, list):
                    continue

                for ele in content:
                    type_ = ele.get("type")

                    if type_ in ("audio", "audio_url"):
                        raw = ele.get("audio", ele.get("audio_url"))
                        # OpenAI-style: {"type": "audio_url", "audio_url": {"url": "..."}}
                        if isinstance(raw, dict):
                            raw = raw.get("url", raw)
                        path = raw
                        if path:
                            if "mime_type" not in ele and isinstance(path, str):
                                filename = ele.get("filename", path if not path.startswith("http") else "a.wav")
                                mime_type = mimetypes.guess_type(filename)[0]
                                if mime_type:
                                    ele["mime_type"] = mime_type

                            info = self._load_audio(
                                path, sr=target_sr,
                                start=ele.get("audio_start", 0.0),
                                end=ele.get("audio_end", None),
                            )
                            audios.append(info["waveform"])
                            sampling_rates.append(info["sampling_rate"])

                    elif type_ in ("image", "image_url"):
                        raw = ele.get("image", ele.get("image_url"))
                        # OpenAI-style: {"type": "image_url", "image_url": {"url": "..."}}
                        if isinstance(raw, dict):
                            raw = raw.get("url", raw)
                        path = raw
                        if path:
                            if "mime_type" not in ele and isinstance(path, str):
                                filename = ele.get("filename", path if not path.startswith("http") else "a.jpg")
                                mime_type = mimetypes.guess_type(filename)[0]
                                if mime_type:
                                    ele["mime_type"] = mime_type

                            info = self._load_image(path)
                            images.append(info["image"])

                    elif type_ in ("video", "video_url"):
                        raw = ele.get("video", ele.get("video_url"))
                        # OpenAI-style: {"type": "video_url", "video_url": {"url": "..."}}
                        if isinstance(raw, dict):
                            raw = raw.get("url", raw)
                        path = raw
                        if path:
                            if "mime_type" not in ele and isinstance(path, str):
                                filename = ele.get("filename", path if not path.startswith("http") else "a.mp4")
                                mime_type = mimetypes.guess_type(filename)[0]
                                if mime_type:
                                    ele["mime_type"] = mime_type

                            info = self._load_video(
                                path,
                                start=ele.get("video_start", 0.0),
                                end=ele.get("video_end", None),
                                max_num_frames=ele.get("max_num_frames", None),
                                sr=target_sr,
                                use_audio_in_video=use_audio_in_video,
                            )
                            videos.append(info["frames"])
                            video_audios.append(info["audio"])
                            video_sampling_rates.append(info["sampling_rate"])
                            video_fps_list.append(info["fps"])

        return {
            "audios": audios if audios else None,
            "sampling_rates": sampling_rates if sampling_rates else None,
            "images": images if images else None,
            "videos": videos if videos else None,
            "video_audios": video_audios if video_audios else None,
            "video_sampling_rates": video_sampling_rates if video_sampling_rates else None,
            "video_fps_list": video_fps_list if video_fps_list else None,
        }

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        chat_template: Optional[str] = None,
        tokenize: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """
        Apply the chat template to a conversation and optionally tokenize the result.

        This override extends the base class behaviour by also loading and processing
        multimodal inputs (audio, images, videos, video-audio tracks) that are
        referenced inside the conversation, then forwarding everything to
        [`__call__`] in a single step.

        Args:
            conversation: A single conversation or a batch of conversations.
            chat_template: Jinja2 template string or template name. Falls back to
                the processor's default chat template when not provided.
            tokenize: If ``True``, tokenize the rendered prompt and process all
                multimodal inputs, returning a [`~transformers.BatchFeature`].
                If ``False`` (default), return the rendered prompt string only.
            return_dict: If ``True`` and ``tokenize=True``, return the full
                [`~transformers.BatchFeature`] dict. If ``False``, return
                only ``input_ids``.
            **kwargs: Split automatically into two groups:

                - *Template kwargs* — any key not recognised by
                  [`HyperCLOVAXVisionV2ProcessorKwargs`] is forwarded to
                  ``tokenizer.apply_chat_template`` as a Jinja variable or
                  standard tokenizer argument (e.g. ``add_generation_prompt``,
                  ``use_audio_in_video``, ``skip_reasoning``, ``tools``).
                - *Processor kwargs* — keys declared in
                  [`HyperCLOVAXVisionV2ProcessorKwargs`] (or its modality
                  sub-dicts such as ``text_kwargs``) are forwarded exclusively
                  to [`__call__`] (e.g. ``return_tensors``, ``padding``).

        Returns:
            ``str`` when ``tokenize=False``;
            [`~transformers.BatchFeature`] when ``tokenize=True`` and
            ``return_dict=True``; ``list[list[int]]`` otherwise.
        """
        # Build the set of kwarg keys that __call__ recognises via _merge_kwargs.
        # These are the flat field names declared in HyperCLOVAXVisionV2ProcessorKwargs
        # and its modality-specific sub-TypedDicts (text_kwargs, images_kwargs, …).
        _processor_keys: set = set()
        for _modality_annot in HyperCLOVAXVisionV2ProcessorKwargs.__annotations__.values():
            if hasattr(_modality_annot, "__annotations__"):
                _processor_keys.update(_modality_annot.__annotations__)
        # Also allow passing modality sub-dicts directly (e.g. text_kwargs={…})
        _processor_keys.update(HyperCLOVAXVisionV2ProcessorKwargs.__annotations__)

        call_kwargs = {k: v for k, v in kwargs.items() if k in _processor_keys}
        template_kwargs = {k: v for k, v in kwargs.items() if k not in _processor_keys}

        # Step 1: render chat template → plain text (no media loading).
        # Use tokenizer.apply_chat_template directly so that the full tokenizer
        # context (special tokens, helper variables, etc.) is available to the
        # Jinja template, matching the behaviour users expect.
        # template_kwargs flows through freely — no hard-coded list needed.
        if "use_audio_in_video" not in template_kwargs:
            template_kwargs["use_audio_in_video"] = bool(
                self.video_processor is not None
                and getattr(self.video_processor, "use_audio_in_video", False)
            )
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            chat_template=chat_template,
            tokenize=False,
            **template_kwargs,
        )

        if not tokenize:
            return prompt

        # Step 2: load all multimodal inputs from the conversation
        mm = self.load_multimodal_inputs(conversation)

        # Step 3: call __call__ with text + all modalities.
        out = self(
            text=prompt,
            audios=mm.get("audios"),
            images=mm.get("images"),
            videos=mm.get("videos"),
            video_audios=mm.get("video_audios"),
            sampling_rates=mm.get("sampling_rates"),
            video_sampling_rates=mm.get("video_sampling_rates"),
            video_fps_list=mm.get("video_fps_list"),
            **call_kwargs,
        )

        if return_dict:
            return out
        return out["input_ids"]

    def _load_audio(
        self,
        path: Union[str, bytes, io.BytesIO, np.ndarray],
        sr: int = 16000,
        start: float = 0.0,
        end: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Load an audio clip from a file path, URL, base64 string, bytes, or numpy array.

        Supports the following input formats:
        - ``np.ndarray``: Used directly (multi-channel arrays are averaged to mono).
        - ``bytes`` / ``io.BytesIO``: Written to a temp file, then loaded via ``librosa.load``.
        - Local file path: Loaded via ``librosa.load``.
        - HTTP/HTTPS URL: Downloaded via ``requests.get``, then loaded via ``librosa.load``.
        - Base64 data URI (``data:audio/...;base64,...``): Decoded then loaded.

        Args:
            path: Audio source — file path, URL, base64 data URI, bytes, or numpy array.
            sr: Target sampling rate in Hz.
            start: Start time in seconds for slicing.
            end: End time in seconds for slicing. ``None`` means until the end.

        Returns:
            Dict with keys ``"waveform"`` (1-D float32 numpy array) and ``"sampling_rate"`` (int).
        """
        import librosa

        duration = (end - start) if end is not None else None

        if isinstance(path, np.ndarray):
            audio = path.mean(axis=1) if path.ndim > 1 else path
            start_idx = int(sr * start)
            end_idx = int(sr * end) if end is not None else None
            return {"waveform": audio[start_idx:end_idx], "sampling_rate": sr}

        if isinstance(path, io.BytesIO):
            path = path.getvalue()

        if isinstance(path, bytes):
            # Detect format from magic bytes and write to temp file for librosa
            suffix = _detect_audio_suffix(path)
            with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=True) as fp:
                fp.write(path)
                fp.flush()
                y, _ = librosa.load(fp.name, sr=sr, offset=start, duration=duration, mono=True)
            return {"waveform": y, "sampling_rate": sr}

        # str path
        if path.startswith("data:audio"):
            _, base64_data = path.split("base64,", 1)
            raw = base64.b64decode(base64_data)
            return self._load_audio(raw, sr=sr, start=start, end=end)

        if path.startswith("http://") or path.startswith("https://"):
            response = requests.get(path, timeout=_DEFAULT_REQUEST_TIMEOUT)
            response.raise_for_status()
            return self._load_audio(response.content, sr=sr, start=start, end=end)

        if path.startswith("file://"):
            path = path[len("file://"):]

        # Local file path
        y, _ = librosa.load(path, sr=sr, offset=start, duration=duration, mono=True)
        return {"waveform": y, "sampling_rate": sr}

    def _load_image(
        self,
        path: Union[str, bytes, np.ndarray, "PIL.Image.Image"],
    ) -> Dict[str, Any]:
        """Load an image from a file path, URL, base64 string, bytes, ndarray, or PIL Image.

        Supports the following input formats:
        - ``PIL.Image.Image``: Used directly (converted to RGB if needed).
        - ``np.ndarray``: Converted via ``Image.fromarray``.
        - ``bytes``: Opened via ``Image.open(BytesIO(...))``.
        - Local file path: Opened via ``PIL.Image.open``.
        - HTTP/HTTPS URL: Downloaded via ``requests.get``, then opened.
        - Base64 data URI (``data:image/...;base64,...``): Decoded then opened.

        Args:
            path: Image source.

        Returns:
            Dict with key ``"image"`` (PIL Image in RGB mode).
        """
        if isinstance(path, Image.Image):
            image = path
        elif isinstance(path, np.ndarray):
            image = Image.fromarray(path)
        elif isinstance(path, bytes):
            image = Image.open(io.BytesIO(path))
        elif isinstance(path, str):
            if path.startswith("data:image"):
                _, base64_data = path.split("base64,", 1)
                image = Image.open(io.BytesIO(base64.b64decode(base64_data)))
            elif path.startswith("http://") or path.startswith("https://"):
                response = requests.get(path, timeout=_DEFAULT_REQUEST_TIMEOUT)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
            elif path.startswith("file://"):
                image = Image.open(path[len("file://"):])
            else:  # local file path
                image = Image.open(path)
        else:
            raise TypeError(f"Unsupported image type: {type(path)}")

        if image.mode != "RGB":
            image = image.convert("RGB")
        return {"image": image}

    def _load_video(
        self,
        path: Union[str, bytes, io.BytesIO],
        start: float = 0.0,
        end: Optional[float] = None,
        max_num_frames: Optional[int] = None,
        fps: float = 2.0,
        sr: int = 16000,
        use_audio_in_video: bool = False,
    ) -> Dict[str, Any]:
        """Load video frames (and optionally audio) from a file, URL, bytes, or base64 string.

        Args:
            path: Video source — local file path, HTTP/HTTPS URL, raw bytes, BytesIO,
                or base64 data URI.
            start: Start time in seconds.
            end: End time in seconds. ``None`` means until the end.
            max_num_frames: Maximum number of frames to return.
            fps: Target frame rate for uniform sampling.
            sr: Target audio sampling rate in Hz (used only when ``use_audio_in_video=True``).
            use_audio_in_video: If ``True``, also extract audio from the video stream.

        Returns:
            Dict with keys:
            - ``"frames"``: ``List[PIL.Image.Image]``;
            - ``"audio"``: 1-D float32 ``np.ndarray`` or ``None``;
            - ``"sampling_rate"``: audio sampling rate (int) or ``None``;
            - ``"fps"``: actual frames-per-second at which frames were sampled (float).
        """
        video_source = self._resolve_video_source(path)

        # decord (primary)
        try:
            import decord
            from decord import cpu as decord_cpu

            frames, actual_fps, tmp_path = self._decord_read_frames(
                video_source, start, end, max_num_frames, fps, decord_cpu
            )

            audio = None
            if use_audio_in_video:
                audio = self._decord_read_audio(
                    video_source if tmp_path is None else tmp_path,
                    sr=sr, start=start, end=end,
                )

            if tmp_path is not None:
                os.remove(tmp_path)

            return {
                "frames": frames,
                "audio": audio,
                "sampling_rate": sr if audio is not None else None,
                "fps": actual_fps,
            }

        except ImportError:
            pass

        # torchvision (fallback)
        import torchvision

        tmp_path = None
        if isinstance(video_source, io.BytesIO):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_source.getvalue())
                tmp_path = tmp.name
            video_source = tmp_path

        try:
            video_tensor, _, info = torchvision.io.read_video(
                video_source, start_pts=start, end_pts=end, pts_unit="sec"
            )
            total_frames = video_tensor.shape[0]
            video_fps = info.get("video_fps", 24.0)

            nframes = max(2, round(total_frames / video_fps * fps))
            if max_num_frames is not None:
                nframes = min(nframes, max_num_frames)
            nframes = max(2, nframes - (nframes % 2))
            nframes = min(nframes, total_frames)

            idx = torch.linspace(0, total_frames - 1, nframes).round().long()
            sampled = video_tensor[idx].numpy()
            frames = [Image.fromarray(sampled[i]) for i in range(sampled.shape[0])]

            audio = None
            if use_audio_in_video:
                logger.warning(
                    "Audio extraction is not supported in the torchvision fallback path; "
                    "install decord for full audio-from-video support."
                )

            clip_duration = total_frames / video_fps if video_fps > 0 else 1.0
            actual_fps = len(frames) / clip_duration if clip_duration > 0 else fps
            return {
                "frames": frames,
                "audio": audio,
                "sampling_rate": sr if audio is not None else None,
                "fps": actual_fps,
            }
        finally:
            if tmp_path is not None:
                os.remove(tmp_path)

    # Private helpers for _load_video

    def _resolve_video_source(
        self,
        path: Union[str, bytes, io.BytesIO],
    ) -> Union[str, io.BytesIO]:
        """Resolve raw video input to a file path string or an BytesIO object."""
        if isinstance(path, (bytes, bytearray)):
            return io.BytesIO(path)
        if isinstance(path, io.BytesIO):
            return path
        if isinstance(path, str):
            if path.startswith("data:video"):
                _, base64_data = path.split("base64,", 1)
                return io.BytesIO(base64.b64decode(base64_data))
            if path.startswith("http://") or path.startswith("https://"):
                response = requests.get(path, timeout=(5, 600))
                response.raise_for_status()
                return io.BytesIO(response.content)
            if path.startswith("file://"):
                path = path[len("file://"):]
            # local file path — return as-is
            return path
        raise TypeError(f"Unsupported video source type: {type(path)}")

    @staticmethod
    def _decord_read_frames(
        video_source: Union[str, io.BytesIO],
        start: float,
        end: Optional[float],
        max_num_frames: Optional[int],
        fps: float,
        decord_cpu,
    ) -> Tuple[List[Image.Image], float, Optional[str]]:
        """Read frames using decord. Returns (frames, actual_fps, tmp_path).

        ``actual_fps`` is the sampling rate (frames per second) at which frames were
        selected from the clip.  ``tmp_path`` is set when a BytesIO had to be flushed
        to a temporary file (decord on some platforms cannot read BytesIO directly);
        the caller is responsible for deleting it.
        """
        import decord

        source = video_source
        tmp_path = None

        # decord.VideoReader may not accept BytesIO on all platforms — try directly first
        try:
            vr = decord.VideoReader(source, ctx=decord_cpu(0))
        except Exception:
            if isinstance(source, io.BytesIO):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(source.getvalue())
                    tmp_path = tmp.name
                vr = decord.VideoReader(tmp_path, ctx=decord_cpu(0))
            else:
                raise

        total_frames = len(vr)
        video_fps = vr.get_avg_fps()

        start_frame = int(start * video_fps) if start else 0
        end_frame = int(end * video_fps) if end is not None else total_frames - 1
        end_frame = min(end_frame, total_frames - 1)
        if start_frame >= end_frame:
            start_frame, end_frame = 0, total_frames - 1
        available_frames = end_frame - start_frame + 1

        nframes = max(2, round(available_frames / video_fps * fps))
        if max_num_frames is not None:
            nframes = min(nframes, max_num_frames)
        nframes = max(2, nframes - (nframes % 2))
        nframes = min(nframes, available_frames)

        idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
        frames_np = vr.get_batch(idx).asnumpy()
        del vr

        frames = [Image.fromarray(frames_np[i]) for i in range(frames_np.shape[0])]
        clip_duration = available_frames / video_fps if video_fps > 0 else 1.0
        actual_fps = len(frames) / clip_duration if clip_duration > 0 else fps
        return frames, actual_fps, tmp_path

    @staticmethod
    def _decord_read_audio(
        source: Union[str, io.BytesIO],
        sr: int,
        start: float,
        end: Optional[float],
    ) -> Optional[np.ndarray]:
        """Extract audio from a video file using decord.AudioReader."""
        try:
            from decord import AudioReader
            from decord import cpu as decord_cpu

            # AudioReader requires a file path string
            tmp_path = None
            if isinstance(source, io.BytesIO):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(source.getvalue())
                    tmp_path = tmp.name
                source = tmp_path

            try:
                ar = AudioReader(source, ctx=decord_cpu(0), sample_rate=sr, mono=True)
                total_samples = ar.shape[1]
                start_sample = int(start * sr)
                end_sample = int(end * sr) if end is not None else total_samples
                end_sample = min(end_sample, total_samples)
                audio = ar[start_sample:end_sample].asnumpy().flatten().astype(np.float32)
                return audio
            finally:
                if tmp_path is not None:
                    os.remove(tmp_path)

        except Exception as e:
            logger.warning("Failed to extract audio from video: %s", e)
            return None

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audios: Optional[AudioInput] = None,
        images: Optional[ImageInput] = None,
        videos: Optional[VideoInput] = None,
        video_audios: Optional[AudioInput] = None,
        sampling_rates: Optional[List[int]] = None,
        video_sampling_rates: Optional[List[Optional[int]]] = None,
        video_fps_list: Optional[List[float]] = None,
        **kwargs: Unpack[HyperCLOVAXVisionV2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare text, audio, image, and video inputs for the model. This method forwards `text`
        and `kwargs` to the tokenizer if `text` is not `None`, processes images/videos through their respective
        processors, and handles audio feature extraction.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            HyperCLOVAXVisionV2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # text processing
        if text is not None:
            if isinstance(text, str):
                text = [text]
            text = copy.deepcopy(text)

        # audio processing
        audio_inputs = dict()
        discrete_audio_inputs = dict()
        if (
            audios is not None
            and self.audio_processor is not None
        ):
            # Normalize to List[waveform]
            if isinstance(audios, np.ndarray):
                # (T,) single → [arr];  (B,T) batch → list of 1-D arrays
                audios = [audios] if audios.ndim == 1 else list(audios)
            elif isinstance(audios, torch.Tensor):
                audios = [audios] if audios.ndim == 1 else list(audios.unbind(0))

            _no_concat_keys = {"num_audio_tokens", "num_discrete_audio_tokens"}
            for _audio in audios:
                # Normalize each element to a list accepted by audio_processor
                if isinstance(_audio, np.ndarray):
                    _audio = [_audio]
                elif isinstance(_audio, torch.Tensor):
                    _audio = [_audio]

                _audio_features = self.audio_processor(
                    audios=_audio,
                    **output_kwargs.get("audio_kwargs", {}),
                )
                for _k, _v in _audio_features.items():
                    if _k in [
                        "discrete_audio_values",
                        "num_discrete_audio_tokens",
                    ]:
                        if _k not in discrete_audio_inputs:
                            discrete_audio_inputs[_k] = list()
                        discrete_audio_inputs[_k].append(_v)
                    else:
                        if _k not in audio_inputs:
                            audio_inputs[_k] = list()
                        audio_inputs[_k].append(_v)
            audio_inputs = {
                _k: torch.cat(_v, dim=0)
                if isinstance(_v[0], torch.Tensor) and _k not in _no_concat_keys else _v
                for _k, _v in audio_inputs.items()
            }
            if discrete_audio_inputs:
                discrete_audio_inputs = {
                    _k: torch.cat(_v, dim=0)
                    if isinstance(_v[0], torch.Tensor) and _k not in _no_concat_keys else _v
                    for _k, _v in discrete_audio_inputs.items()
                }

        # image processing
        image_inputs = dict()
        discrete_image_inputs = dict()
        if (
            images is not None
            and self.image_processor is not None
        ):
            # Normalize to List[image]
            if isinstance(images, PIL.Image.Image):
                images = [images]
            elif isinstance(images, np.ndarray):
                # (H,W,C) single → [arr];  (B,H,W,C) batch → list of arrays
                images = [images] if images.ndim == 3 else list(images)
            elif isinstance(images, torch.Tensor):
                images = [images] if images.ndim == 3 else list(images.unbind(0))

            _no_concat_keys = {"num_image_tokens", "num_discrete_image_tokens"}
            for _image in images:
                # Normalize each element to a list accepted by image_processor
                if isinstance(_image, PIL.Image.Image):
                    _image = [_image]
                elif isinstance(_image, np.ndarray):
                    _image = [_image]
                elif isinstance(_image, torch.Tensor):
                    _image = [_image]

                _image_features = self.image_processor(
                    images=_image,
                    **output_kwargs.get("images_kwargs", {}),
                )
                for _k, _v in _image_features.items():
                    if _k in [
                        "discrete_pixel_values",
                        "discrete_image_ratios",
                        "num_discrete_image_tokens",
                    ]:
                        if _k not in discrete_image_inputs:
                            discrete_image_inputs[_k] = list()
                        discrete_image_inputs[_k].append(_v)
                    else:
                        if _k not in image_inputs:
                            image_inputs[_k] = list()
                        image_inputs[_k].append(_v)
            image_inputs = {
                _k: torch.cat(_v, dim=0)
                if isinstance(_v[0], torch.Tensor) and _k not in _no_concat_keys else _v
                for _k, _v in image_inputs.items()
            }
            if discrete_image_inputs:
                discrete_image_inputs = {
                    _k: torch.cat(_v, dim=0)
                    if isinstance(_v[0], torch.Tensor) and _k not in _no_concat_keys else _v
                    for _k, _v in discrete_image_inputs.items()
                }

        # video processing
        video_inputs = dict()
        if (
            videos is not None
            and self.video_processor is not None
        ):
            # Normalize to List[video]
            if isinstance(videos, np.ndarray):
                # (T,H,W,C) single → [(T,H,W,C)];  (B,T,H,W,C) batch → list of (T,H,W,C)
                videos = [videos] if videos.ndim == 4 else list(videos)
            elif isinstance(videos, torch.Tensor):
                videos = [videos] if videos.ndim == 4 else list(videos.unbind(0))
            elif isinstance(videos, (list, tuple)) and len(videos) > 0:
                # List[PIL.Image] = single video given as a flat frame list
                if isinstance(videos[0], Image.Image):
                    videos = [list(videos)]

            _no_concat_keys = {
                "num_video_tokens", "num_discrete_video_tokens",
                "num_video_audio_tokens", "num_discrete_video_audio_tokens"
            }
            for _video in videos:
                # Normalize each video to List[np.ndarray (T,H,W,C)] expected by video_processor
                if isinstance(_video, (list, tuple)) and len(_video) > 0 and isinstance(_video[0], Image.Image):
                    # List[PIL.Image] frames → (T,H,W,C) numpy array
                    _video = [np.stack([np.array(f) for f in _video], axis=0)]
                elif isinstance(_video, np.ndarray) and _video.ndim == 4:
                    _video = [_video]
                elif isinstance(_video, torch.Tensor) and _video.ndim == 4:
                    _video = [_video]

                _video_features = self.video_processor(
                    videos=_video,
                    **output_kwargs.get("videos_kwargs", {}),
                )
                for _k, _v in _video_features.items():
                    if _k not in video_inputs:
                        video_inputs[_k] = list()
                    video_inputs[_k].append(_v)

            if (
                self.video_processor.use_audio_in_video
                and isinstance(video_audios, (list, tuple))
                and len(video_audios) == len(videos)
                and self.audio_processor is not None
            ):
                for _video_audio in video_audios:
                    if _video_audio is None:
                        continue
                    if (
                        not isinstance(_video_audio, (list, tuple))
                        and isinstance(_video_audio, np.ndarray)
                    ):
                        _video_audio = [_video_audio]

                    _video_audio_features = self.audio_processor(
                        audios=_video_audio,
                        prefix="video_",
                        **output_kwargs.get("audio_kwargs", {}),
                    )
                    for _k, _v in _video_audio_features.items():
                        if _k not in video_inputs:
                            video_inputs[_k] = list()
                        video_inputs[_k].append(_v)

            video_inputs = {
                _k: torch.cat(_v, dim=0)
                if isinstance(_v[0], torch.Tensor) and _k not in _no_concat_keys else _v
                for _k, _v in video_inputs.items()
            }

        # replace <|audio_duration|> placeholders with actual durations
        if (
            text is not None
            and audios is not None
        ):
            sr = 16000
            if (
                self.audio_processor is not None
                and isinstance(getattr(self.audio_processor, "sampling_rate", None), int)
            ):
                sr = self.audio_processor.sampling_rate
            # audios can be [batch][audio_idx] or [audio_idx] format
            flat_audios = audios
            if len(audios) > 0 and isinstance(audios[0], list):
                flat_audios = [a for batch in audios for a in batch]
            audio_dur_idx = 0
            for _batch_idx, _text in enumerate(text):
                while "<|audio_duration|>" in _text and audio_dur_idx < len(flat_audios):
                    audio_data = flat_audios[audio_dur_idx]
                    duration_sec = len(audio_data) / sr
                    # Add quotes around duration to maintain valid JSON format
                    _text = _text.replace("<|audio_duration|>", f'"{duration_sec:.2f}s"', 1)
                    audio_dur_idx += 1
                text[_batch_idx] = _text

        # replace <|video_duration|> placeholders with actual durations
        if (
            text is not None
            and videos is not None
        ):
            # videos is in [batch] format, each batch is a list of PIL images
            fps = output_kwargs.get("videos_kwargs", {}).get("fps", 2.0)
            flat_videos = videos
            if (
                len(videos) > 0
                and isinstance(videos[0], list)
                and len(videos[0]) > 0
                and isinstance(videos[0][0], list)
            ):
                flat_videos = [v for batch in videos for v in batch]
            video_dur_idx = 0
            for _batch_idx, _text in enumerate(text):
                while "<|video_duration|>" in _text and video_dur_idx < len(flat_videos):
                    video_frames = flat_videos[video_dur_idx]
                    num_frames = len(video_frames) if isinstance(video_frames, list) else video_frames.shape[0]
                    duration_sec = round(num_frames / fps, 2)
                    _text = _text.replace("<|video_duration|>", f"{duration_sec}s", 1)
                    video_dur_idx += 1
                text[_batch_idx] = _text

        # expand discrete audio placeholders
        if (
            text is not None
            and discrete_audio_inputs
            and self.audio_processor is not None
            and self.audio_processor.use_discrete_token
        ):
            for _batch_idx, (_text_before, _num_discrete_audio_tokens) in enumerate(
                zip(text, discrete_audio_inputs["num_discrete_audio_tokens"])
            ):
                discrete_audio_block_pattern = (
                    re.escape(self.audio_processor.discrete_audio_start_token)
                    + r".*?"
                    + re.escape(self.audio_processor.discrete_audio_token)
                    + r".*?"
                    + re.escape(self.audio_processor.discrete_audio_end_token)
                )
                _find_iters = list(re.finditer(discrete_audio_block_pattern, _text_before))
                if len(_find_iters) > 0:
                    _text_after = ""
                    _prev_end_idx = 0
                    for _sample_idx, _discrete_audio_match in enumerate(_find_iters):
                        _inplace_str = self.get_audio_token_replacement(
                            num_audio_tokens=None,
                            num_discrete_audio_tokens=_num_discrete_audio_tokens[_sample_idx],
                            include_boundary_tokens=True,
                            tokenize=False,
                        )
                        _text_after += _text_before[_prev_end_idx : _discrete_audio_match.start()]
                        _text_after += _inplace_str
                        _prev_end_idx = _discrete_audio_match.end()
                    _text_after += _text_before[_prev_end_idx:]
                    text[_batch_idx] = _text_after

        # expand continuous audio placeholders
        if (
            text is not None
            and audio_inputs
            and self.audio_processor is not None
        ):
            for _batch_idx, (_text_before, _num_audio_tokens) in enumerate(
                zip(text, audio_inputs["num_audio_tokens"])
            ):
                cont_audio_block_pattern = (
                    re.escape(self.audio_processor.audio_start_token)
                    + r".*?"
                    + re.escape(self.audio_processor.audio_token)
                    + r".*?"
                    + re.escape(self.audio_processor.audio_end_token)
                )
                _find_iters = list(re.finditer(cont_audio_block_pattern, _text_before))
                if len(_find_iters) > 0:
                    _text_after = ""
                    _prev_end_idx = 0
                    for _sample_idx, _continuous_audio_match in enumerate(_find_iters):
                        _inplace_str = self.get_audio_token_replacement(
                            num_audio_tokens=_num_audio_tokens[_sample_idx],
                            num_discrete_audio_tokens=None,
                            include_boundary_tokens=True,
                            tokenize=False,
                        )
                        _text_after += _text_before[_prev_end_idx : _continuous_audio_match.start()]
                        _text_after += _inplace_str
                        _prev_end_idx = _continuous_audio_match.end()
                    _text_after += _text_before[_prev_end_idx:]
                    text[_batch_idx] = _text_after

        # expand discrete image placeholders
        if (
            text is not None
            and discrete_image_inputs
            and self.image_processor is not None
            and self.image_processor.use_discrete_token
        ):
            _item_idx = 0
            for _batch_idx, (_text_before, _num_discrete_image_tokens) in enumerate(
                zip(text, discrete_image_inputs["num_discrete_image_tokens"])
            ):
                discrete_image_block_pattern = (
                    re.escape(self.image_processor.discrete_image_start_token)
                    + r".*?"
                    + re.escape(self.image_processor.discrete_image_token)
                    + r".*?"
                    + re.escape(self.image_processor.discrete_image_end_token)
                )
                _find_iters = list(re.finditer(discrete_image_block_pattern, _text_before))
                if len(_find_iters) > 0:
                    _text_after = ""
                    _prev_end_idx = 0
                    for _sample_idx, _discrete_image_match in enumerate(_find_iters):
                        _inplace_str = self.get_image_token_replacement(
                            num_image_tokens=None,
                            num_discrete_image_tokens=_num_discrete_image_tokens[_sample_idx],
                            discrete_image_ratio=discrete_image_inputs["discrete_image_ratios"][_item_idx],
                            include_boundary_tokens=True,
                            tokenize=False,
                        )
                        _text_after += _text_before[_prev_end_idx : _discrete_image_match.start()]
                        _text_after += _inplace_str
                        _prev_end_idx = _discrete_image_match.end()
                        _item_idx += 1
                    _text_after += _text_before[_prev_end_idx:]
                    text[_batch_idx] = _text_after

        # expand continuous image placeholders
        if (
            text is not None
            and image_inputs
            and self.image_processor is not None
        ):
            for _batch_idx, (_text_before, _num_image_tokens) in enumerate(
                zip(text, image_inputs["num_image_tokens"])
            ):
                cont_image_block_pattern = (
                    re.escape(self.image_processor.image_start_token)
                    + r".*?"
                    + re.escape(self.image_processor.image_token)
                    + r".*?"
                    + re.escape(self.image_processor.image_end_token)
                )
                _find_iters = list(re.finditer(cont_image_block_pattern, _text_before))
                if len(_find_iters) > 0:
                    _text_after = ""
                    _prev_end_idx = 0
                    for _sample_idx, _continuous_image_match in enumerate(_find_iters):
                        _inplace_str = self.get_image_token_replacement(
                            num_image_tokens=_num_image_tokens[_sample_idx],
                            num_discrete_image_tokens=None,
                            discrete_image_ratio=None,
                            include_boundary_tokens=True,
                            tokenize=False,
                        )
                        _text_after += _text_before[_prev_end_idx : _continuous_image_match.start()]
                        _text_after += _inplace_str
                        _prev_end_idx = _continuous_image_match.end()
                    _text_after += _text_before[_prev_end_idx:]
                    text[_batch_idx] = _text_after

        # expand video placeholders
        if (
            text is not None
            and video_inputs
            and self.video_processor is not None
        ):
            _use_video_audio = getattr(self.video_processor, "use_audio_in_video", False)
            _num_video_audio_tokens_all = (
                video_inputs.get("num_video_audio_tokens")
                if _use_video_audio else None
            )

            if _use_video_audio:
                # Template: <|video_start|><|VIDEO_PAD|><|VIDEO_AUDIO_PAD|><|video_end|>
                video_block_pattern = (
                    re.escape(self.video_processor.video_start_token)
                    + r".*?"
                    + re.escape(self.video_processor.video_token)
                    + r".*?"
                    + re.escape(self.video_audio_token)
                    + r".*?"
                    + re.escape(self.video_processor.video_end_token)
                )
            else:
                # Template: <|video_start|><|VIDEO_PAD|><|video_end|>
                video_block_pattern = (
                    re.escape(self.video_processor.video_start_token)
                    + r".*?"
                    + re.escape(self.video_processor.video_token)
                    + r".*?"
                    + re.escape(self.video_processor.video_end_token)
                )

            for _batch_idx, (_text_before, _num_video_tokens) in enumerate(zip(
                text, video_inputs["num_video_tokens"],
            )):
                _num_va_per_batch = (
                    _num_video_audio_tokens_all[_batch_idx]
                    if _num_video_audio_tokens_all is not None
                    else None
                )
                _find_iters = list(re.finditer(video_block_pattern, _text_before))
                if len(_find_iters) > 0:
                    _text_after = ""
                    _prev_end_idx = 0
                    for _sample_idx, _continuous_video_match in enumerate(_find_iters):
                        _num_va = (
                            _num_va_per_batch[_sample_idx]
                            if _num_va_per_batch is not None
                            else None
                        )
                        _inplace_str = self.get_video_token_replacement(
                            num_video_tokens=_num_video_tokens[_sample_idx],
                            num_video_audio_tokens=_num_va,
                            include_boundary_tokens=True,
                            tokenize=False,
                        )
                        _text_after += _text_before[_prev_end_idx:_continuous_video_match.start()]
                        _text_after += _inplace_str
                        _prev_end_idx = _continuous_video_match.end()
                    _text_after += _text_before[_prev_end_idx:]
                    text[_batch_idx] = _text_after

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)

        text_inputs = dict()
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                **output_kwargs["text_kwargs"],
                return_tensors=None,
            )
            self._check_special_mm_tokens(
                text,
                text_inputs,
                modalities=self.modalities,
            )

        if (
            return_mm_token_type_ids
            and hasattr(self, "image_token_id")
        ):
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1

        data = {
            **audio_inputs,
            **image_inputs,
            **text_inputs,
            **video_inputs,
        }
        if (
            discrete_audio_inputs
            and self.audio_processor is not None
            and self.audio_processor.use_discrete_token
        ):
            data.update(discrete_audio_inputs)
        if (
            discrete_image_inputs
            and self.image_processor is not None
            and self.image_processor.use_discrete_token
        ):
            data.update(discrete_image_inputs)

        # Drop processing-only token counts: they are consumed above for placeholder
        # expansion and are not model forward inputs (would trip generate()'s kwarg
        # validation). Downstream tools can derive counts from `*_grid_thw`.
        for _count_key in _PLACEHOLDER_COUNT_KEYS:
            data.pop(_count_key, None)

        model_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        return model_inputs

    def get_audio_placeholder(
        self,
        tokenize: bool = False,
        include_boundary_tokens: bool = True,
    ) -> Union[str, List[int]]:
        """Build the audio placeholder string (or token ids) for a single audio input.

        Includes discrete audio tokens (if enabled) followed by continuous audio tokens,
        each wrapped in their respective start/end boundary tokens.

        Args:
            tokenize: If ``True``, return token ids instead of the raw string.

        Returns:
            Placeholder string, or list of token ids when ``tokenize`` is ``True``.
        """
        audio_placeholder = ""
        if self.audio_processor is None:
            if tokenize:
                return list()
            else:
                return audio_placeholder

        if self.audio_processor.use_discrete_token:
            _discrete_audio_placeholder = f'{self.audio_processor.discrete_audio_token}'
            if include_boundary_tokens:
                _discrete_audio_placeholder = f'{self.audio_processor.discrete_audio_start_token}{_discrete_audio_placeholder}{self.audio_processor.discrete_audio_end_token}'
            audio_placeholder += f'{_discrete_audio_placeholder}\n'

        _continuous_audio_placeholder = f'{self.audio_processor.audio_token}'
        if include_boundary_tokens:
            _continuous_audio_placeholder = f'{self.audio_processor.audio_start_token}{_continuous_audio_placeholder}{self.audio_processor.audio_end_token}'
        audio_placeholder += f'{_continuous_audio_placeholder}'

        if tokenize:
            audio_placeholder = self.tokenizer.encode(audio_placeholder)
        return audio_placeholder

    def get_audio_token_replacement(
        self,
        num_audio_tokens: Optional[Union[int, List[int], Tuple[int, ...], torch.Tensor]] = None,
        num_discrete_audio_tokens: Optional[Union[int, List[int], Tuple[int, ...], torch.Tensor]] = None,
        include_boundary_tokens: bool = False,
        tokenize: bool = False,
        audio_token: str = None,
        discrete_audio_token: str = None,
        return_tuple: Optional[bool] = None,
    ) -> Union[str, List[int], Tuple[str, str], Tuple[List[int], List[int]]]:
        """Build a replacement string (or token ids) for audio placeholder tokens.

        Expands the placeholder into the correct number of continuous and/or
        discrete audio tokens, optionally wrapped with boundary tokens.

        Args:
            num_audio_tokens: Number of continuous audio tokens (or a 1-element
                list/tensor). ``None`` to skip continuous replacement.
            num_discrete_audio_tokens: Number of discrete audio tokens (or a
                1-element list/tensor). ``None`` to skip discrete replacement.
            include_boundary_tokens: Whether to wrap with start/end tokens.
            tokenize: If ``True``, return token ids instead of the raw string.
            return_tuple: If ``True``, return ``(continuous, discrete)`` tuple.

        Returns:
            Replacement string, token id list, or a tuple of two depending on
            ``tokenize`` and ``return_tuple``.
        """
        if not audio_token:
            audio_token = self.audio_processor.audio_token
        if not discrete_audio_token:
            discrete_audio_token = self.audio_processor.discrete_audio_token

        continuous_replacement, discrete_replacement = "", ""
        if self.audio_processor is None:
            if return_tuple:
                return (continuous_replacement, discrete_replacement)
            else:
                return ""

        if num_audio_tokens is not None:
            if (
                isinstance(num_audio_tokens, (list, tuple))
                or (isinstance(num_audio_tokens, torch.Tensor) and num_audio_tokens.dim() >= 1)
            ):
                num_audio_tokens = num_audio_tokens[0]
            continuous_replacement = audio_token * num_audio_tokens
            if include_boundary_tokens:
                continuous_replacement = f"{self.audio_processor.audio_start_token}{continuous_replacement}{self.audio_processor.audio_end_token}"

        if (
            num_discrete_audio_tokens is not None
            and self.audio_processor.use_discrete_token
        ):
            if (
                isinstance(num_discrete_audio_tokens, (list, tuple))
                or (isinstance(num_discrete_audio_tokens, torch.Tensor) and num_discrete_audio_tokens.dim() >= 1)
            ):
                num_discrete_audio_tokens = num_discrete_audio_tokens[0]
            discrete_replacement = discrete_audio_token * num_discrete_audio_tokens
            if include_boundary_tokens:
                discrete_replacement = f"{self.audio_processor.discrete_audio_start_token}{discrete_replacement}{self.audio_processor.discrete_audio_end_token}"
            discrete_replacement = f'{discrete_replacement}\n'

        if return_tuple:
            if tokenize:
                continuous_replacement = self.tokenizer.encode(continuous_replacement)
                discrete_replacement = self.tokenizer.encode(discrete_replacement)
            return (continuous_replacement, discrete_replacement)
        else:
            replacement = f'{discrete_replacement}{continuous_replacement}'
            if tokenize:
                replacement = self.tokenizer.encode(replacement)
            return replacement

    def get_image_placeholder(
        self,
        tokenize: bool = False,
        include_boundary_tokens: bool = True,
    ) -> Union[str, List[int]]:
        """Build the image placeholder string (or token ids) for a single image input.

        Includes discrete image tokens (if enabled) followed by continuous image tokens,
        each wrapped in their respective start/end boundary tokens.

        Args:
            tokenize: If ``True``, return token ids instead of the raw string.

        Returns:
            Placeholder string, or list of token ids when ``tokenize`` is ``True``.
        """
        image_placeholder = ""
        if self.image_processor is None:
            if tokenize:
                return list()
            else:
                return image_placeholder

        if self.image_processor.use_discrete_token:
            _discrete_image_placeholder = f'{self.image_processor.discrete_image_token}'
            if include_boundary_tokens:
                _discrete_image_placeholder = f'{self.image_processor.discrete_image_start_token}{_discrete_image_placeholder}{self.image_processor.discrete_image_end_token}'
            image_placeholder += f'{_discrete_image_placeholder}\n'

        _continuous_image_placeholder = f'{self.image_processor.image_token}'
        if include_boundary_tokens:
            _continuous_image_placeholder = f'{self.image_processor.image_start_token}{_continuous_image_placeholder}{self.image_processor.image_end_token}'
        image_placeholder += f'{_continuous_image_placeholder}'

        if tokenize:
            image_placeholder = self.tokenizer.encode(image_placeholder)
        return image_placeholder

    def get_image_token_replacement(
        self,
        num_image_tokens: Optional[Union[int, List[int], Tuple[int, ...], torch.Tensor]] = None,
        num_discrete_image_tokens: Optional[Union[int, List[int], Tuple[int, ...], torch.Tensor]] = None,
        discrete_image_ratio: Optional[Union[List[int], Tuple[int, ...], torch.Tensor]] = None,
        include_boundary_tokens: bool = False,
        tokenize: bool = False,
        return_tuple: Optional[bool] = None,
    ) -> Union[str, List[int], Tuple[str, str], Tuple[List[int], List[int]]]:
        """Build a replacement string (or token ids) for image placeholder tokens.

        Expands the placeholder into the correct number of continuous and/or
        discrete image tokens, optionally prefixed with a ratio token and
        wrapped with boundary tokens.

        Args:
            num_image_tokens: Number of continuous image tokens (or a 1-element
                list/tensor). ``None`` to skip continuous replacement.
            num_discrete_image_tokens: Number of discrete image tokens (or a
                1-element list/tensor). ``None`` to skip discrete replacement.
            discrete_image_ratio: Aspect ratio ``[h, w]`` for the discrete
                image ratio token. ``None`` to omit the ratio prefix.
            include_boundary_tokens: Whether to wrap with start/end tokens.
            tokenize: If ``True``, return token ids instead of the raw string.
            return_tuple: If ``True``, return ``(continuous, discrete)`` tuple.

        Returns:
            Replacement string, token id list, or a tuple of two depending on
            ``tokenize`` and ``return_tuple``.
        """
        continuous_replacement, discrete_replacement = "", ""
        if self.image_processor is None:
            if return_tuple:
                return (continuous_replacement, discrete_replacement)
            else:
                return ""

        if num_image_tokens is not None:
            if (
                isinstance(num_image_tokens, (list, tuple))
                or (isinstance(num_image_tokens, torch.Tensor) and num_image_tokens.dim() >= 1)
            ):
                num_image_tokens = num_image_tokens[0]
            continuous_replacement = self.image_processor.image_token * num_image_tokens
            if include_boundary_tokens:
                continuous_replacement = f"{self.image_processor.image_start_token}{continuous_replacement}{self.image_processor.image_end_token}"

        if (
            num_discrete_image_tokens is not None
            and self.image_processor.use_discrete_token
        ):
            if (
                isinstance(discrete_image_ratio, (list, tuple))
                or (isinstance(discrete_image_ratio, torch.Tensor) and discrete_image_ratio.dim() >= 2)
            ) and len(discrete_image_ratio) == 1: # [[16, 9]], or torch.Tensor([[16, 9]])
                discrete_image_ratio = discrete_image_ratio[0]

            row_str = self.image_processor.discrete_image_token * self.image_processor.discrete_token_size
            discrete_replacement = row_str * self.image_processor.discrete_token_size
            if discrete_image_ratio is not None:
                if isinstance(discrete_image_ratio, (list, tuple)):
                    ratio_key = f"{int(discrete_image_ratio[0])}:{int(discrete_image_ratio[1])}"
                elif isinstance(discrete_image_ratio, torch.Tensor):
                    ratio_key = f"{discrete_image_ratio[0].item()}:{discrete_image_ratio[1].item()}"
                discrete_image_ratio_token = self.image_processor.discrete_image_ratio_tokens[ratio_key]
                discrete_replacement = f"{discrete_image_ratio_token}{discrete_replacement}"
            if include_boundary_tokens:
                discrete_replacement = f"{self.image_processor.discrete_image_start_token}{discrete_replacement}{self.image_processor.discrete_image_end_token}"
            discrete_replacement = f'{discrete_replacement}\n'

        if return_tuple:
            if tokenize:
                continuous_replacement = self.tokenizer.encode(continuous_replacement)
                discrete_replacement = self.tokenizer.encode(discrete_replacement)
            return (continuous_replacement, discrete_replacement)
        else:
            replacement = f'{discrete_replacement}{continuous_replacement}'
            if tokenize:
                replacement = self.tokenizer.encode(replacement)
            return replacement

    def get_video_placeholder(
        self,
        tokenize: bool = False,
        include_boundary_tokens: bool = True,
    ) -> Union[str, List[int]]:
        """Build the video placeholder string (or token ids) for a single video input.

        The placeholder consists of continuous video tokens wrapped in start/end
        boundary tokens.

        Args:
            tokenize: If ``True``, return token ids instead of the raw string.

        Returns:
            Placeholder string, or list of token ids when ``tokenize`` is ``True``.
        """
        video_placeholder = ""
        if self.video_processor is None:
            if tokenize:
                return list()
            else:
                return video_placeholder

        _continuous_video_placeholder = f'{self.video_processor.video_token}'
        if include_boundary_tokens:
            _continuous_video_placeholder = f'{self.video_processor.video_start_token}{_continuous_video_placeholder}{self.video_processor.video_end_token}'
        video_placeholder += f'{_continuous_video_placeholder}'

        if tokenize:
            video_placeholder = self.tokenizer.encode(video_placeholder)
        return video_placeholder

    def get_video_audio_placeholder(
        self,
        tokenize: bool = False,
        include_boundary_tokens: bool = False,
    ) -> Union[str, List[int]]:
        """Build the video-audio placeholder string (or token ids) for a single video-audio input.

        The placeholder consists of a single video-audio token, optionally wrapped
        in start/end boundary tokens.

        Args:
            tokenize: If ``True``, return token ids instead of the raw string.

        Returns:
            Placeholder string, or list of token ids when ``tokenize`` is ``True``.
        """
        video_audio_placeholder = ""
        if (
            self.video_processor is None
            or self.audio_processor is None
        ):
            if tokenize:
                return list()
            else:
                return video_audio_placeholder

        # No start/end tokens: video_audio_placeholder is embedded within video_placeholder.
        _continuous_video_audio_placeholder = f'{self.video_processor.video_audio_token}'
        if include_boundary_tokens:
            _continuous_video_audio_placeholder = f'{self.video_processor.video_audio_start_token}{_continuous_video_audio_placeholder}{self.video_processor.video_audio_end_token}'
        video_audio_placeholder += f'{_continuous_video_audio_placeholder}'

        if tokenize:
            video_audio_placeholder = self.tokenizer.encode(video_audio_placeholder)
        return video_audio_placeholder

    def get_video_token_replacement(
        self,
        num_video_tokens: Optional[Union[int, List[int], Tuple[int, ...], torch.Tensor]] = None,
        num_video_audio_tokens: Optional[Union[int, List[int], Tuple[int, ...], torch.Tensor]] = None,
        include_boundary_tokens: bool = False,
        tokenize: bool = False,
        return_tuple: Optional[bool] = None,
    ) -> Union[str, List[int], Tuple[str, str], Tuple[List[int], List[int]]]:
        """Build a replacement string (or token ids) for video placeholder tokens.

        Expands the placeholder into the correct number of continuous video
        tokens, optionally followed by video-audio tokens, all wrapped with
        boundary tokens.

        Args:
            num_video_tokens: Number of continuous video tokens (or a 1-element
                list/tensor). ``None`` to skip replacement.
            num_video_audio_tokens: Number of video-audio tokens to append after
                the video tokens. ``None`` or ``0`` to omit.
            include_boundary_tokens: Whether to wrap with start/end tokens.
            tokenize: If ``True``, return token ids instead of the raw string.
            return_tuple: If ``True``, return ``(continuous, discrete)`` tuple.

        Returns:
            Replacement string, token id list, or a tuple of two depending on
            ``tokenize`` and ``return_tuple``.
        """
        continuous_replacement, discrete_replacement = "", ""
        if self.video_processor is None:
            if return_tuple:
                return (continuous_replacement, discrete_replacement)
            else:
                return ""

        if num_video_tokens is not None:
            if (
                isinstance(num_video_tokens, (list, tuple))
                or (isinstance(num_video_tokens, torch.Tensor) and num_video_tokens.dim() >= 1)
            ):
                num_video_tokens = num_video_tokens[0]
            continuous_replacement = self.video_processor.video_token * int(num_video_tokens)

            if num_video_audio_tokens is not None:
                if (
                    isinstance(num_video_audio_tokens, (list, tuple))
                    or (isinstance(num_video_audio_tokens, torch.Tensor) and num_video_audio_tokens.dim() >= 1)
                ):
                    num_video_audio_tokens = num_video_audio_tokens[0]
                _n_va = int(num_video_audio_tokens)
                if _n_va > 0:
                    continuous_replacement += self.video_audio_token * _n_va

            if include_boundary_tokens:
                continuous_replacement = (
                    f"{self.video_processor.video_start_token}"
                    f"{continuous_replacement}"
                    f"{self.video_processor.video_end_token}"
                )

        if return_tuple:
            if tokenize:
                continuous_replacement = self.tokenizer.encode(continuous_replacement)
                discrete_replacement = self.tokenizer.encode(discrete_replacement)
            return (continuous_replacement, discrete_replacement)
        else:
            replacement = f'{discrete_replacement}{continuous_replacement}'
            if tokenize:
                replacement = self.tokenizer.encode(replacement)
            return replacement


__all__ = [
    "HyperCLOVAXVisionV2AudioKwargs",
    "HyperCLOVAXVisionV2TextKwargs",
    "HyperCLOVAXVisionV2VideosKwargs",
    "HyperCLOVAXVisionV2ProcessorKwargs",
    "HyperCLOVAXVisionV2Processor",
]
