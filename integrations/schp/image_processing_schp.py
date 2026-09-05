"""
SCHPImageProcessor — preprocessing for SCHPForSemanticSegmentation.

Resizes images to the model's expected input size and normalises with the
SCHP BGR-indexed mean/std convention (channels are RGB in the tensor but
the normalisation constants come from a BGR-trained ResNet-101).
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import BaseImageProcessor
from transformers.image_processing_utils import BatchFeature


class SCHPImageProcessor(BaseImageProcessor):
    """
    Image processor for SCHP (Self-Correction Human Parsing).

    Args:
        size (`dict`, *optional*, defaults to ``{"height": 473, "width": 473}``):
            Resize target. The LIP model was trained at 473×473.
        image_mean (`list[float]`):
            Per-channel mean in **RGB channel order** using BGR-indexed values:
            ``[0.406, 0.456, 0.485]``.
        image_std (`list[float]`):
            Per-channel std  in **RGB channel order** using BGR-indexed values:
            ``[0.225, 0.224, 0.229]``.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size or {"height": 473, "width": 473}
        # BGR-indexed normalisation constants used during SCHP training
        self.image_mean = image_mean or [0.406, 0.456, 0.485]
        self.image_std = image_std or [0.225, 0.224, 0.229]

    def preprocess(
        self,
        images: Union[
            Image.Image,
            np.ndarray,
            torch.Tensor,
            List[Union[Image.Image, np.ndarray, torch.Tensor]],
        ],
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ) -> BatchFeature:
        """
        Pre-process one or more images.

        Returns a :class:`BatchFeature` with a ``pixel_values`` key of shape
        ``(batch, 3, H, W)`` as a ``torch.Tensor`` (when ``return_tensors="pt"``).
        """
        if not isinstance(images, (list, tuple)):
            images = [images]

        h = self.size["height"]
        w = self.size["width"]
        mean = self.image_mean
        std = self.image_std

        tensors = []
        for img in images:
            # --- normalise input type to PIL RGB ---
            pil: Image.Image
            if isinstance(img, torch.Tensor):
                # (C, H, W) float tensor in [0, 1]
                pil = TF.to_pil_image(img.cpu())
            elif isinstance(img, np.ndarray):
                pil = Image.fromarray(np.asarray(img, dtype=np.uint8))
            else:
                assert isinstance(img, Image.Image)
                pil = img
            pil = pil.convert("RGB")

            # --- resize → tensor → normalise ---
            pil = pil.resize((w, h), resample=Image.Resampling.BILINEAR)
            t = TF.to_tensor(pil)  # float32 in [0, 1], shape (3, H, W)
            t = TF.normalize(t, mean=mean, std=std)
            tensors.append(t)

        pixel_values = torch.stack(tensors)  # (B, 3, H, W)
        return BatchFeature({"pixel_values": pixel_values}, tensor_type=return_tensors)
