"""
SCHP (Self-Correction Human Parsing) — Transformers-compatible implementation.

Architecture inlined from https://github.com/GoGoDuck912/Self-Correction-Human-Parsing
(networks/AugmentCE2P.py) with the CUDA-only InPlaceABNSync replaced by a pure-PyTorch
drop-in, making the model fully runnable on CPU.
"""

import functools
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_schp import SCHPConfig


# ── Pure-PyTorch InPlaceABNSync shim ──────────────────────────────────────────
class InPlaceABNSync(nn.BatchNorm2d):
    """CPU-compatible drop-in for InPlaceABNSync.

    Subclasses ``nn.BatchNorm2d`` directly so that state-dict keys
    (weight, bias, running_mean, running_var) match the original SCHP
    checkpoints without any nesting.
    """

    def __init__(self, num_features, activation="leaky_relu", slope=0.01, **kwargs):
        bn_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ("eps", "momentum", "affine", "track_running_stats")
        }
        super().__init__(num_features, **bn_kwargs)
        self.activation = activation
        self.slope = slope

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        input = super().forward(input)
        if self.activation == "leaky_relu":
            return F.leaky_relu(input, negative_slope=self.slope, inplace=True)
        elif self.activation == "elu":
            return F.elu(input, inplace=True)
        return input


# BatchNorm2d with no activation (activation="none")
BatchNorm2d = functools.partial(InPlaceABNSync, activation="none")
affine_par = True


# ── Model architecture (inlined from AugmentCE2P.py) ─────────────────────────
def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation * multi_grid,
            dilation=dilation * multi_grid,
            bias=False,
        )
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu_inplace(out + residual)


class _PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(features, out_features, kernel_size=1, bias=False),
                    InPlaceABNSync(out_features),
                )
                for size in sizes
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                features + len(sizes) * out_features,
                out_features,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            InPlaceABNSync(out_features),
        )

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(
                stage(feats), size=(h, w), mode="bilinear", align_corners=True
            )
            for stage in self.stages
        ] + [feats]
        return self.bottleneck(torch.cat(priors, dim=1))


class _Edge_Module(nn.Module):
    def __init__(self, in_fea=(256, 512, 1024), mid_fea=256, out_fea=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, bias=False),
            InPlaceABNSync(mid_fea),
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()
        ef1 = self.conv1(x1)
        ef2 = self.conv2(x2)
        ef3 = self.conv3(x3)
        e1 = self.conv4(ef1)
        e2 = F.interpolate(
            self.conv4(ef2), size=(h, w), mode="bilinear", align_corners=True
        )
        e3 = F.interpolate(
            self.conv4(ef3), size=(h, w), mode="bilinear", align_corners=True
        )
        ef2 = F.interpolate(ef2, size=(h, w), mode="bilinear", align_corners=True)
        ef3 = F.interpolate(ef3, size=(h, w), mode="bilinear", align_corners=True)
        edge = self.conv5(torch.cat([e1, e2, e3], dim=1))
        edge_fea = torch.cat([ef1, ef2, ef3], dim=1)
        return edge, edge_fea


class _Decoder_Module(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            InPlaceABNSync(256),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            InPlaceABNSync(48),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, bias=False),
            InPlaceABNSync(256),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            InPlaceABNSync(256),
        )
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(
            self.conv1(xt), size=(h, w), mode="bilinear", align_corners=True
        )
        xl = self.conv2(xl)
        x = self.conv3(torch.cat([xt, xl], dim=1))
        return self.conv4(x), x


class _SCHPResNet(nn.Module):
    """SCHP ResNet-101 backbone + decoder (reproduced from AugmentCE2P.py)."""

    def __init__(self, num_classes: int):
        self.inplanes = 128
        super().__init__()
        # Three-layer stem
        self.conv1 = _conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = _conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = _conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ResNet stages
        self.layer1 = self._make_layer(_Bottleneck, 64, 3)
        self.layer2 = self._make_layer(_Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(_Bottleneck, 256, 23, stride=2)
        self.layer4 = self._make_layer(
            _Bottleneck, 512, 3, stride=1, dilation=2, multi_grid=(1, 1, 1)
        )
        # Head modules
        self.context_encoding = _PSPModule(2048, 512)
        self.edge = _Edge_Module()
        self.decoder = _Decoder_Module(num_classes)
        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True),
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(planes * block.expansion, affine=affine_par),
            )

        def _grid(i, g):
            return g[i % len(g)] if isinstance(g, tuple) else 1

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                dilation=dilation,
                downsample=downsample,
                multi_grid=_grid(0, multi_grid),
            )
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilation=dilation,
                    multi_grid=_grid(i, multi_grid),
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        context = self.context_encoding(x5)
        parsing_result, parsing_fea = self.decoder(context, x2)
        edge_result, edge_fea = self.edge(x2, x3, x4)
        fusion_result = self.fushion(torch.cat([parsing_fea, edge_fea], dim=1))
        # Return format mirrors the original: [[parsing, fusion], [edge]]
        return [[parsing_result, fusion_result], [edge_result]]


# ── Transformers output dataclass ────────────────────────────────────────────
@dataclass
class SCHPSemanticSegmenterOutput(ModelOutput):
    """
    Output type for :class:`SCHPForSemanticSegmentation`.

    Args:
        loss: Cross-entropy loss (only when ``labels`` is provided).
        logits: Final fusion logits, shape ``(batch, num_labels, H, W)``,
            upsampled to the input image resolution.
        parsing_logits: Decoder-branch logits before fusion,
            shape ``(batch, num_labels, H, W)``.
        edge_logits: Edge-branch logits, shape ``(batch, 2, H, W)``.
    """

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    parsing_logits: Optional[torch.Tensor] = None
    edge_logits: Optional[torch.Tensor] = None


# ── PreTrainedModel wrapper ───────────────────────────────────────────────────
class SCHPForSemanticSegmentation(PreTrainedModel):
    """
    SCHP ResNet-101 for human parsing / semantic segmentation.

    Usage — loading from an original SCHP ``.pth`` checkpoint::

        model = SCHPForSemanticSegmentation.from_schp_checkpoint(
            "checkpoints/schp/exp-schp-201908301523-atr.pth"
        )

    Usage — loading after :meth:`save_pretrained`::

        model = SCHPForSemanticSegmentation.from_pretrained(
            "./my-schp-model", trust_remote_code=True
        )
    """

    config_class = SCHPConfig
    # num_batches_tracked is not stored in the original SCHP checkpoints
    _keys_to_ignore_on_load_missing = [r"\.num_batches_tracked$"]

    def __init__(self, config: SCHPConfig):
        super().__init__(config)
        self.model = _SCHPResNet(num_classes=config.num_labels)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SCHPSemanticSegmenterOutput, Tuple]:
        """
        Args:
            pixel_values: ``(batch, 3, H, W)`` — normalised with SCHP BGR-indexed means.
            labels: ``(batch, H, W)`` integer class map for computing CE loss.
            return_dict: Override ``config.use_return_dict``.
        """
        return_dict = return_dict if return_dict is not None else True

        h, w = pixel_values.shape[-2:]
        raw = self.model(pixel_values)
        # raw = [[parsing_result, fusion_result], [edge_result]]

        logits = F.interpolate(
            raw[0][1], size=(h, w), mode="bilinear", align_corners=True
        )
        parsing_logits = F.interpolate(
            raw[0][0], size=(h, w), mode="bilinear", align_corners=True
        )
        edge_logits = F.interpolate(
            raw[1][0], size=(h, w), mode="bilinear", align_corners=True
        )

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return SCHPSemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            parsing_logits=parsing_logits,
            edge_logits=edge_logits,
        )

    @classmethod
    def from_schp_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[SCHPConfig] = None,
        map_location: str = "cpu",
    ) -> "SCHPForSemanticSegmentation":
        """
        Load from an original SCHP ``.pth`` checkpoint.

        Handles the ``module.`` prefix added by ``DataParallel`` training and
        remaps keys to the ``model.*`` namespace used by this wrapper.

        Args:
            checkpoint_path: Path to the ``.pth`` file.
            config: :class:`SCHPConfig` instance. Defaults to ATR-18 config.
            map_location: PyTorch device string (``"cpu"`` or ``"cuda"``).
        """
        if config is None:
            config = SCHPConfig()

        model = cls(config)

        raw = torch.load(checkpoint_path, map_location=map_location)
        state_dict = raw.get("state_dict", raw)

        # Strip DataParallel module. prefix if present
        if all(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

        # Remap to model.* namespace (self.model = _SCHPResNet)
        state_dict = {"model." + k: v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        real_missing = [k for k in missing if "num_batches_tracked" not in k]
        if real_missing:
            raise RuntimeError(
                f"Missing keys when loading SCHP checkpoint ({len(real_missing)} total): "
                f"{real_missing[:5]}"
            )
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys when loading SCHP checkpoint ({len(unexpected)} total): "
                f"{unexpected[:5]}"
            )

        return model
