from transformers import PretrainedConfig

_LIP_LABELS = [
    "Background",
    "Hat",
    "Hair",
    "Glove",
    "Sunglasses",
    "Upper-clothes",
    "Dress",
    "Coat",
    "Socks",
    "Pants",
    "Jumpsuits",
    "Scarf",
    "Skirt",
    "Face",
    "Left-arm",
    "Right-arm",
    "Left-leg",
    "Right-leg",
    "Left-shoe",
    "Right-shoe",
]


class SCHPConfig(PretrainedConfig):
    r"""
    Configuration for **Self-Correction-Human-Parsing (SCHP)**.

    Args:
        num_labels (`int`, *optional*, defaults to 20):
            Number of segmentation classes (20 for LIP dataset).
        input_size (`int`, *optional*, defaults to 473):
            Spatial resolution the model expects (height = width).
        backbone (`str`, *optional*, defaults to `"resnet101"`):
            Backbone architecture name. Only `"resnet101"` is supported.
    """

    model_type = "schp"

    def __init__(
        self,
        num_labels: int = 20,
        input_size: int = 473,
        backbone: str = "resnet101",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.input_size = input_size
        self.backbone = backbone

        if "id2label" not in kwargs:
            self.id2label = {
                str(i): lbl for i, lbl in enumerate(_LIP_LABELS[:num_labels])
            }
        if "label2id" not in kwargs:
            self.label2id = {
                lbl: str(i) for i, lbl in enumerate(_LIP_LABELS[:num_labels])
            }
