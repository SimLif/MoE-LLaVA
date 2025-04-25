from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # ===================================================================
    image_tower: Optional[str] = field(default=None)
    video_tower: Optional[str] = field(default=None)
    image_projector_type: Optional[str] = field(default='linear')
    video_projector_type: Optional[str] = field(default='linear')
    video_global_proj: bool = field(default=False)
    video_temproal_proj: bool = field(default=False)
    video_spatial_proj: bool = field(default=False)
    # ===================================================================

    # =============================================================
    only_lora_ffn: bool = True
    moe_enable: bool = False
    train_modules: Optional[List[str]] = field(default=None, metadata={"help": ""})
    moe_mode: str = field(
        default="second_half",
        metadata={
            "help": "The backend to be used for half precision.",
            "choices": ["first_half", "second_half", "sparse", "dense"],
        },
    )
    moe_layers_idx: Optional[List[int]] = field(default=None, metadata={"help": "where to place moe layers."})
    ep_size: int = 1
    num_experts: Optional[List[int]] = field(default=4, metadata={"help": "number of experts for each moe layer."})
    top_k_experts: int = 2
    capacity_factor: float = 1.
    eval_capacity_factor: float = 2.
    min_capacity: int = 0
    use_residual: bool = False
    router_aux_loss_coef: float = 0.01
    # =============================================================

    skip_moe_init: bool = False
    ffn_only: bool = False
    load_k_experts: bool = False
    k_experts_path: Optional[str] = None
    use_shared_experts: bool = False
    mone_enable: bool = False
    mone_r: int = 128
    mone_dropout: float = 0.00
    mone_expert_type: str = "small_expert"
    mone_gate_type: str = "token_gating"
    mone_num_heads: int = 8
    mone_use_query_bn: bool = True
    mone_act_fn: str = "silu"
    mone_use_expert_gate: bool = False
    mone_load_original: bool = False
    mone_forward_mode: str = "batched"
    unfreeze_shared_epoch: int = 1
    use_combined_gate: bool = False
    combined_gate_type: str = 'ds'
    combined_gate_drop: float = 0.002
    from_pretrained: bool = False
    from_pretrained_path: str = ""
    warm_up_experts: bool = False
    kd_align: bool = False

@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    # ===================================================================
    data_path: Optional[List[str]] = field(default=None, metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    num_frames: int = 8
    # ===================================================================
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    fps: float = 1.0

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    freeze_shared: bool = False
    shared_lr: Optional[float] = None