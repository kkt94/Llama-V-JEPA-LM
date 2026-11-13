"""Configuration of V-JEPA-LM (JEPA-LLM): frozen V-JEPA 2.1 video encoder -> spectral bridge -> LLM (Qwen3 or Llama 3)."""
from transformers import AutoConfig, PretrainedConfig


class JepaLlmConfig(PretrainedConfig):
    model_type = "jepa_llm"
    sub_configs = {"text_config": AutoConfig}

    def __init__(self, text_config=None, vision_config=None, num_frames: int = 32, image_size: int = 384,
                 evidence_cells: int = 8, vision_levels: int = 4, vision_level_dim: int = 1408, spectral_k: int = 2048,
                 bridge_hidden: int = 2048, max_tubelets: int = 64, time_dim: int = 64, rope_inv_freq_bf16: bool = True,
                 vision_variant: str = "vit_giant_384", chat_template_name: str = "qwen3_instruct", **kwargs):
        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config.setdefault("model_type", "qwen3")
            from transformers import CONFIG_MAPPING
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            from transformers import CONFIG_MAPPING
            text_config = CONFIG_MAPPING["qwen3"]()
        self.text_config = text_config
        # V-JEPA 2.1 encoder: "vit_giant_384" (ViT-g/16, vjepa2_1_encoder.vjepa2_1_vit_giant_384, 4 output levels) or
        # "vit_large_384" (ViT-L/16 distilled from ViT-G, vjepa2_1_encoder.vjepa2_1_vit_large_384, 1 output level);
        # vision_config keys override the constructor's defaults
        self.vision_variant = vision_variant
        self.vision_config = dict(vision_config or {})
        self.num_frames = num_frames                # frames per clip (16 tubelets of 2 frames)
        self.image_size = image_size                # square crop side, pixels
        self.evidence_cells = evidence_cells        # the 24 x 24 patch grid pooled to cells x cells per tubelet
        self.vision_levels = vision_levels          # encoder output levels used (ViT-g: blocks 9 / 19 / 29 / 39; ViT-L: last block)
        self.vision_level_dim = vision_level_dim
        self.spectral_k = spectral_k                # principal directions kept by the equalisation
        self.bridge_hidden = bridge_hidden
        self.max_tubelets = max_tubelets
        self.time_dim = time_dim
        # chat template the model was trained with (processing_jepa_llm.CHAT_TEMPLATES): "qwen3_instruct" (plain turns),
        # "qwen3_nothink" (every assistant turn starts with an empty <think></think> block, Qwen3's non-thinking format),
        # "llama32_instruct" / "llama31_instruct" (the Llama 3.2 / 3.1 Instruct templates with a fixed date line)
        self.chat_template_name = chat_template_name
        # True: the LLM's rotary frequencies are rounded to bf16 (what casting the whole model to bf16 does), as in the
        # evaluation setup this export reproduces. False: fp32 rotary frequencies (the transformers default for a bf16 load).
        self.rope_inv_freq_bf16 = bool(rope_inv_freq_bf16)
        kwargs.setdefault("tie_word_embeddings", getattr(text_config, "tie_word_embeddings", True))
        super().__init__(**kwargs)
