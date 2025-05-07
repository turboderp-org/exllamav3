from __future__ import annotations
from abc import ABC, abstractproperty, abstractmethod
import torch
import torch.nn.functional as F
from torch import nn
import os, json
from ..util.rope import RopeSettings, RopeStyle
from ..loader import SafetensorsCollection, VariantSafetensorsCollection
from ..util.file import read_dict, no_value, no_default
import uuid

class Config(ABC):
    arch_string = None
    load_isq: bool

    def __init__(
        self,
        directory: str,
        model_class,
        **kwargs,
    ):
        """
        Read a HF model config and prepare it for instantiation and loading

        :param directory:
            Directory containg the model config.json, weights, etc.

        :param expect_arch:
            Expected achitecture string
        """

        self.directory = directory
        self.model_class = model_class
        self.uuid = uuid.uuid4()

        # Verify architecture
        self.config_filename = os.path.join(directory, "config.json")
        with open(self.config_filename, encoding = "utf8") as f:
            self.config_dict = json.load(f)

        assert len(self.config_dict["architectures"]) == 1, \
            f"Multiple architectures defined in {self.config_filename}"

        arch = self.config_dict["architectures"][0]
        assert arch == self.arch_string, \
            f"Unexpected architecture {arch} in {self.config_filename}, should be {self.arch_string}."
        self.architecture = arch

        # Special mode to load tensors from across multiple variants of the same model
        if kwargs.get("st_variants"):
            self.stc = VariantSafetensorsCollection(kwargs.get("st_variants"))

        # Collect all .safetensors files in directory
        else:
            self.stc = SafetensorsCollection(directory, load_method = kwargs.get("load_method"))

        # Standard params, vocab
        self.bos_token_id = self.read_cfg(int, "bos_token_id", None)
        self.eos_token_id = self.read_cfg([int, list], "eos_token_id", None)
        self.pad_token_id = self.read_cfg(int, "pad_token_id", None)
        self.vocab_size = self.read_cfg(int, ["vocab_size", "text_config->vocab_size"], None)
        if isinstance(self.eos_token_id, list):
            self.eos_token_id_list = self.eos_token_id
            self.eos_token_id = self.eos_token_id[0]
        else:
            self.eos_token_id_list = [self.eos_token_id]

        # Standard params, unused
        self.initializer_range = self.read_cfg(float, "initializer_range", 0.02)

        # Universal params
        self.num_hidden_layers = -1
        self.head_dim = -1
        self.num_q_heads = -1
        self.num_kv_heads = -1
        self.pos_encoding_mode = "NONE"

        # Load parameters
        self.load_isq = False


    def read_cfg(self, *args):
        """
        Read from config.json, see read()
        """
        return read_dict(self.config_dict, *args)


    def assert_cfg(
        self,
        expected_type: type | list[type],
        keys: str | list[str],
        expected_value = no_value,
        optional = False
    ):
        """
        Read from config.json, see read(). Assert that config item either:
            - has expected value, or
            - has one of the expected values (if expected_value is list), or
            - is not present (if expected_value == no_value), or
        """

        value = self.read_cfg(expected_type, keys, no_value)
        if isinstance(expected_value, list):
            if value not in expected_value:
                raise ValueError(f"Key {keys} expected to be one of {expected_value} but was {value}")
        else:
            if value == no_value and not optional:
                raise ValueError(f"Key {keys} expected but not present.")
            if value != no_value and value != expected_value:
                raise ValueError(f"Key {keys} expected to have value {expected_value} but was {value}")


    @staticmethod
    def from_directory(directory: str, **kwargs) -> Config:
        """
        Create config from the specified directory if it contains a HF model of a supported architecture

        :param directory:
            Directory containing model files

        :param kwargs:
            load_method:
                See exllamav3.loader.safetensors.SafetensorsCollection

        :return:
            Architecture-specific config deriving from Exl2Config
        """

        from exllamav3.models.architectures import get_architectures
        architectures = get_architectures()

        config_filename = os.path.join(directory, "config.json")
        with open(config_filename, encoding = "utf8") as f:
            config_dict = json.load(f)

        assert "architectures" in config_dict, f"No architecture defined in {config_filename}"
        archs = config_dict["architectures"]
        assert len(archs) == 1, f"Multiple architectures defined in {config_filename}"
        arch = archs[0]
        assert arch in architectures, f"Unknown architecture {arch} in {config_filename}"

        arch_def = architectures[arch]
        config_class = arch_def["config_class"]
        config = config_class(directory, **kwargs)
        return config


    def read_rope_settings_default(self, rope_style: RopeStyle):
        return RopeSettings(
            head_dim = self.head_dim,
            rope_theta = self.read_cfg(float, "rope_theta", 10000.0),
            rope_scaling = self.read_cfg(dict, "rope_scaling", None),
            partial_rotary_factor = self.read_cfg(float, "partial_rotary_factor", 1.0),
            max_position_embeddings = self.read_cfg(int, "max_position_embeddings", None),
            original_max_position_embeddings = self.read_cfg(int, "original_max_position_embeddings", None),
            rope_style = rope_style,
        )


    def override_dynamic_seq_len(self, new_max_position_embeddings: int):
        """
        Override max_position_embeddings from the config. Necessary for some models (like Phi) that have two
        sets of RoPE factors, so the correct set can be loaded as the model is initialized. Changing this after
        the model is created has no effect.
        """
        pass