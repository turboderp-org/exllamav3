from __future__ import annotations
from abc import ABC
import os, json
from dataclasses import dataclass
from ..util.rope import RopeSettings, RopeStyle
from ..loader import SafetensorsCollection
from ..util.file import read_dict, no_value, no_default
import uuid

@dataclass
class InferParams:
    """
    Runtime inference parameters. Configure before loading model/modules
    """

    # Avoid reconstruct path during GEMM. Forces use of low-bsz GEMM/GEMV kernels. Also disables MGEMM path
    no_reconstruct: bool = False

    # Bitrate threshold for enabling MGEMM
    mgemm_K_threshold: int = 0

    # Width threshold for enabling MGEMM regardless of bitrate
    mgemm_n_threshold: int = 0

    def __init__(self):
        # With the int8 GEMV mode (on by default), separate int8 GEMV calls beat the fused MGEMM
        # only when a single matrix is wide enough to fill the GPU on its own (and K is within the
        # int8 gate); narrow same-input pairs stay fused, where batching is what restores
        # utilization. Only pairs the int8 path can take (mul1 codebook) are ever unfused
        if int(os.environ.get("EXL3_INT8_GEMV", 2)) > 0:
            self.mgemm_K_threshold = int(os.environ.get("EXL3_MGEMM_K_THRESHOLD", 6))
            self.mgemm_n_threshold = int(os.environ.get("EXL3_MGEMM_N_THRESHOLD", 8192))

    def use_mgemm(self, K: int, out_features: int, mul1: bool = False) -> bool:
        # Unfusing only pays when the separate GEMV calls can actually take the int8 path, which
        # requires the mul1 codebook; other tensors always keep the fused MGEMM
        if not mul1:
            return True
        # Fuse when K is at/above the bitrate threshold (int8 GEMV can't take those anyway) or the
        # matrices are too narrow for separate GEMV calls to fill the GPU
        return K >= self.mgemm_K_threshold or (self.mgemm_n_threshold > 0 and out_features < self.mgemm_n_threshold)


class NullConfig:
    def __init__(self):
        self.infer_params = InferParams()


class Config(ABC):
    arch_string = None
    load_isq: bool

    def __init__(
        self,
        directory: str,
        model_classes: dict,
        layer_map: list[int] | str | None = None,
        **kwargs,
    ):
        """
        Read HF model config and prepare it for instantiation and loading

        :param directory:
            Directory containg the model config.json, weights, etc.

        :param layer_map:
            List of layer indices for RYS relayering. Forward passes will traverse the model in this order.
            If layers repeat, attached cache will allocate individual key/value tensors for each instance of
            each layer. Model weights are still loaded in the original layer order.

            Alternatively, can be a string spec, e.g.:
                "0,1,2,3,4,5,6,4,5,6"   list of ints to parse
                "0..6,4..6"             inclusive intervals
                "0..6,4,5,6"            mixed ints and intervals
                "..,4.."                open-ended intervals (limited by model)
        """

        self.directory = directory
        self.model_classes = model_classes
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

        # Collect all .safetensors files in directory
        self.stc = SafetensorsCollection(directory, kwargs.get("load_method"), self.get_tensor_name_fixes())

        # Standard params, vocab
        self.bos_token_id = self.read_cfg(int, ["bos_token_id", "text_config->bos_token_id"], None)
        self.eos_token_id = self.read_cfg([int, list], ["eos_token_id", "text_config->eos_token_id"], None)
        self.pad_token_id = self.read_cfg(int, ["pad_token_id", "text_config->pad_token_id"], None)
        self.vocab_size = self.read_cfg(int, ["vocab_size", "text_config->vocab_size"], None)
        if isinstance(self.eos_token_id, list):
            self.eos_token_id_list = self.eos_token_id
            self.eos_token_id = self.eos_token_id[0]
        else:
            self.eos_token_id_list = [self.eos_token_id]

        # Make sure no None entries in list
        self.eos_token_id_list = [e for e in self.eos_token_id_list if e is not None]

        # Standard params, unused
        self.initializer_range = self.read_cfg(float, "initializer_range", 0.02)

        # Universal params
        self.num_hidden_layers = -1
        self.head_dim = -1
        self.num_q_heads = -1
        self.num_kv_heads = -1
        self.pos_encoding_mode = "NONE"
        self.max_position_embeddings = self.read_cfg(int, "max_position_embeddings", self.default_max_position_embeddings())

        # Main RoPE module (for MRoPE, individual attn layers have their own modules)
        self.g_rope = None

        # Load parameters
        self.load_isq = False

        # Layer map
        if layer_map is None:
            self.layer_map = None
            self.layer_map_str = None
        elif isinstance(layer_map, str):
            self.layer_map = None
            self.layer_map_str = layer_map
        else:
            assert isinstance(layer_map, list), "layer_map must be string or list of ints"
            self.layer_map = layer_map
            self.layer_map_str = None

        # Inference parameters
        self.infer_params = InferParams()


    def get_tensor_name_fixes(self):
        return {}


    def default_max_position_embeddings(self):
        return 8192


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
    # def from_directory(directory: str, arch_override: str | None = None, **kwargs) -> Config:
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

        from exllamav3.architecture.architectures import get_architectures
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
        # config = config_class(directory, arch_override = arch_override, **kwargs)
        config = config_class(directory, **kwargs)
        return config


    def read_rope_settings_default(
        self,
        rope_style: RopeStyle,
        default_rope_theta: float = 10000.0,
        default_partial_rotary_factor: float = 1.0,
        config_dict: dict | None = None,
        theta_key: str | list = None,
        override_type: str = None,
        override_head_dim: int | None = None
    ):
        if config_dict is None:
            config_dict = self.config_dict

        if theta_key is None:
            theta_key = ["rope_theta", "rope_parameters->rope_theta"]

        return RopeSettings(
            head_dim = override_head_dim or self.head_dim,
            rope_theta = read_dict(
                config_dict,
                float,
                theta_key,
                default_rope_theta,
                wrong_type_as_missing = True
            ),
            rope_scaling = read_dict(config_dict, dict, ["rope_scaling", "rope_parameters"], None),
            rotary_dim = read_dict(config_dict, int, "rotary_dim", None),
            partial_rotary_factor = read_dict(
                config_dict,
                float,
                ["partial_rotary_factor", "rope_parameters->partial_rotary_factor"],
                default_partial_rotary_factor,
                wrong_type_as_missing = True
            ),
            max_position_embeddings = read_dict(config_dict, int, "max_position_embeddings", None),
            original_max_position_embeddings = read_dict(config_dict, int, "original_max_position_embeddings", None),
            rope_style = rope_style,
            override_type = override_type
        )


    def override_dynamic_seq_len(self, new_max_position_embeddings: int):
        """
        Override max_position_embeddings from the config. Necessary for some models (like Phi) that have two
        sets of RoPE factors, so the correct set can be loaded as the model is initialized. Changing this after
        the model is created has no effect.
        """
        pass