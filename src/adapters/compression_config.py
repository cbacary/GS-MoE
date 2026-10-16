from dataclasses import dataclass, fields, field, MISSING
from typing import Optional, get_origin, get_args
import argparse


@dataclass
class CompressionConfig:

    # The actual model to compress
    model: str = "mistralai/Mixtral-8x7B-v0.1"

    # The architecture to pull tokenizer from if compressing already compressed model
    tokenizer_src: str = "mistralai/Mixtral-8x7B-v0.1"

    seed: int = 1234

    layers_per_compression_step: int = 4
    compression_ratio: float = 0.5
    calib_size: int = 512
    output_dir: str = "compressed_output"
    device: int = 0
    calibs_batch_size: int = 16

    order: Optional[str] = "gs-moe"

    compressed_acts_save_path: str = ""  # empty means disabled, takes path

    n_shared_projs: int = 1

    save: bool = False  # saves the model to output_dir/model
    uniformity: str = "leverage"  # taylor | leverage
    dynamic_fact_nys_ratio: bool = False

    grouped_iters: int = 25
    grouped_restarts: int = 8

    flipped_down: bool = True  # don't change. deprecated i guess. must be true.

    nys_to_fact_ratio: float = 0.6

    disable_up: bool = False
    disable_down: bool = False

    dataset: str = "alpaca"

    archive: str = ""  # path to save the final model to. copies contents of output_dir to `archive`
    benchmark: bool = False
    note: str = "NA"

    nystrom_layer_smoothing: float = 1.0
    factorization_layer_smoothing: float = 1.0

    nystrom_ridge: float = 1e-2
    Ex_ridge: float = 0.0
    optuna: bool = False

    # Number of parallel GPUs to use. You should do one per gpu if you can fit entire model on each gpu
    shards: int = 1

    # Set by the parent on the workers it spawns. -1 == not a shard worker.
    shard_rank: int = -1
    shard_start_layer: int = 0
    shard_end_layer: int = -1
    shard_config: str = ""

    _parser_extras: dict = field(default=None, init=False, repr=False, compare=False)

    _FIELD_HELP = {
        "order": "nystrom:0.3:0.5,factorization:0.25:4 -- <method>:<compression_ratio>:<sparsity>,<method>...",
    }

    @classmethod
    def _resolve_type(cls, tp):
        """Unwrap Optional[X] → X, leave primitives unchanged."""
        origin = get_origin(tp)
        if origin is type(None):
            return None
        if origin is not None:  # e.g. Optional = Union[X, None]
            inner = [a for a in get_args(tp) if a is not type(None)]
            return inner[0] if inner else str
        return tp

    @classmethod
    def make_parser(cls, parser=None):
        parser = parser or argparse.ArgumentParser()
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            flag = f"--{f.name}"
            resolved = cls._resolve_type(f.type)
            if resolved is bool:
                parser.add_argument(flag, action="store_true", default=f.default)
            else:
                kwargs = {"type": resolved}
                if f.default is not MISSING:
                    kwargs["default"] = f.default
                else:
                    kwargs["required"] = True
                if f.name in cls._FIELD_HELP:
                    kwargs["help"] = cls._FIELD_HELP[f.name]
                parser.add_argument(flag, **kwargs)
        return parser

    @classmethod
    def from_args(cls, args=None):
        parser = cls.make_parser()
        parsed = parser.parse_args(args)
        init_fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in vars(parsed).items() if k in init_fields})

    def get(self, key: str, default=None):
        """Allow config.get('ridge_Cx', 1e-4) like a dict."""
        val = getattr(self, key, default)
        return val if val is not None else default

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __contains__(self, key: str):
        return hasattr(self, key)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (useful for metrics)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
