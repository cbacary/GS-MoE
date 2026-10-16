def __getattr__(name):
    _map = {
        "CompressionConfig": ".compression_config",
        "MLPComponents": ".model_adapter",
        "MLPTensors": ".model_adapter",
        "AttentionComponents": ".model_adapter",
        "ModelAdapter": ".model_adapter",
        "LlamaAdapter": ".llama_adapter",
        "OptAdapter": ".opt_adapter",
        "MoEAdapter": ".moe_adapter",
        "MixtralAdapter": ".mixtral_adapter",
        "DeepseekAdapter": ".deepseek_adapter",
    }
    if name in _map:
        import importlib

        module = importlib.import_module(_map[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
