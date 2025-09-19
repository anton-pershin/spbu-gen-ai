# vLLM

## Installation

The official doc recommends to use `uv`.
I still prefer to use conda to handle my environments so we go like this:
```bash
conda create -n myenv python=3.12
conda activate myenv
pip install --upgrade uv
uv pip install vllm --torch-backend=auto
```

This combination of `uv` and `--torch-backend=auto` is nice because vllm will automatically select the torch version suitable for the available CUDA driver.
You can also do it manually.

Also ensure that `build-essential` is installed because vllm needs a C compiler for triton JIT:
```bash
apt-get install build-essential
```

## Offline batch inference (without serving)

In general, vLLM uses HuggingFace models in the same way as `transformers` uses them.
It means that it will try to download the required model if it is not in the cache and will use the configs provided with the model.

See [this script](vllm_offline_batch_inference.py) for an example. 

## Online batch inference (serving an OpenAI-compatible API)

Very straightforward:
```bash
vllm serve <model_name> --host <host> --port <port>
```
Skip `--host` if you do it locally.
