# llama.cpp

The whole idea of `llama.cpp` is to have purely C/C++ implementation for LLM inference.
Originally, Georgi Gerganov, the author of `llama.cpp`, focused on CPU inference resulting in quite a good list of CPU architectures to be supported (x86\_64, ARM, Apple Silicon).
While GPU was shipped to `llama.cpp` at some point, it is still in infancy compared to the CPU support.

The main "engine" of `llama.cpp` is `ggml`, a tensor library similar to PyTorch and TensorFlow but fully written in C/C++.
You can have a look at [this article](https://huggingface.co/blog/introduction-to-ggml) to see how it can be used for arbitrary computations.

Here is the list of features you can count on in `llama.cpp`:
* quantization including KV cache quantization (a significant effort has been undertaken by Georgi Gerganov to support various quantization types)
* CPU AVX-like extensions for optimization
* speculative decoding (check it out in the [llama-server docs](https://github.com/ggml-org/llama.cpp/tree/master/tools/server))
* pipeline parallelism and tensor parallelism when using multiple GPUs (check out `--split-mode` in the [llama-server docs](https://github.com/ggml-org/llama.cpp/tree/master/tools/server))
* LoRA adapters (check out `--lora` [here](https://github.com/ggml-org/llama.cpp/tree/master/tools/main))
* continuous batching (check out `-cb` in the [llama-server docs](https://github.com/ggml-org/llama.cpp/tree/master/tools/server))
* function calling (see [this](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md))

Interestingly, `llama.cpp` had a fine-tune support some time ago but [abandoned it in the mid-2024](https://github.com/ggml-org/llama.cpp/pull/8669).
Nonetheless, as of July 2025, they are working on LLM training support including fine-tuning (see [PR 10544](https://github.com/ggml-org/llama.cpp/pull/10544) and [PR 13873](https://github.com/ggml-org/llama.cpp/pull/13873)).

## Installation

Via `brew` on mac and linux:
```bash
brew install llama.cpp
```

Via docker:
```bash
docker image pull <image>
```
See the list of containers here: https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md 

For CUDA, you want to use `ghcr.io/ggml-org/llama.cpp:full-cuda`

## Running the docker container

Assuming that we have the full version of the docker image, here is the standard template for running the container:
```bash
docker run -v /path/to/models:/models <image> <command> <command_options>
```

Here are the available commands as of now:
* `--convert` (conversion to gguf)
* `--quantize` (quantization)
* `--finetune` (LoRA fine-tuning)
* `--run` (model inference with a prompt)
* `--server` (same as llama-server)

### Conversion to gguf

`llama.cpp` runs models converted to the gguf format so it is necessary to convert a model into this format.
This process is extremely efficient.
You can convert 32B model on one CPU core within 5 minutes.

Here is an example of the command:
```bash
docker run -v /path/to/models:/models ghcr.io/ggml-org/llama.cpp:full-cuda --convert --outtype f16 "/models/Qwen2.5-0.5B"
```
The `models` dir you mount here should contain separate model dirs in the Hugging Face format (i.e., you either download them or use a snapshot from the cache dir).
Some other formats are also supported but I have not tested them.
Note that the output `.gguf` file will be saved in the same directory.

### Model inference

This command will directly run the model inference without deploying a server.
It is convenient for testing the model setup.

Here is an example of the command:
```bash
docker run --gpus '"device=1,2,3,4,5,6,7,8"' -v /path/to/models:/models ghcr.io/ggml-org/llama.cpp:full-cuda --run -m "/models/Qwen-2.5-0.5B-F16.gguf"  -p "Building a website can be done in 10 simple steps:" -n 512 -ngl 35
```
The following options have been used:
* `--gpus '"device=<comma_separated_list_of_device_ids>"'` (GPU IDs to be mounted to the contained)
* `-m </path/to/gguf_model/in/container>` (the gguf model file path) 
* `-p "<prompt>"` (prompt to be processed)
* `-n <number_of_completion_tokens>` (maximum number of completion tokens)
* `-ngl <number_of_layers>` (number of layers which will be uploaded to GPUs; other layers will be uploaded on CPU)

`llama.cpp` uses a bit weird terminology.
They say that they "offload" layers on GPUs so do not be confused with this language.
During the model loading, there will be a message printed to stdout saying how many layers are "offloaded" to GPUs and CPU.
Check this message carefully to be sure that the layers have been distributed as planned.

`llama.cpp` employs pipeline parallelism which means that the each GPU hosts its own layers and GPUs are run sequentially.
Not every efficient, that's why vLLM with its tensor parallelism may be preferred.

Once the prompt has been processed, you should observe the following performance metrics (maybe even more):
```
prompt eval time =    2286.99 ms /    31 tokens (   73.77 ms per token,    13.55 tokens per second)
       eval time =  286282.06 ms /  2441 tokens (  117.28 ms per token,     8.53 tokens per second)
      total time =  288569.05 ms /  2472 tokens
```

### Server

This commands deploy the model on a specified port.

Here is an example of the command:
```bash
docker run --gpus '"device=1,2,3,4,5,6,7,8"' -v /path/to/models:/models ghcr.io/ggml-org/llama.cpp:full-cuda --server -m "/models/Qwen-2.5-0.5B-F16.gguf" -c 4096 -ngl 35 --port 8123 --host 0.0.0.0
```

The following options have been used:
* `-c <context_length>` (context length)
* `--port <port>` (post which the server will listen to)
* `--host 0.0.0.0` (host requests from all the incoming ips)

Without using `--host 0.0.0.0`, the server will be available for local requests only which means that you will not be able to access the server outside the container.

After running the container, the following line will be printed to stdout:
```
main: server is listening on http://0.0.0.0:8123 - starting the main loop
```
