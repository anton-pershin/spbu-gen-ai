# llama.cpp

The whole idea of `llama.cpp` is to have purely C/C++ implementation for LLM inference.
Originally, Georgi Gerganov, the author of `llama.cpp`, focused on CPU inference resulting in quite a good list of CPU architectures to be supported (x86\_64, ARM, Apple Silicon).
While GPU was shipped to `llama.cpp` at some point, it is still in infancy compared to the CPU support.

The main "engine" of `llama.cpp` is `ggml`, a tensor library similar to PyTorch and TensorFlow but fully written in C/C++.
You can have a look at [this article](https://huggingface.co/blog/introduction-to-ggml) to see how it can be used for arbitrary computations.

To run a model, it must be converted to the [GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
It consists of two parts: tensors and metadata.
Metadata can be treated as `config.json`, `generation_config.json` etc. (from the HF format) combined.
For example, you can find a Jinja template in metadata in `tokenizer.chat_template`.

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

Note that the server caches prompts by default. Take this into account while measuring performance.

## Inference peculiarities

### Input layer (token embedding) on CPU

No matter whether you offload all the layers on GPU or not, the input layer is [hardcoded to run on CPU](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp):
```cpp
// assign the input layer
// there is very little benefit to offloading the input layer, so always keep it on the CPU
pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };
```

This makes sense.
The embedding matrix is huge while there are no compute-bound operations related to it - we just take a vector from it by index (i.e., token).
Let's take Qwen 2.5 7B in BF16 as an example.
Its embedding matrix has shape 152064x3584, i.e. its would consume about 1 GB of VRAM without any compute being involved.
One could say that CPU -> GPU transfer would be time-consuming but the amount of data transferred is actually small.
In the prefill stage, it is just 28 MB for 4k context.
Given that the standard duration of the prefill stage is about 2 s, this overhead can be ignored.
In the decode stage, it is just 7 KB per one forward pass which typically takes 50 ms.

### Parallel decoding

You can enable batched inference (i.e., processing several requests in parallel) by `-np <N>` flag.
By default, continuous batching is supported benefiting different-length-of-prompts-or-completions scenarios.
If you want to use static batching benefiting same-length scenarios, you need to add `-nocb` flag.

There is an interesting problem related to parallel decoding.
Despite the name, flag `-c <context_length>` does not precisely correspond to the context size.
In fact, this is the number of tokens in a KV cache in a single layer.
Given that the KV cache is unified in `llama.cpp`, it means that the actual eligible context length is `context_length / N`.

### Distributed inference

`llama.cpp` supports two types of distributed inference: tensor parallelism and pipeline parallelism.
They terminology is a bit different though.
They call it the "split mode" where tensor parallelism is "splitting by rows" and pipeline parallelism is "splitting by layers":
* `-sm <mode>` (`layer` or `row`)

Splitting by layers is enabled by default.
It will just distribute layers to the available GPUs evenly.
In a single-request mode, this will lead to GPU underutilization because when the i-th GPU is working others are idle.
Theoretically, this should not be a problem in batched inference but I am not sure `llama.cpp` supports "parallel" mode in pipeline parallelism.

Splitting by rows implies splitting all the matrix rows beforehand and sending them to dedicated GPUs.
It looks like tensor parallelism but [it seems](https://github.com/ggml-org/llama.cpp/issues/9086) that they do `all-reduce` after each operation.
E.g., they do not have simple optimizations for subsequent matmuls (split by columns then split by rows).
Perhaps, this overhead is negligible if we have an NVLink bridge but it will only slow down inference if our GPUs are connected via PCIExpress only (pipeline parallelism should be used in this case).
The conclusion is that this mode implementation is highly suboptimal.
Here is a [post](https://www.ahmadosman.com/blog/do-not-use-llama-cpp-or-ollama-on-multi-gpus-setups-use-vllm-or-exllamav2) on this topic.

### FlashAttention

`llama.cpp` supports FlashAttention with the `-fa` flag which is disabled by default.
It significanty reduces the prefill time for long prompts if the model is fully offloaded to GPU (CPU and Vulkan support is either absent or limited). 

### KV cache quantization

You can control K and V cache quantization separately:
* `-ctk <type>` (K values)
* `-ctv <type>` (V values)

By default, F16 precision is used.
The safe option is `q8_0` for both K and V.
The practical benefit comes from reduced memory consumption.
For multi-head attention, KV cache memory consumption in bytes is $N_{layers} \times N_{batch} \times N_{ctx} \times d \times 2 \times \text{sizeof(dtype)}$, where $d$ is the embedding dimension.
E.g., for 4k context and batch size 16, it is around 27 GB for a BF16 KV cache.
Using 8-bit quantization, you cut a half of this consumption (equivalently, you double the maximum context).

[Here](https://github.com/ggml-org/llama.cpp/pull/7412) you can find an extensive study on different combinations of weight, K and V quantization modes in terms of their effect on accuracy (ppl in their case).
The practical conclusion is that 8-bit quantization of KV values lead to virtually no accuracy drop ($10^{-3}$ in ppl, somewhat comparable to the sampling uncertainty of the BF16 model).
Interestingly, it is K values that are the most sensitive to quantization rather than V values (and weights are more sensitive than both K and V). 
