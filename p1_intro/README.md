# Hugging Face models

## Model structure

Originally, the models are stored in a repo which can be downloaded via git lfs.
Here are the most important files describing a model:
```
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
vocal.json
```

File `config.json` contains general information about the model.
File `generation_config.json` contains general information about the model generation process.
File `tokenizer.json` contains all the information necessary for the tokenizer (incl. what is contained in `tokenizer_config.json` and `vocal.json`)
File `tokenizer_config.json` contains general information about tokenization and also the chat template, added tokens and special tokens (those which are not in the vocabulary)
File `vocal.json` is simply a "token as a substring" -> "token as an integer" mapping
File `model.safetensors` contains the model weights. It is a replacement for pickle. It is a dict-like structure with tensors stored as values. If the tensor format is `pt` (that's a typical case), these tensores are stored as `torch.Tensor`. On the hugging face web-site, you can easily browse the structure of any safetensor file. That's quite a handy way to learn the structure of any model. E.g., you may learn that the token embeddings at the first layer occupy more than 25% model parameters in small models (e.g., 0.5B) 

## Sampling parameters

Most LLMs are featured with three sampling parameters: temperature, top_k and top_p. They are applied sequentially: the logits are first scaled by temperature, then top_k largest logits are selected, then they are softmaxed to get probabilities (but they will not sum to one because we dropped some tokens), and finally we pass only those the most probable tokens for top_k selected ones whose cumulative probability (their sum) is less then top_p. Finally, the remaining probabilities are re-normalized to sum to one and we start sampling from this discrete distribution.

Temperature scales the logits: $x \leftarrow x/T$
* `temperature` = 0 implies greedy sampling
* `temperature` $\in (0; 1)$ makes likely tokens even more likely
* `temperature` > 1 makes the token distribution more uniform

# Base and instruct models

## Instruct models

Let's have a look at the chat template used by Qwen 2.5:
```python
{%- if tools %}\n    
    {{- '<|im_start|>system\\n' }}\n

    {%- if messages[0]['role'] == 'system' %}\n
        {{- messages[0]['content'] }}\n
    {%- else %}\n
        {{- 'You are a helpful assistant.' }}\n
    {%- endif %}\n

    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n

    {%- for tool in tools %}\n
        {{- \"\\n\" }}\n
        {{- tool | tojson }}\n
    {%- endfor %}\n    

    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n

{%- else %}\n
    {%- if messages[0]['role'] == 'system' %}\n
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    
    {%- else %}\n
        {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}\n
    {%- endif %}\n
{%- endif %}\n

{%- for message in messages %}\n
    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n
    {%- elif message.role == \"assistant\" %}\n
        {{- '<|im_start|>' + message.role }}\n
        {%- if message.content %}\n
            {{- '\\n' + message.content }}\n
        {%- endif %}\n

        {%- for tool_call in message.tool_calls %}\n
            {%- if tool_call.function is defined %}\n
                {%- set tool_call = tool_call.function %}\n
            {%- endif %}\n

            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n
            {{- tool_call.name }}\n
            {{- '\", \"arguments\": ' }}\n
            {{- tool_call.arguments | tojson }}\n
            {{- '}\\n</tool_call>' }}\n
        {%- endfor %}\n

        {{- '<|im_end|>\\n' }}\n
    {%- elif message.role == \"tool\" %}\n
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n
            {{- '<|im_start|>user' }}\n
        {%- endif %}\n

        {{- '\\n<tool_response>\\n' }}\n
        {{- message.content }}\n
        {{- '\\n</tool_response>' }}\n
        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n
            {{- '<|im_end|>\\n' }}\n
        {%- endif %}\n
    {%- endif %}\n
{%- endfor %}\n

{%- if add_generation_prompt %}\n
    {{- '<|im_start|>assistant\\n' }}\n
{%- endif %}\n",
```

# OpenAI REST API

API specification: https://platform.openai.com/docs/api-reference/introduction

TODO
