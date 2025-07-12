from transformers import AutoModelForCausalLM, AutoTokenizer


def print_llm_response(tokenizer, generated_ids) -> None:
    print()
    print(tokenizer.batch_decode(generated_ids)[0])
    print()


def run_base_model():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Bad example #1: base model will repeat stuff
    model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_length=64, max_new_tokens=64)
    print_llm_response(tokenizer, generated_ids)

    # Bad example #2: base model will not follow the instruction
    model_inputs = tokenizer(["Write the Fibonacci sequence generator. "], return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_length=64, max_new_tokens=64)
    print_llm_response(tokenizer, generated_ids)

    # Good example #3: base model will follow the instruction
    model_inputs = tokenizer(["Exercise #1: solve $x^2 - 4x + 1 = 0$ for x"], return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_length=64, max_new_tokens=64)
    print_llm_response(tokenizer, generated_ids)

    # Bad example #4: just one "period" added => base model does not follow the instruction
    model_inputs = tokenizer(["Exercise #1: solve $x^2 - 4x + 1 = 0$ for x."], return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_length=64, max_new_tokens=64)
    print_llm_response(tokenizer, generated_ids)

    # Bad example #5: add more letters => base model diverges. Base models are just not robust
    model_inputs = tokenizer(["Exercise #1: solve $x^2 - 4x + 1 = 0$ for x. S"], return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_length=64, max_new_tokens=64)
    print_llm_response(tokenizer, generated_ids)

    # Good example #6: base model saw some examples of intrustion-tuning corpora
    # so it can follow the instruct format (see the template in README)
    lines = [
        "<|im_start|>system",
        "You are a helpful assistant",
        "<|im_end|>",
        "<|im_start|>user",
        "Write the Fibonacci sequence generator.",
        "<|im_end|>",
        "<|im_start|>assistant\n",
    ]
    model_inputs = tokenizer(["\n".join(lines)], return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_length=64, max_new_tokens=64)
    print_llm_response(tokenizer, generated_ids)


def run_instruct_model():
    # If not downloaded, HG will download it to
    # ~/.cache/huggingface/hub
    # Note that the cache format is not the same as the repo format
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Neutral example #1: instruct model is better than base one but may still be a bit unguided
    model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_length=64, max_new_tokens=64)
    print_llm_response(tokenizer, generated_ids)

    # Neutral example #2: repeat the request to show the effect non-zero temperature
    generated_ids = model.generate(**model_inputs, max_length=64, max_new_tokens=64)
    print_llm_response(tokenizer, generated_ids)

    # Neutral example #3: let's load the instruct model with zero temperature and 
    # repeat the experiment again
    # HuggingFace does not allow one to set `temperature=0` with `do_sample=True`
    # (the latter refers to "stochastic sampling") so we effectively set the zero 
    # temperature by setting `do_sample=False` and unsetting `temperature`, `top_p` 
    # and `top_k` which have been brought from `generation_config.json`
    generated_ids = model.generate(
        **model_inputs,
        max_length=64,
        max_new_tokens=64,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    print_llm_response(tokenizer, generated_ids)

    generated_ids = model.generate(
        **model_inputs,
        max_length=64,
        max_new_tokens=64,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    print_llm_response(tokenizer, generated_ids)

    # Good example #4: the template makes the  use improves when we follow the instruct template
    lines = [
        "<|im_start|>system",
        "You are a helpful assistant",
        "<|im_end|>",
        "<|im_start|>user",
        "The secret to baking a good cake is",
        "<|im_end|>",
        "<|im_start|>assistant\n",
    ]
    model_inputs = tokenizer(["\n".join(lines)], return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_length=64, max_new_tokens=64)
    print_llm_response(tokenizer, generated_ids)

    # Good example #5: let's generate the same template using the HG interface
    # TODO: continue here, use the examples examples but with templates

    # Good example #6: Fibanacci solved easily 
    model_inputs = tokenizer(["Write the Fibonacci sequence generator. "], return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_length=64, max_new_tokens=64)
    print_llm_response(tokenizer, generated_ids)

    # Good example #7: quadratic equation solved easily 
    model_inputs = tokenizer(["Exercise #1: solve $x^2 - 4x + 1 = 0$ for x."], return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_length=64, max_new_tokens=64)
    print_llm_response(tokenizer, generated_ids)

    # Good example #8: quadratic equation solved easily + robustness
    model_inputs = tokenizer(["Exercise #1: solve $x^2 - 4x + 1 = 0$ for x. S"], return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_length=64, max_new_tokens=64)
    print_llm_response(tokenizer, generated_ids)


if __name__ == "__main__":
    #run_base_model()
    run_instruct_model()
