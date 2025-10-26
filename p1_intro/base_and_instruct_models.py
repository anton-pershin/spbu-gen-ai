from functools import partial

from transformers import AutoModelForCausalLM, AutoTokenizer


def print_llm_response(tokenizer, generated_ids) -> None:
    print()
    print(tokenizer.batch_decode(generated_ids)[0])
    print()


def generate_from_prompt(tokenizer, model, prompt: str) -> None:
    model_inputs = tokenizer([prompt], return_tensors="pt")
    generated_ids = model.generate(**model_inputs, max_length=64, max_new_tokens=64)
    print_llm_response(tokenizer, generated_ids)


def generate_from_chat(tokenizer, model, chat: list[dict[str, str]]) -> None:
    model_inputs = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    print_llm_response(tokenizer, generated_ids)


def repeat(func) -> None:
    for i in range(3):
        print(f"### Generation {i + 1} ###")
        func()


def run_base_model():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Bad example #1: base model will repeat stuff
    generate_from_prompt(tokenizer, model, "The secret to baking a good cake is ")

    # Bad example #2: base model will not follow the instruction
    generate_from_prompt(tokenizer, model, "Write the Fibonacci sequence generator. ")

    # Good example #3: base model will follow the instruction
    generate_from_prompt(tokenizer, model, "Exercise #1: solve $x^2 - 4x + 1 = 0$ for x")

    # Bad example #4: just one "period" added => base model does not follow the instruction
    generate_from_prompt(tokenizer, model, "Exercise #1: solve $x^2 - 4x + 1 = 0$ for x.")

    # Bad example #5: add more letters => base model diverges. Base models are just not robust
    generate_from_prompt(tokenizer, model, "Exercise #1: solve $x^2 - 4x + 1 = 0$ for x. S")

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
    generate_from_prompt(tokenizer, model, "\n".join(lines))


def run_instruct_model():
    # If not downloaded, HG will download it to
    # ~/.cache/huggingface/hub
    # Note that the cache format is not the same as the repo format
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


    # Neutral example #1: instruct model is better than base one but may still be a bit unguided
    # Here we have non-zero temperature, so let's repeat
    repeat(partial(generate_from_prompt, tokenizer, model, "The secret to baking a good cake is "))

    # Neutral example #2: let's generate the response with zero temperature and 
    # HuggingFace does not allow one to set `temperature=0` with `do_sample=True`
    # (the latter refers to "stochastic sampling") so we effectively set the zero 
    # temperature by setting `do_sample=False` and unsetting `temperature`, `top_p` 
    # and `top_k` which have been brought from `generation_config.json`
    model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt")
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

    # Good example #3: the template makes the  use improves when we follow the instruct template
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

    # Bad example #4: let's generate the same template using the HG interface and also require some format
    # Surprisingly, the LLM just refuses to answer => alignment
    chat = [
        {
            "role": "system",
            "content": 'You are a helpful assistant. Each message you write should end with the sentence "Thank you for your attention to this matter"',
        },
        {
            "role": "user",
            "content": "Write the Fibonacci sequence generator. ",
        },
    ]

    # Good example #5: once we specify our request more precisely, it works fine. But still does not follow the format
    prompt_as_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    print()
    print(prompt_as_text)
    print()
    repeat(partial(generate_from_chat, tokenizer, model, chat))

    chat = [
        {
            "role": "system",
            "content": 'You are a helpful assistant. Each message you write should end with the sentence "Thank you for your attention to this matter"',
        },
        {
            "role": "user",
            "content": "Write the python code generating the Fibonacci sequence till 100. ",
        },
    ]
    repeat(partial(generate_from_chat, tokenizer, model, chat))

    # Good example #6: we can enforce our format in the user prompt, and sometimes it follows
    chat = [
        {
            "role": "system",
            "content": 'You are a helpful assistant. Each message you write should end with the sentence "Thank you for your attention to this matter"',
        },
        {
            "role": "user",
            "content": "Write the python code generating the Fibonacci sequence till 100. End your message with 'Thank you for your attention to this matter'",
        },
    ]
    repeat(partial(generate_from_chat, tokenizer, model, chat))

    # Good example #7: quadratic equation solved easily 
    chat = [
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },
        {
            "role": "user",
            "content": "Exercise #1: solve $x^2 - 4x + 1 = 0$ for x",
        },
    ]
    repeat(partial(generate_from_chat, tokenizer, model, chat))

    # Good example #8: quadratic equation solved easily + robustness
    chat[1]["content"] = "Exercise #1: solve $x^2 - 4x + 1 = 0$ for x."
    repeat(partial(generate_from_chat, tokenizer, model, chat))

    # Good example #9: quadratic equation solved easily + robustness
    chat[1]["content"] = "Exercise #1: solve $x^2 - 4x + 1 = 0$ for x. S"
    repeat(partial(generate_from_chat, tokenizer, model, chat))

if __name__ == "__main__":
    #run_base_model()
    run_instruct_model()
