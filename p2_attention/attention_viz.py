from functools import partial

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bertviz import model_view, head_view
import scipy
import json


def print_llm_response(tokenizer, generated_ids) -> None:
    print()
    print(tokenizer.batch_decode(generated_ids)[0])
    print()


def generate_with_attention_output(tokenizer, model, prompt: str) -> tuple[list[str], torch.Tensor]:
    model_inputs = tokenizer([prompt], return_tensors="pt")
    # Just call one forward, no need for generation
    #outputs = model.generate(**model_inputs, return_dict_in_generate=True, max_new_tokens=1)
    outputs = model(**model_inputs, return_dict_in_generate=True, output_attentions=True, max_new_tokens=1)
    input_tokens = tokenizer.convert_ids_to_tokens(model_inputs.input_ids[0])
    return input_tokens, outputs.attentions


def plot_attention_score_distribution(attention):
    for l_i in range(len(attention)):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.hist(
            attention[l_i][0, :, :].reshape(-1).detach().to(dtype=torch.float32).numpy(),
            bins=20,
            density=True
        )
        ax.set_yscale("log")
        ax.set_xlabel("Attention scores across heads", fontsize=12)
        ax.set_ylabel("Fraction", fontsize=12)
        ax.grid()
        fig.savefig(f"attention_dist_layer_{l_i}.png", dpi=200)


def plot_attention_score_distribution_per_layer(attention, layer_i):
    l_i = layer_i
    for h_i in range(attention[l_i].shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.matshow(
            attention[l_i][0, h_i, :, :].detach().to(dtype=torch.float32).numpy(),
        )
        ax.set_ylabel("Token being attended", fontsize=12)
        fig.savefig(f"attention_dist_layer_{l_i}_head_{h_i}.png", dpi=200)


def dump_attention_entropy(attention, token_i):
    d = {
        "uniform": scipy.stats.entropy(token_i*[1./token_i]),
        "attention_heads": [],
    }
    for l_i in range(len(attention)):
        d["attention_heads"].append([])
        for h_i in range(attention[l_i].shape[1]):
            e = scipy.stats.entropy(attention[l_i][0, h_i, token_i, :token_i].detach().to(dtype=torch.float32).numpy())
            d["attention_heads"][-1].append(float(e))

    with open("attention_entropy.json", "w") as f:
        json.dump(d, f, indent=2)


if __name__ == "__main__":
    plt.style.use("dark_background")

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype="auto", attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

#    prompt = "Write the Fibonacci sequence generator."
#    prompt = "The lawyers advised the clients because they were knowledgeable."
    prompt = """When you drop a ball from rest it accelerates downward at 9.8 m/s^2. If you instead throw it
downward assuming no air resistance its acceleration immediately after leaving your hand is
(A) 9.8 m/s^2
(B) more than 9.8 m/s^2
(C) less than 9.8 m/s^2
(D) Cannot say unless the speed of throw is given."""
    input_tokens, attention = generate_with_attention_output(tokenizer, model, prompt)

    plot_attention_score_distribution_per_layer(attention, layer_i=23)   

    #plot_attention_score_distribution(attention)

    dump_attention_entropy(attention, token_i=6)

    ipython_html = model_view(attention, input_tokens, html_action="return")
    with open("model_view.html", "w") as f:
        f.write(ipython_html.data)

    ipython_html = head_view(attention, input_tokens, html_action="return")
    with open("head_view.html", "w") as f:
        f.write(ipython_html.data)
