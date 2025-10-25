from functools import partial
import matplotlib.pyplot as plt
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


def analyze_activations(model, tokenizer):
    """Analyze activation distributions and statistics during generation."""
    layer_stats = []
    
    # Prepare input
    prompt = "Hello, how are you?"
#    prompt = """When you drop a ball from rest it accelerates downward at 9.8 m/s^2. If you instead throw it
#downward assuming no air resistance its acceleration immediately after leaving your hand is
#(A) 9.8 m/s^2
#(B) more than 9.8 m/s^2
#(C) less than 9.8 m/s^2
#(D) Cannot say unless the speed of throw is given."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with output_hidden_states=True to get activations
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
    
    # Get hidden states from the first generation step
    hidden_states = outputs.hidden_states[0]  # First generation step
    
    # Analyze first 5 layers
    for layer_idx in range(len(hidden_states)):
        activations = hidden_states[layer_idx].detach().cpu().float()
        
        # Overall layer statistics
        activations_flat = activations.numpy().flatten()
        layer_stats.append({
            'layer': f'layer_{layer_idx}',
            'q10': float(np.percentile(activations_flat, 10)),
            'mean': float(np.mean(activations_flat)),
            'q90': float(np.percentile(activations_flat, 90)),
            'absmax': float(np.max(np.abs(activations_flat)))
        })
        
        # Per-channel statistics and plotting
        num_channels = activations.shape[-1]  # Last dimension is typically channels
        channel_activations = [activations[..., ch].numpy().flatten() for ch in range(num_channels)]
        
        q10s = [float(np.percentile(a, 10)) for a in channel_activations]
        means = [float(np.mean(a)) for a in channel_activations]
        q90s = [float(np.percentile(a, 90)) for a in channel_activations]
        
        # Plot channel statistics
        plt.figure(figsize=(10, 4))
        plt.plot(q10s, label='Q10', alpha=0.75)
        plt.plot(means, label='Mean', alpha=0.75)
        plt.plot(q90s, label='Q90', alpha=0.75)
        plt.yscale("symlog")
        plt.title(f'Channel Statistics - Layer {layer_idx}')
        plt.xlabel('Channel Index')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f'activation_channel_stats_layer_{layer_idx}.png', dpi=200)
        plt.close()
        
        # Plot histogram for this layer
        plt.figure(figsize=(6, 4))
        plt.hist(activations_flat, bins=50, alpha=0.75)
        plt.xscale("symlog")
        plt.title(f'Activation Distribution - Layer {layer_idx}')
        plt.xlabel('Activation Value')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'activation_dist_layer_{layer_idx}.png', dpi=200)
        plt.close()
    
    # Plot absmax trend across layers
    plt.figure(figsize=(10, 4))
    absmaxes = [stat['absmax'] for stat in layer_stats]
    plt.plot(absmaxes, marker='o')
    plt.title('Absolute Maximum Values Across Layers')
    plt.xlabel('Layer Index')
    plt.ylabel('Absolute Maximum Value')
    plt.grid(True, alpha=0.3)
    plt.savefig('activation_absmax_trend.png', dpi=200)
    plt.close()

    # Save layer statistics to file
    with open('activation_statistics.json', 'w') as f:
        json.dump(layer_stats, f, indent=2)


if __name__ == "__main__":
    plt.style.use("dark_background")

    print("Loading model...")
    model_name = "Qwen/Qwen2.5-7B"
#    model_name = "Qwen/Qwen3-8B"
#    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Analyzing activations...")
    analyze_activations(model, tokenizer)
    print("Analysis complete! Check activation_statistics.json for statistics and activation_dist_*.png files for distributions.")
