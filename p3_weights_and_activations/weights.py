from functools import partial
import matplotlib.pyplot as plt
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


def analyze_weights(model):
    """Analyze weight distributions and statistics per layer."""
    layer_stats = []
    #layer_count = 0
    
    for name, param in model.named_parameters():
#        if layer_count >= 5:
#            break
            
        if 'weight' in name:
            #layer_count += 1
            weights = param.detach().cpu().float()
            
            # Overall layer statistics
            weights_flat = weights.numpy().flatten()
            layer_stats.append({
                'layer': name,
                'q10': float(np.percentile(weights_flat, 10)),
                'mean': float(np.mean(weights_flat)),
                'q90': float(np.percentile(weights_flat, 90)),
                'absmax': float(np.max(np.abs(weights_flat)))
            })
            
            # Per-channel statistics and plotting
            if "gate_proj" in name:
                print("qwer")
            if ("k_proj" in name) or ("v_proj" in name) or ("up_proj" in name) or ("gate_proj" in name):
                num_channels = weights.shape[1]
                channel_weights = [weights[:, ch].numpy().flatten() for ch in range(num_channels)]
            else:
                num_channels = weights.shape[0]
                channel_weights = [weights[ch].numpy().flatten() for ch in range(num_channels)]
            
            q10s = [float(np.percentile(w, 10)) for w in channel_weights]
            means = [float(np.mean(w)) for w in channel_weights]
            q90s = [float(np.percentile(w, 90)) for w in channel_weights]
            
            # Plot channel statistics
            plt.figure(figsize=(10, 4))
            plt.plot(q10s, label='Q10', alpha=0.75)
            plt.plot(means, label='Mean', alpha=0.75)
            plt.plot(q90s, label='Q90', alpha=0.75)
            plt.title(f'Channel Statistics - {name}')
            plt.xlabel('Channel Index')
            plt.ylabel('Value')
            plt.yscale("symlog")
            if "norm" not in name:
                plt.ylim(-0.2, 0.2)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'channel_stats_{name.replace(".", "_")}.png', dpi=200)
            plt.close()
            
            # Plot histogram for this layer
            plt.figure(figsize=(6, 4))
            plt.hist(weights_flat, bins=50, alpha=0.75)
            plt.title(f'Weight Distribution - {name}')
            plt.xlabel('Weight Value')
            plt.xscale("symlog")
            plt.ylabel('Count')
            if "norm" not in name:
                plt.xlim(-1, 1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'weight_dist_{name.replace(".", "_")}.png', dpi=200)
            plt.close()
    
    # Track absmax values by weight type
    kqvo_stats = {
        'key': {'layers': [], 'absmax': []},
        'query': {'layers': [], 'absmax': []},
        'value': {'layers': [], 'absmax': []},
        'output': {'layers': [], 'absmax': []},
        'up_proj': {'layers': [], 'absmax': []},
        'gate_proj': {'layers': [], 'absmax': []},
        'down_proj': {'layers': [], 'absmax': []}
    }

    # Categorize weights from layer_stats
    for stat in layer_stats:
        name = stat['layer']
        if 'k_proj' in name:
            kqvo_stats['key']['absmax'].append(stat['absmax'])
            kqvo_stats['key']['layers'].append(name)
        elif 'q_proj' in name:
            kqvo_stats['query']['absmax'].append(stat['absmax'])
            kqvo_stats['query']['layers'].append(name)
        elif 'v_proj' in name:
            kqvo_stats['value']['absmax'].append(stat['absmax'])
            kqvo_stats['value']['layers'].append(name)
        elif 'o_proj' in name:
            kqvo_stats['output']['absmax'].append(stat['absmax'])
            kqvo_stats['output']['layers'].append(name)
        elif 'up_proj' in name:
            kqvo_stats['up_proj']['absmax'].append(stat['absmax'])
            kqvo_stats['up_proj']['layers'].append(name)
        elif 'gate_proj' in name:
            kqvo_stats['gate_proj']['absmax'].append(stat['absmax'])
            kqvo_stats['gate_proj']['layers'].append(name)
        elif 'down_proj' in name:
            kqvo_stats['down_proj']['absmax'].append(stat['absmax'])
            kqvo_stats['down_proj']['layers'].append(name)

    # Plot absmax trends
    plt.figure(figsize=(12, 6))
    for weight_type, data in kqvo_stats.items():
        if data['absmax']:  # Only plot if we have data
            plt.plot(data['absmax'], marker='o', label=weight_type, alpha=0.75)
    plt.title('Absolute Maximum Values Across Layers by Weight Type')
    plt.xlabel('Layer Index')
    plt.ylabel('Absolute Maximum Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('weight_absmax_trends.png', dpi=200)
    plt.close()

    # Save layer statistics to file
    with open('weight_statistics.json', 'w') as f:
        json.dump(layer_stats, f, indent=2)


if __name__ == "__main__":
    plt.style.use("dark_background")

    print("Loading model...")
    model_name = "Qwen/Qwen2.5-7B"
#    model_name = "Qwen/Qwen3-8B"
#    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", attn_implementation="eager")
    
    print("Analyzing weights...")
    analyze_weights(model)
    print("Analysis complete! Check weight_statistics.json for statistics and weight_dist_*.png files for distributions.")

