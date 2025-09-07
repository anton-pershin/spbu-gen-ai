import torch
from spbunn.mlp.simple_relu_nn import ShallowReLUModel 


MAX_ENTRIES = 100000


if __name__ == "__main__":
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_ENTRIES
    )

    model = ShallowReLUModel(hidden_layer_dim=1024)
    x = torch.rand(128, dtype=torch.float32)
    y = model(x)

    try:
        torch.cuda.memory._dump_snapshot("simple_model_memory_snapshot.pickle")
    except:
        print(f"Failed to capture memory snapshot {e}")

    torch.cuda_memory._record_memory_history(enabled=None)
