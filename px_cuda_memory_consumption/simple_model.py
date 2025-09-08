import torch
from spbunn.mlp.simple_relu_nn import DeepReLUModel, ShallowReLUModel 


MAX_ENTRIES = 100000


def shallow_network_inference_mode():
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_ENTRIES
    )

    with torch.no_grad():
        # CUDA libraries:
        # 8 MiB
        # input layer:
        # 128 * 1024 * 4 bytes = 524288 bytes = 512 KiB
        # output layer:
        # 128 * 1024 * 4 bytes = 524288 bytes = 512 KiB
        # total model size:
        # 512 KiB + 512 KiB = 1 MiB
        model = ShallowReLUModel(input_dim=128, output_dim=128, hidden_layer_dim=1024, bias=False)
        # input tensor:
        # 2048 * 128 * 4 bytes = 1048576 bytes = 1 MiB
        x = torch.rand(2048, 128, dtype=torch.float32)
        # activation after input linear layer:
        # 2048 * 1024 * 4 bytes = 8388608 bytes = 8 MiB
        # activation after input ReLU layer:
        # 2048 * 1024 * 4 bytes = 8388608 bytes = 8 MiB
        # output tensor:
        # 128 * 1024 * 4 bytes = 524288 bytes = 512 KiB
        y = model(x)

        # Expected order (checked via memory_viz, that's correct):
        # 1. input layer (512 KiB)
        # 2. output layer (512 KiB)
        # => 1 MiB block
        # 3. input tensor (1 MiB)
        # 4. CUDA libraries (8.1 MiB, loaded at the first matmul, will not be freed)
        # 5. activation after input linear layer (8 MiB, should be freed)
        # 6. activation after input ReLU layer (8 MiB, should be freed)
        # 7. output tensor (1 MiB)

    try:
        torch.cuda.memory._dump_snapshot("shallow_network_inference_mode.pickle")
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")

    torch.cuda.memory._record_memory_history(enabled=None)


def shallow_network_backward_mode():
    # see shallow_network_inference_mode for memory breakdown till the backward step
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_ENTRIES
    )

    model = ShallowReLUModel(input_dim=128, output_dim=128, hidden_layer_dim=1024, bias=False)
    x = torch.rand(2048, 128, dtype=torch.float32)
    y = model(x)
    loss = torch.sum(y)
    # Expected order starting from backward
    # 1. gradient for output layer (512 KiB)
    # 2. gradient for input layer (512 KiB)

    # Reality:
    # 1. gradient for output layer (512 KiB)
    # 2. 1 MiB of something (freed quickly; I checked, it scales with output_dim only)
    # 3. 8.1 MiB of something (probably, again CUDA library)
    # 4. 8 MiB of something (will be freed; I checked it scales with hidden_layer_dim)
    # 5. 1 MiB of something (freed quickly; I checked, it scales with output_dim only)
    # 6. 8 MiB of something (will be freed; I checked it scales with hidden_layer_dim)
    # 7. gradient for input layer (512 KiB)

    loss.backward()

    try:
        torch.cuda.memory._dump_snapshot("shallow_network_backward_mode.pickle")
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")

    print("qwer")


def shallow_network_training_mode():
    # see shallow_network_backward_mode for memory breakdown till the optimizer step
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_ENTRIES
    )

    model = ShallowReLUModel(input_dim=128, output_dim=128, hidden_layer_dim=1024, bias=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(2048, 128, dtype=torch.float32)
    y = model(x)
    loss = torch.sum(y)
    loss.backward()

    # Expected order starting from optimizer (checked via memory_viz, that's correct)
    # 1. Adam states (1x3 MiB = 3MiB) 
    opt.step()

    try:
        torch.cuda.memory._dump_snapshot("shallow_network_training_mode.pickle")
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("This script should be used in the CUDA environment only")

    torch.set_default_device("cuda")

    # Comment out a mode you wish to check

    #shallow_network_inference_mode()
    #shallow_network_backward_mode()
    shallow_network_training_mode()

