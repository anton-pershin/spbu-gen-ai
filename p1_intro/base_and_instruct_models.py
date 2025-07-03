from transformers import AutoModelForCausalLM, AutoTokenizer


# If not downloaded, HG will download it to
# ~/.cache/huggingface/hub
# Note that the cache format is not the same as the repo format
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt")
generated_ids = model.generate(**model_inputs, max_length=30)
tokenizer.batch_decode(generated_ids)[0]
generated_ids
