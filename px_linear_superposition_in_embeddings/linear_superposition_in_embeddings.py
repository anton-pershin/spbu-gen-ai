import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_rank(word_triple, word_db, word_db_embs, true_word):
    # word_triple = (w1, w2, w3)
    # we want to get w4 = w2 - w1 + w3
    word_triple_embs = []
    for i in range(len(word_triple)):
        word_triple_embs.append(word_db_embs[word_db.index(word_triple[i])])

    w4_emb = word_triple_embs[1] - word_triple_embs[0] + word_triple_embs[2]
    distances_btw_w4_and_all_embs = [
        torch.nn.functional.cosine_similarity(w4_emb.unsqueeze(0), w.unsqueeze(0)).item()
        for w in word_db_embs
    ]
    
    dists_with_indices = sorted(enumerate(distances_btw_w4_and_all_embs), key=lambda x: x[1], reverse=True)
    dists = [x[1] for x in dists_with_indices]
    indices = [x[0] for x in dists_with_indices]
    rank = indices.index(word_db.index(true_word)) + 1
    print(rank, word_db[indices[0]], word_db[indices[1]])


if __name__ == "__main__":
    plt.style.use("dark_background")

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype="auto", attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    word_db = [
        "king",
        "queen",
        "man",
        "woman",
        "apple",
        "apples",
        "plum",
        "plums",
        "car",
        "cars",
        "year",
        "years",
        "uncle",
        "aunt",
        "atmosphere",
    ]

    
    # Get tokens (adding space before each word)
    tokens = [tokenizer.encode(" " + word, add_special_tokens=False)[0] for word in word_db]
    
    # Get embeddings from the model's embedding matrix
    embedding_matrix = model.get_input_embeddings().weight
    word_db_embs = [embedding_matrix[token] for token in tokens]
    
    # Words for the analogy like: king - man = queen - woman
    test_cases = [
        {
            "word_triple": ["man", "king", "woman"],
            "true_word": "queen",
        },
        {
            "word_triple": ["apple", "apples", "car"],
            "true_word": "cars",
        },
        {
            "word_triple": ["year", "years", "apple"],
            "true_word": "apples",
        },
        {
            "word_triple": ["man", "uncle", "woman"],
            "true_word": "aunt",
        },
        {
            "word_triple": ["atmosphere", "uncle", "woman"],
            "true_word": "aunt",
        },
        {
            "word_triple": ["apple", "apples", "car"],
            "true_word": "cars",
        },
    ]


    for tc in test_cases:
        print(f"Test for {tc['true_word']}")

        compute_rank(
            word_triple=tc["word_triple"],
            word_db=word_db,
            word_db_embs=word_db_embs,
            true_word=tc["true_word"],
        )

