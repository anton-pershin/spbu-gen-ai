# Sampling parameters

Most LLMs are featured with three sampling parameters: temperature, top_k and top_p. They are applied sequentially: the logits are first scaled by temperature, then top_k largest logits are selected, then they are softmaxed to get probabilities (but they will not sum to one because we dropped some tokens), and finally we pass only those the most probable tokens for top_k selected ones whose cumulative probability (their sum) is less then top_p. Finally, the remaining probabilities are re-normalized to sum to one and we start sampling from this discrete distribution.

Temperature scales the logits: $x <- x/T$
* `temperature = 0` implies greedy sampling
* `temperature \in (0; 1)` makes likely tokens even more likely
* `temperature > 1` makes the token distribution more uniform