# GPT-1
Generative Pre-trained Transformer 1 (GPT-1)

# Architecture
The GPT-1 architecture was a twelve-layer decoder-only transformer, using twelve masked self-attention heads, with 64-dimensional states each (for a total of 768). Rather than simple stochastic gradient descent, the Adam optimization algorithm was used; the learning rate was increased linearly from zero over the first 2,000 updates to a maximum of 2.5×10−4, and annealed to 0 using a cosine schedule.
