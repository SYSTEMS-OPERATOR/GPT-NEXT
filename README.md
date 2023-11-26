# GPT-1
Generative Pre-trained Transformer 1 (GPT-1)
![GPT-1](https://upload.wikimedia.org/wikipedia/commons/9/91/Full_GPT_architecture.png)

## Architecture
The GPT-1 architecture is a twelve-layer decoder-only transformer, utilizing twelve masked self-attention heads, with 64-dimensional states each (for a total of 768). The model utilizes the Adam optimization algorithm, diverging from simple stochastic gradient descent; the learning rate was progressively increased from zero over the first 2,000 updates to a peak of 2.5×10−4, followed by annealing to 0 using a cosine schedule.

## Training
GPT-1 was trained on the BooksCorpus dataset, containing over 7,000 unique unpublished books, amounting to nearly 800 million words. This extensive corpus provided a diverse range of vocabulary, narrative styles, and topics, enabling the model to develop a broad understanding of language patterns and structures.

## Capabilities
GPT-1 showcased remarkable capabilities in various natural language processing tasks, such as:
- Language modeling
- Machine translation
- Text summarization
- Question answering

The model's performance was particularly notable in tasks requiring contextual understanding and the generation of coherent, contextually relevant text.

## Implementation
This repository provides implementation details and resources for GPT-1. Users can utilize this model for various NLP tasks, adapting it to specific requirements and datasets.

### Getting Started
Instructions on how to set up and run GPT-1 in your environment are provided, along with examples of usage.

### Prerequisites
Details about necessary prerequisites, including software and hardware requirements.

### Installation
Step-by-step guide to installing and configuring GPT-1 on your system.

## Contributing
We welcome contributions from the community. Please refer to the [CONTRIBUTING.md](LINK_TO_YOUR_CONTRIBUTING.MD) for guidelines on how to contribute.

## Versioning
For the versions available, see the [tags on this repository](https://github.com/yourproject/tags).

## Authors and Acknowledgements
- [Your Name] - [Mind-Interfaces/GPT-1/](https://github.com/Mind-Interfaces/GPT-1/)
- Akshat Pandey - [Pytorch implementation of GPT-1](https://github.com/akshat0123/GPT-1/)
- Sosuke Kobayashi - [Homemade BookCorpus](https://github.com/soskek/bookcorpus)
- Acknowledgements to anyone whose resources were used

## License
This project is licensed under MIT - see the [LICENSE](LICENSE) file for details.
