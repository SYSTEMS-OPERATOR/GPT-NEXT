# Program Flow Overview

This document outlines the high-level workflow of the project and shows how the main modules interact. The diagram summarizes the preprocessing, training and generation stages.

```mermaid
flowchart TD
    subgraph Preprocessing
        A1([Raw text files]) --> A2(train_bpe.py)
        A2 --> A3[Trained BytePairTokenizer]
        A1 --> A4(tokenize_dataset.py)
        A4 --> A5[Tokenized dataset]
    end

    subgraph Training
        B1[TokenIDDataset / TokenIDSubset] --> B2[DataLoader]
        B2 --> B3[Trainer]
        B3 --> B4[GPT Model]
        B3 --> B5[Checkpoint]
    end

    subgraph Generation
        C1[Checkpoint] --> C2[GPT Model]
        C2 --> C3[Sequencer]
        C3 --> C4(Output text)
    end

    A3 -->|used by| B1
    A5 -->|used by| B1
    B5 -->|loads| C2
```

## Stage descriptions

- **Preprocessing**
  - `train_bpe.py` trains a `BytePairTokenizer` from raw text files.
  - `tokenize_dataset.py` uses the trained tokenizer to convert the corpus into sequences of token IDs.

- **Training**
  - `TokenIDDataset`/`TokenIDSubset` create iterable datasets from the tokenized files.
  - `DataLoader` feeds batches to the `Trainer`.
  - `Trainer` runs the training loop on the `GPT` model, saving checkpoints for later use.

- **Generation**
  - The saved checkpoint is loaded into the `GPT` model.
  - The `Sequencer` performs autoregressive decoding to produce the final text output.
