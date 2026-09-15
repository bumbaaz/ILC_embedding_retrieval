# ILC_embedding_retrieval

This repository contains a small exploratory experiment using sentence embeddings to retrieve potentially relevant classes from the **Integrative Levels Classification (ILC)**. Given a text, the program compares its embedding with precomputed embeddings of ILC class descriptions and returns candidate classes ranked by semantic similarity.

The experiment is intended as **candidate generation rather than automatic classification**: semantic similarity does not guarantee classificatory validity, and the retrieved candidates still need to be evaluated within the structure and semantic constraints of ILC.

## Usage

A Python virtual environment is suggested.
Install the required Python packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

The example texts and other parameters can be changed directly in `main.py`.

## Links

ILC homepage
https://www.iskoi.org/ilc/index.php
