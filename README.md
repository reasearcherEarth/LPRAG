# **LP-RAG: Link Prediction for Retrieval-Augmented Generation**

> A preliminary version of LP-RAG, which builds a graph of chunks and synthetic queries and uses a GNN-based link predictor (NCN) for retrieval.

<p align="center">
  <a href="https://openreview.net/pdf?id=jbcxmanNWk" target="_blank">
    <img src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&style=flat-square" alt="arXiv">
  </a>
  <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-Datasets-yellow?logo=huggingface&style=flat-square" alt="HuggingFace">
  </a>
  <a href="https://github.com/reasearcherEarth/LPRAG" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?logo=github&style=flat-square" alt="GitHub">
  </a>
</p>

<!-- TODO: replace the "#" links above with the real paper / dataset / repo URLs. -->

---

## 🚀 **Highlights**
- ✅ **Graph-Based Retrieval**: Chunks and synthetic queries are connected in a single graph, so retrieval becomes a link prediction problem.
- ✅ **GNN Link Predictor**: Uses [Neural Common Neighbor (NCN)](https://github.com/GraphPKU/NeuralCommonNeighbor) to score query–chunk links.
- ✅ **Multi-Dataset Support**: Works with ICLR, MuSiQue, HotpotQA, MultiHop-RAG and RAG-QA through a single config file.

<!-- TODO: adjust the highlights to match the actual contributions of LP-RAG. -->

---

## 🛠️ **Usage**

### 1️⃣ Install Dependencies

**Step 1: Install Python packages**

```bash
pip install langchain_groq
pip install openai
```

**Step 2: Install PyTorch and PyTorch Geometric**

Follow the official guides:
- [PyTorch](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

**Step 3: Install GNN NCN**

NCN is required by `gnn.py`. Follow the [official repo](https://github.com/GraphPKU/NeuralCommonNeighbor).

**Step 4: Set up your Groq API key**

Create a key in the [Groq console](https://console.groq.com/keys) and paste it into the `groq_key` variable in `./src/inference_lp_rag.py`.

**Step 5: Download Datasets**

Download the dataset you want to work with:

| Dataset | Source |
|---------|--------|
| ICLR | [GraphEval](https://github.com/ulab-uiuc/GraphEval) |
| MuSiQue | [stonybrooknlp/musique](https://github.com/stonybrooknlp/musique) |
| HotpotQA | [hotpotqa.github.io](https://hotpotqa.github.io/) |
| MultiHop-RAG | [yixuantt/MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) |
| RAG-QA | [awslabs/rag-qa-arena](https://github.com/awslabs/rag-qa-arena) |

**Step 6: Configure the project**

Edit `./src/config.yaml` according to the dataset you want to use: set the dataset name and the appropriate file paths.

---

### 2️⃣ Quick Start Example

With `./src/config.yaml` already configured, run the full pipeline in order:

```bash
# 1. Generate chunks
python -m src.chunk_extraction

# 2. Generate synthetic queries
python -m src.synthetic_query_generation

# 3. Build the graph with chunks and synthetic queries
python -m src.build_graph

# 4. Train LP-RAG
python -m src.train_lp_rag

# 5. Run inference
python -m src.inference_lp_rag
```

---

## 🎯 **Performance**

<!-- TODO: add result figures, e.g.
<div align="center">
<img src="figure/generation_results.png" alt="results" width="1000">

**Main results of end-to-end performance**
</div>
-->

---

## 📬 Citation

If you find this work helpful, please consider citing us:

```bibtex
@article{lprag,
  title={LP-RAG: Learning to Retrieve with Link Predictors},
  author={do Nascimento, Erik Jhones Freitas and Franco, Jorge Luiz and Souza, Amauri H}
  journal={neurips},
  year={2026}
}
```
