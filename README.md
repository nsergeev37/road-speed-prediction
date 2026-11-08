# Road Speed Prediction with Graph Machine Learning

A reproducible notebook comparing graph neural networks, graph-derived features, and gradient boosting for **node-level traffic speed regression** on the public [GraphLand `city-roads-L` dataset](https://zenodo.org/records/16895532).

## Task and dataset

Each node is a road segment; edges describe permitted transitions between segments. We predict the road segment's traffic speed using tabular node attributes and graph structure. Experiments use GraphLand's **`RH`** split (50% train / 25% validation / 25% test) in a **transductive** setting. The notebook converts the graph to an undirected representation for feature construction and message passing.

Dataset source: [GraphLand on Zenodo](https://zenodo.org/records/16895532). The data are downloaded by the notebook via PyTorch Geometric; **data files are not distributed in this repository**. Follow the source dataset's attribution and licensing requirements if redistributing any data.

## Methods

- **Graph Transformer** baseline, with and without DeepWalk-style structural embeddings learned via PyG `Node2Vec(p=q=1)`.
- **MLP** baselines on tabular features, DeepWalk embeddings and engineered graph statistics.
- **Neighborhood feature engineering:** one-hop mean/min/max/degree features, plus means over exact two- and three-hop neighborhoods.
- **Target-based graph features:** labeled-neighbor target means, counts and missing indicators by hop. The final method uses 5-fold **OOF** target encoding for training nodes and only training labels for validation/test nodes.
- **Gradient boosting:** LightGBM feature-group ablations and a comparison with XGBoost and CatBoost.

## Results

Results on `city-roads-L`, **RH** split. The following are saved experimental results from the notebook, not claimed results of a fresh independent run.

| Model | Test R² |
|---|---:|
| Graph Transformer | 0.5756 |
| Graph Transformer + DeepWalk | 0.6271 |
| MLP | 0.5285 |
| MLP + DeepWalk | 0.6091 |
| MLP + DeepWalk + neighborhood statistics | 0.6505 |
| MLP + DeepWalk + neighborhood + target statistics | 0.6701 |
| LightGBM + graph features | 0.6912 |
| **LightGBM + multi-hop features + OOF target encoding** | **0.7598** |

Final model: **MAE = 1.5664**, **RMSE = 2.1164** (in the target's original units). Results depend on the supplied split and training environment.

On the earlier graph-feature representation, the boosting comparison gave validation R² of **0.6940** (XGBoost), **0.6938** (LightGBM) and **0.6894** (CatBoost); LightGBM trained fastest in that run.

## Run in Google Colab

1. Open [`road_speed_graphland.ipynb`](road_speed_graphland.ipynb) in Google Colab.
2. Select a **GPU** runtime and execute the cells from top to bottom.
3. The setup cell installs packages and a `pyg-lib` wheel matched to Colab's PyTorch/CUDA version. If required by the runtime after package installation, restart and rerun the notebook.
4. PyG automatically downloads and processes the public dataset under `./data/`.

The full sequence includes GNN training, random-walk embedding training, feature engineering, ablations, and gradient boosting; it may require substantial compute and memory. The `requirements.txt` is a convenience reference for a local environment; the notebook's setup cell handles the special `pyg-lib` wheel separately.

## Leakage controls and limitations

For final multi-hop target encoding, the training nodes are divided into five folds. Features for each held-out fold use labels from the other four folds only; validation and test features use **training labels exclusively**. Other node attributes and graph topology are visible across all splits by design (transductive learning).

OOF encoding prevents direct reuse of a sample's own target, but train and evaluation features have different label availability. The random-node split also does not assess out-of-area geographical generalization. During development we inspected test results, so metrics are **exploratory** rather than a claim of untouched final evaluation.

## Acknowledgments

Dataset and benchmark: **GraphLand**, Yandex Research ([dataset record](https://zenodo.org/records/16895532)). This project was developed from coursework and extended with ablation experiments, boosting comparisons and multi-hop OOF feature engineering. Any externally sourced model or teaching-template code should retain its original attribution where applicable.
