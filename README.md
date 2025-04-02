# CANS-Framework
**CANS (Causal Adaptive Neural System)** is a modular deep learning framework for **causal inference** on structured, textual, and heterogeneous data.

# 🧠 CANS: Causal Adaptive Neural System

**CANS (Causal Adaptive Neural System)** is a modular deep learning framework for **causal inference** on structured, textual, and heterogeneous data. It seamlessly integrates **Graph Neural Networks (GNNs)**, **Transformers (e.g., BERT)**, a **Gated Fusion mechanism**, and **Counterfactual Regression Networks (CFRNet)**.

Initially developed for misinformation propagation on social networks, CANS generalizes to domains like **healthcare**, **legal**, and **finance**, offering a plug-and-play pipeline for real-world causal modeling and counterfactual simulation.

---

## 🔧 Key Features

- ✅ **Modular Design**: Plug in any GNN (GCN, GAT, GraphSAGE) and any Transformer (BERT, RoBERTa, BioBERT, etc.)
- ✅ **Gated Fusion Layer**: Learn optimal mixing of graph and textual signals
- ✅ **CFRNet Module**: Estimate potential outcomes \(\mu_0, \mu_1\) under interventions
- ✅ **Counterfactual Simulation**: Evaluate interventions using `simulate_counterfactual()`
- ✅ **Dataset-Agnostic**: Works with PHEME, healthcare claims, legal documents, etc.
- ✅ **Ablation Support**: Easy toggles for GNN-only, BERT-only, or hybrid modeling



## 🏗️ Architecture

sql
Copy
Edit
 +-----------+     +-----------+
 |  GNN Emb  |     |  BERT Emb |
 +-----------+     +-----------+
        \\             //
         \\ Fusion Layer
          \\     //
         +-----------+
         |  Fused Rep |
         +-----------+
               |
           CFRNet
        /          \\
   mu_0(x)       mu_1(x)
yaml
Copy
Edit

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/cans-framework.git
cd cans-framework

# Install dependencies
pip install -r requirements.txt

Dependencies:
torch
transformers
torch-geometric
scikit-learn

🧩 Framework Modules
Module	Description
cans/models/cans.py	Core CANS model (GNN + BERT + Fusion + CFRNet)
cans/models/gnn_modules.py	Pluggable GCN, GAT GNNs
cans/pipeline/runner.py	Training, prediction, evaluation runner
cans/utils/data.py	Dataset loaders (e.g., PHEME graph builder)
cans/utils/causal.py	Counterfactual simulation & causal refutation
🧪 Example Usage
python
Copy
Edit
from cans.models import CANS
from cans.models.gnn_modules import GCN
from transformers import BertModel
from cans.pipeline import CANSRunner

# Create pluggable modules
gnn = GCN(in_dim=128, hidden_dim=256, output_dim=256)
bert = BertModel.from_pretrained("bert-base-uncased")

# Instantiate CANS
model = CANS(gnn_module=gnn, text_encoder=bert)

# Train with runner
runner = CANSRunner(model, optimizer=torch.optim.Adam(model.parameters()))
runner.fit(train_loader, epochs=5)
results = runner.evaluate(test_loader)

🧠 Counterfactual Simulation
Python
from cans.utils import simulate_counterfactual
cf_results = simulate_counterfactual(model, test_loader, intervention=0)


📂 Planned Extensions
✅ CANS++: Drop SCM, use CFRNet with interventional regularization
✅ Weakly Supervised Learning support
✅ DoWhy or econml integration
✅ Graph Construction Toolkit for any dataset
✅ Docker + Colab Demo


👨‍🔬 Authors
Durai Rajamanickam – @duraimuruganr

📜 License
MIT License. Free to use, modify, and distribute with attribution.

