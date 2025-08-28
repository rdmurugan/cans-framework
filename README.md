# 🧠 CANS: Causal Adaptive Neural System

**CANS (Causal Adaptive Neural System)** is a **production-ready** deep learning framework for **causal inference** on structured, textual, and heterogeneous data. It seamlessly integrates **Graph Neural Networks (GNNs)**, **Transformers (e.g., BERT)**, a **Gated Fusion mechanism**, and **Counterfactual Regression Networks (CFRNet)**.

Initially developed for misinformation propagation on social networks, CANS generalizes to domains like **healthcare**, **legal**, and **finance**, offering a robust, well-tested pipeline for real-world causal modeling and counterfactual simulation.

## 🚀 What's New in v2.0

- 🔧 **Configuration Management**: Centralized, validated configs with JSON/YAML support
- 🛡️ **Enhanced Error Handling**: Comprehensive validation with informative error messages  
- 📊 **Logging & Checkpointing**: Built-in experiment tracking with automatic model saving
- 🧪 **Comprehensive Testing**: 100+ unit tests ensuring production reliability
- 📈 **Advanced Data Pipeline**: Multi-format loading (CSV, JSON) with automatic preprocessing
- ⚡ **Enhanced Training**: Early stopping, gradient clipping, multiple loss functions

## 🔧 Key Features

- ✅ **Production-Ready**: Robust error handling, logging, and comprehensive testing
- ✅ **Easy Configuration**: JSON/YAML configs with automatic validation  
- ✅ **Modular Design**: Plug in any GNN (GCN, GAT) and any Transformer (BERT, RoBERTa, etc.)
- ✅ **Smart Data Loading**: CSV, JSON, or synthetic data with automatic preprocessing
- ✅ **Gated Fusion Layer**: Learn optimal mixing of graph and textual signals
- ✅ **CFRNet Module**: Estimate potential outcomes μ₀, μ₁ under interventions
- ✅ **Counterfactual Analysis**: Built-in tools for treatment effect estimation
- ✅ **Experiment Tracking**: Automatic logging, checkpointing, and result visualization



## 🏗️ Architecture

```
 +-----------+     +-----------+
 |  GNN Emb  |     |  BERT Emb |
 +-----------+     +-----------+
        \             /
         \ Fusion Layer /
          \     /
         +-----------+
         |  Fused Rep |
         +-----------+
               |
           CFRNet
        /          \
   mu_0(x)       mu_1(x)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rdmurugan/cans-framework.git
cd cans-framework

# Install dependencies
pip install -r cans/requirements.txt.rtf
```

**Core Dependencies:**
- `torch>=2.0.0`
- `transformers>=4.38.0`
- `torch-geometric>=2.3.0`
- `scikit-learn>=1.3.0`
- `pandas>=2.0.0`

### Basic Usage (30 seconds to results)

```python
from cans.config import CANSConfig
from cans.utils.data import create_sample_dataset, get_data_loaders
from cans.models import CANS
from cans.models.gnn_modules import GCN
from cans.pipeline.runner import CANSRunner
from transformers import BertModel
import torch

# 1. Create configuration
config = CANSConfig()
config.training.epochs = 10

# 2. Load data (or create sample data)
datasets = create_sample_dataset(n_samples=1000, n_features=64)
train_loader, val_loader, test_loader = get_data_loaders(datasets, batch_size=32)

# 3. Create model
gnn = GCN(in_dim=64, hidden_dim=128, output_dim=256)
bert = BertModel.from_pretrained("bert-base-uncased")
model = CANS(gnn, bert, fusion_dim=256)

# 4. Train
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
runner = CANSRunner(model, optimizer, config)
history = runner.fit(train_loader, val_loader)

# 5. Evaluate
results = runner.evaluate(test_loader)
print(f"Test MSE: {results['mse']:.4f}")
print(f"Average Treatment Effect: {results['ate']:.4f}")
```

## 📊 Usage Examples

### Example 1: CSV Data with Real Causal Inference

```python
from cans.utils.data import load_csv_dataset
from cans.config import CANSConfig, DataConfig

# Configure data processing
config = CANSConfig()
config.data.graph_construction = "knn"  # or "similarity" 
config.data.knn_k = 5
config.data.scale_node_features = True

# Load your CSV data
datasets = load_csv_dataset(
    csv_path="your_data.csv",
    text_column="review_text",        # Column with text data
    treatment_column="intervention",   # Binary treatment (0/1)
    outcome_column="conversion_rate",  # Continuous outcome  
    feature_columns=["age", "income", "education"],  # Numerical features
    config=config.data
)

train_dataset, val_dataset, test_dataset = datasets

# Check data quality
stats = train_dataset.get_statistics()
print(f"Treatment proportion: {stats['treatment_proportion']:.3f}")
print(f"Propensity overlap valid: {stats['propensity_overlap_valid']}")
```

### Example 2: Advanced Configuration & Experiment Tracking

```python
from cans.config import CANSConfig

# Create detailed configuration
config = CANSConfig()

# Model configuration
config.model.gnn_type = "GCN"
config.model.gnn_hidden_dim = 256
config.model.fusion_dim = 512
config.model.text_model = "distilbert-base-uncased"  # Faster BERT variant

# Training configuration  
config.training.learning_rate = 0.001
config.training.batch_size = 64
config.training.epochs = 50
config.training.early_stopping_patience = 10
config.training.gradient_clip_norm = 1.0
config.training.loss_type = "huber"  # Robust to outliers

# Experiment tracking
config.experiment.experiment_name = "healthcare_causal_analysis"
config.experiment.save_every_n_epochs = 5
config.experiment.log_level = "INFO"

# Save configuration for reproducibility
config.save("experiment_config.json")

# Later: load and use
loaded_config = CANSConfig.load("experiment_config.json")
```

### Example 3: Counterfactual Analysis & Treatment Effects

```python
from cans.utils.causal import simulate_counterfactual
import numpy as np

# After training your model...
runner = CANSRunner(model, optimizer, config)
runner.fit(train_loader, val_loader)

# Comprehensive evaluation
test_metrics = runner.evaluate(test_loader)
print("Performance Metrics:")
for metric, value in test_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Counterfactual analysis
cf_control = simulate_counterfactual(model, test_loader, intervention=0)
cf_treatment = simulate_counterfactual(model, test_loader, intervention=1)

# Calculate causal effects
ate = np.mean(cf_treatment) - np.mean(cf_control)
print(f"\nCausal Analysis:")
print(f"Average Treatment Effect (ATE): {ate:.4f}")
print(f"Expected outcome under control: {np.mean(cf_control):.4f}")
print(f"Expected outcome under treatment: {np.mean(cf_treatment):.4f}")

# Individual treatment effects
individual_effects = np.array(cf_treatment) - np.array(cf_control)
print(f"Treatment effect std: {np.std(individual_effects):.4f}")
print(f"% benefiting from treatment: {(individual_effects > 0).mean()*100:.1f}%")
```

### Example 4: Custom Data Pipeline

```python
from cans.utils.preprocessing import DataPreprocessor, GraphBuilder
from cans.config import DataConfig
import pandas as pd

# Custom preprocessing pipeline
config = DataConfig()
config.graph_construction = "similarity"
config.similarity_threshold = 0.7
config.scale_node_features = True

preprocessor = DataPreprocessor(config)

# Process your DataFrame  
df = pd.read_csv("social_media_posts.csv")
dataset = preprocessor.process_tabular_data(
    data=df,
    text_column="post_content",
    treatment_column="fact_check_label",
    outcome_column="share_count",
    feature_columns=["user_followers", "post_length", "sentiment_score"],
    text_model="bert-base-uncased",
    max_text_length=256
)

# Split with custom ratios
train_ds, val_ds, test_ds = preprocessor.split_dataset(
    dataset, 
    train_size=0.7, 
    val_size=0.2, 
    test_size=0.1
)
```

## 🧪 Testing & Development

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories  
pytest tests/test_models.py -v        # Model tests
pytest tests/test_validation.py -v    # Validation tests
pytest tests/test_pipeline.py -v      # Training pipeline tests

# Run with coverage
pytest tests/ --cov=cans --cov-report=html

# Run example scripts
python examples/enhanced_usage_example.py
```


## 📁 Framework Structure

```
cans-framework/
├── cans/
│   ├── __init__.py              # Main imports
│   ├── config.py                # ✨ Configuration management
│   ├── exceptions.py            # ✨ Custom exceptions
│   ├── validation.py            # ✨ Data validation utilities
│   ├── models/
│   │   ├── cans.py             # Core CANS model (enhanced)
│   │   └── gnn_modules.py      # GNN implementations
│   ├── pipeline/
│   │   └── runner.py           # ✨ Enhanced training pipeline
│   └── utils/
│       ├── causal.py           # Counterfactual simulation
│       ├── data.py             # ✨ Enhanced data loading
│       ├── preprocessing.py     # ✨ Advanced preprocessing
│       ├── logging.py          # ✨ Structured logging
│       └── checkpointing.py    # ✨ Model checkpointing
├── tests/                       # ✨ Comprehensive test suite
├── examples/                    # Usage examples
└── CLAUDE.md                   # Development guide
```
**✨ = New/Enhanced in v2.0**

## 🎯 Use Cases & Applications

### Healthcare & Medical
```python
# Analyze treatment effectiveness with patient records + clinical notes
datasets = load_csv_dataset(
    csv_path="patient_outcomes.csv",
    text_column="clinical_notes",
    treatment_column="medication_type", 
    outcome_column="recovery_score",
    feature_columns=["age", "bmi", "comorbidities"]
)
```

### Marketing & A/B Testing  
```python
# Marketing campaign effectiveness with customer profiles + ad content
datasets = load_csv_dataset(
    csv_path="campaign_data.csv", 
    text_column="ad_content",
    treatment_column="campaign_variant",
    outcome_column="conversion_rate",
    feature_columns=["customer_ltv", "demographics", "behavior_score"]
)
```

### Social Media & Content Moderation
```python  
# Impact of content moderation on engagement
datasets = load_csv_dataset(
    csv_path="posts_data.csv",
    text_column="post_text", 
    treatment_column="moderation_action",
    outcome_column="engagement_score",
    feature_columns=["user_followers", "post_length", "sentiment"]
)
```

## 🔬 Research & Methodology

CANS implements state-of-the-art causal inference techniques:

- **Counterfactual Regression Networks (CFRNet)**: Learn representations that minimize treatment assignment bias
- **Gated Fusion**: Adaptively combine graph-structured and textual information  
- **Balanced Representation**: Minimize distributional differences between treatment groups
- **Propensity Score Validation**: Automatic overlap checking for reliable causal estimates

**Key Papers:**
- Shalit et al. "Estimating individual treatment effect: generalization bounds and algorithms" (ICML 2017)
- Yao et al. "Representation learning for treatment effect estimation from observational data" (NeurIPS 2018)

## 🚀 Performance & Scalability

- **Memory Efficient**: Optimized batch processing and gradient checkpointing
- **GPU Acceleration**: Full CUDA support with automatic device selection
- **Parallel Processing**: Multi-core data loading and preprocessing
- **Production Ready**: Comprehensive error handling and logging

**Benchmarks** (approximate, hardware-dependent):
- **Small**: 1K samples, 32 features → ~30 sec training  
- **Medium**: 100K samples, 128 features → ~10 min training
- **Large**: 1M+ samples → Scales with batch size and hardware

## 📚 Additional Resources

- **Documentation**: See `CLAUDE.md` for detailed development guide
- **Examples**: Check `examples/` directory for complete workflows
- **Tests**: `tests/` contains 100+ unit tests demonstrating usage
- **Issues**: Report bugs and feature requests on GitHub

## 🤝 Contributing

Contributions welcome! Please:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Add tests** for new functionality  
4. **Run tests**: `pytest tests/ -v`
5. **Submit** a pull request

Areas we'd love help with:
- Additional GNN architectures (GraphSAGE, Graph Transformers)
- More evaluation metrics for causal inference
- Integration with popular ML platforms (MLflow, Weights & Biases)
- Performance optimizations

## 👨‍🔬 Authors

**Durai Rajamanickam** – [@duraimuruganr](https://github.com/rdmurugan)

## 📜 License

MIT License. Free to use, modify, and distribute with attribution.

---

**Ready to get started?** Try the 30-second quick start above, or dive into the detailed examples! 🚀

