# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CANS (Causal Adaptive Neural System) is a modular deep learning framework for causal inference on structured, textual, and heterogeneous data. It integrates Graph Neural Networks (GNNs), Transformers (BERT), Gated Fusion mechanism, and Counterfactual Regression Networks (CFRNet).

## Architecture

The framework follows a modular design with these key components:

1. **GNN Module** (`cans/models/gnn_modules.py`) - Pluggable GNN implementations (GCN, GAT)
2. **CANS Model** (`cans/models/cans.py`) - Core model integrating GNN, BERT, fusion layer, and CFRNet
3. **Pipeline Runner** (`cans/pipeline/runner.py`) - Training, prediction, and evaluation orchestration
4. **Data Utils** (`cans/utils/data.py`) - Dataset loading and graph construction utilities
5. **Causal Utils** (`cans/utils/causal.py`) - Counterfactual simulation functions

## Key Dependencies

Install dependencies with: `pip install -r cans/requirements.txt.rtf` (note: RTF format file)

Core dependencies:
- `torch>=2.0.0`
- `transformers>=4.38.0` 
- `torch-geometric>=2.3.0`
- `scikit-learn>=1.3.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`

## Usage Pattern

The framework follows this typical usage pattern:

1. **Create pluggable modules:**
   ```python
   from cans.models.gnn_modules import GCN
   from transformers import BertModel
   
   gnn = GCN(in_dim=128, hidden_dim=256, output_dim=256)
   bert = BertModel.from_pretrained("bert-base-uncased")
   ```

2. **Instantiate CANS model:**
   ```python
   from cans.models import CANS
   model = CANS(gnn_module=gnn, text_encoder=bert)
   ```

3. **Train with runner:**
   ```python
   from cans.pipeline import CANSRunner
   runner = CANSRunner(model, optimizer=torch.optim.Adam(model.parameters()))
   runner.fit(train_loader, epochs=5)
   ```

4. **Counterfactual simulation:**
   ```python
   from cans.utils import simulate_counterfactual
   cf_results = simulate_counterfactual(model, test_loader, intervention=0)
   ```

## Data Format Requirements

The framework expects data in a specific format:
- **Graph data:** PyTorch Geometric `Data` objects with `x` (node features) and `edge_index`
- **Text data:** Tokenized inputs compatible with HuggingFace transformers
- **Treatment:** Float tensor (0 or 1)
- **Outcome:** Float tensor (continuous or binary)

Each batch should contain: `{'graph': Data, 'text': dict, 'treatment': tensor, 'outcome': tensor}`

## Model Architecture Notes

- **GatedFusion:** Learns optimal mixing of GNN and BERT embeddings
- **CFRNet:** Estimates potential outcomes μ₀ and μ₁ under different treatments
- **Modular GNNs:** All GNN modules must have `output_dim` attribute for compatibility
- **Device handling:** Models automatically move to CUDA if available

## Enhanced Framework Features (v2.0)

### Configuration Management
- **Centralized config**: Use `CANSConfig` for all model, training, data, and experiment settings
- **Validation**: Automatic config validation with helpful error messages
- **Serialization**: Save/load configs in JSON or YAML format
- **Usage**: `config = CANSConfig.load('my_config.json')`

### Error Handling & Validation
- **Custom exceptions**: Specific error types (`DataError`, `ModelError`, `ValidationError`)
- **Input validation**: Automatic validation of data batches, model compatibility
- **Graceful failures**: Informative error messages with recovery suggestions

### Logging & Checkpointing
- **Structured logging**: Use `CANSLogger` for experiment tracking
- **Auto checkpointing**: Automatic model saving with configurable frequency
- **Early stopping**: Built-in early stopping with patience and minimum delta
- **Resume training**: Load checkpoints to resume interrupted training

### Enhanced Data Pipeline
- **Multiple formats**: Load from CSV, JSON, or create synthetic datasets
- **Auto preprocessing**: Automatic feature scaling, graph construction, text tokenization
- **Data validation**: Check propensity overlap, treatment balance, missing values
- **Usage**: `datasets = load_csv_dataset('data.csv', 'text_col', 'treatment_col', 'outcome_col')`

### Advanced Training
- **Enhanced runner**: `CANSRunner` with comprehensive metrics, progress tracking
- **Multiple loss functions**: MSE, Huber, MAE with configurable parameters
- **Gradient clipping**: Automatic gradient norm clipping for stability
- **Scheduler support**: Learning rate scheduling (cosine, step, plateau)

### Common Commands

```bash
# Run tests
pytest tests/ -v

# Run enhanced example
python examples/enhanced_usage_example.py

# Create sample dataset
python -c "from cans.utils.data import create_sample_dataset; create_sample_dataset(n_samples=1000)"
```

## Development Guidelines

- Use `CANSConfig` for all configuration management
- Always validate inputs using `DataValidator` and `ModelValidator`
- Implement proper error handling with custom exception types
- Use structured logging via `CANSLogger` for debugging and monitoring
- GNN modules must implement `forward(data)` and expose `output_dim` attribute
- Text encoders must be HuggingFace compatible with `config.hidden_size`
- Write comprehensive tests for new functionality