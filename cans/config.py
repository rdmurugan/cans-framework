from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    gnn_type: str = "GCN"  # GCN, GAT, GraphSAGE
    gnn_hidden_dim: int = 256
    gnn_output_dim: int = 256
    gnn_num_layers: int = 2
    gnn_dropout: float = 0.1
    
    fusion_dim: int = 256
    fusion_type: str = "gated"  # gated, attention, concat
    
    cfrnet_hidden_dim: int = 128
    cfrnet_num_layers: int = 2
    cfrnet_dropout: float = 0.1
    
    text_model: str = "bert-base-uncased"
    text_max_length: int = 128
    freeze_text_encoder: bool = False


@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 50
    weight_decay: float = 1e-5
    
    scheduler_type: str = "cosine"  # cosine, step, plateau
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    gradient_clip_norm: Optional[float] = 1.0
    
    # Loss function parameters
    loss_type: str = "mse"  # mse, huber, mae
    loss_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Data processing configuration"""
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    random_seed: int = 42
    shuffle_data: bool = True
    
    # Graph construction
    graph_construction: str = "manual"  # manual, knn, similarity
    knn_k: int = 5
    similarity_threshold: float = 0.5
    
    # Feature scaling
    scale_node_features: bool = True
    scale_outcomes: bool = False
    
    # Data validation
    validate_treatment_overlap: bool = True
    min_treatment_samples: int = 10


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration"""
    experiment_name: str = "cans_experiment"
    run_name: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    save_best_only: bool = True
    
    # Metrics tracking
    track_metrics: bool = True
    metrics_dir: str = "metrics"
    
    # Visualization
    plot_training_curves: bool = True
    plot_attention_weights: bool = False


@dataclass
class CANSConfig:
    """Complete CANS framework configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def save(self, path: str):
        """Save configuration to file"""
        path = Path(path)
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'experiment': self.experiment.__dict__
        }
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix in ['.yml', '.yaml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from file"""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yml', '.yaml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def validate(self):
        """Validate configuration parameters"""
        # Data split validation
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        
        # Model parameter validation
        if self.model.gnn_output_dim != self.model.fusion_dim:
            raise ValueError("GNN output dimension must match fusion dimension")
        
        # Training parameter validation
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        return True