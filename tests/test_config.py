"""Tests for configuration management"""

import pytest
import tempfile
import json
from pathlib import Path

from cans.config import CANSConfig, ModelConfig, TrainingConfig, DataConfig, ExperimentConfig
from cans.exceptions import ConfigurationError


class TestModelConfig:
    """Test model configuration"""
    
    def test_default_model_config(self):
        """Test default model configuration values"""
        config = ModelConfig()
        
        assert config.gnn_type == "GCN"
        assert config.gnn_hidden_dim == 256
        assert config.fusion_dim == 256
        assert config.text_model == "bert-base-uncased"
    
    def test_custom_model_config(self):
        """Test custom model configuration"""
        config = ModelConfig(
            gnn_type="GAT",
            gnn_hidden_dim=512,
            fusion_dim=1024
        )
        
        assert config.gnn_type == "GAT"
        assert config.gnn_hidden_dim == 512
        assert config.fusion_dim == 1024


class TestTrainingConfig:
    """Test training configuration"""
    
    def test_default_training_config(self):
        """Test default training configuration values"""
        config = TrainingConfig()
        
        assert config.learning_rate == 0.001
        assert config.batch_size == 16
        assert config.epochs == 50
        assert config.scheduler_type == "cosine"
    
    def test_custom_training_config(self):
        """Test custom training configuration"""
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=32,
            epochs=100
        )
        
        assert config.learning_rate == 0.01
        assert config.batch_size == 32
        assert config.epochs == 100


class TestDataConfig:
    """Test data configuration"""
    
    def test_default_data_config(self):
        """Test default data configuration values"""
        config = DataConfig()
        
        assert config.train_split == 0.8
        assert config.val_split == 0.1
        assert config.test_split == 0.1
        assert config.random_seed == 42
    
    def test_custom_data_config(self):
        """Test custom data configuration"""
        config = DataConfig(
            train_split=0.7,
            val_split=0.2,
            test_split=0.1
        )
        
        assert config.train_split == 0.7
        assert config.val_split == 0.2
        assert config.test_split == 0.1


class TestExperimentConfig:
    """Test experiment configuration"""
    
    def test_default_experiment_config(self):
        """Test default experiment configuration values"""
        config = ExperimentConfig()
        
        assert config.experiment_name == "cans_experiment"
        assert config.log_level == "INFO"
        assert config.checkpoint_dir == "checkpoints"
        assert config.save_best_only == True
    
    def test_custom_experiment_config(self):
        """Test custom experiment configuration"""
        config = ExperimentConfig(
            experiment_name="my_experiment",
            log_level="DEBUG",
            save_best_only=False
        )
        
        assert config.experiment_name == "my_experiment"
        assert config.log_level == "DEBUG"
        assert config.save_best_only == False


class TestCANSConfig:
    """Test complete CANS configuration"""
    
    def test_default_cans_config(self):
        """Test default CANS configuration"""
        config = CANSConfig()
        
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.experiment, ExperimentConfig)
    
    def test_config_validation_valid(self):
        """Test validation of valid configuration"""
        config = CANSConfig()
        assert config.validate() == True
    
    def test_config_validation_invalid_splits(self):
        """Test validation with invalid data splits"""
        config = CANSConfig()
        config.data.train_split = 0.6
        config.data.val_split = 0.3
        config.data.test_split = 0.2  # Sum > 1.0
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            config.validate()
    
    def test_config_validation_dimension_mismatch(self):
        """Test validation with dimension mismatch"""
        config = CANSConfig()
        config.model.gnn_output_dim = 256
        config.model.fusion_dim = 512  # Should match gnn_output_dim
        
        with pytest.raises(ValueError, match="must match fusion dimension"):
            config.validate()
    
    def test_config_validation_negative_lr(self):
        """Test validation with negative learning rate"""
        config = CANSConfig()
        config.training.learning_rate = -0.01
        
        with pytest.raises(ValueError, match="must be positive"):
            config.validate()
    
    def test_config_validation_zero_batch_size(self):
        """Test validation with zero batch size"""
        config = CANSConfig()
        config.training.batch_size = 0
        
        with pytest.raises(ValueError, match="must be positive"):
            config.validate()
    
    def test_config_save_load_json(self):
        """Test saving and loading configuration to/from JSON"""
        config = CANSConfig()
        config.model.gnn_type = "GAT"
        config.training.learning_rate = 0.01
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save config
            config.save(temp_path)
            
            # Load config
            loaded_config = CANSConfig.load(temp_path)
            
            assert loaded_config.model.gnn_type == "GAT"
            assert loaded_config.training.learning_rate == 0.01
        
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_save_load_yaml(self):
        """Test saving and loading configuration to/from YAML"""
        pytest.importorskip("yaml")  # Skip if PyYAML not installed
        
        config = CANSConfig()
        config.model.gnn_type = "GraphSAGE"
        config.training.batch_size = 64
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save config
            config.save(temp_path)
            
            # Load config
            loaded_config = CANSConfig.load(temp_path)
            
            assert loaded_config.model.gnn_type == "GraphSAGE"
            assert loaded_config.training.batch_size == 64
        
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_save_unsupported_format(self):
        """Test saving configuration to unsupported format"""
        config = CANSConfig()
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                config.save(temp_path)
        
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_load_nonexistent_file(self):
        """Test loading from non-existent file"""
        with pytest.raises(FileNotFoundError):
            CANSConfig.load("nonexistent_file.json")
    
    def test_config_partial_loading(self):
        """Test loading partial configuration"""
        # Create partial config dict
        partial_config = {
            "model": {"gnn_type": "GAT"},
            "training": {"learning_rate": 0.005}
            # Missing data and experiment configs
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(partial_config, f)
            temp_path = f.name
        
        try:
            # Should load with defaults for missing sections
            loaded_config = CANSConfig.load(temp_path)
            
            assert loaded_config.model.gnn_type == "GAT"
            assert loaded_config.training.learning_rate == 0.005
            # Defaults should be used for missing values
            assert loaded_config.data.train_split == 0.8  # Default
            assert loaded_config.experiment.log_level == "INFO"  # Default
        
        finally:
            Path(temp_path).unlink(missing_ok=True)