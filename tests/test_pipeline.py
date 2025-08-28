"""Tests for pipeline and training functionality"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

from cans.pipeline.runner import CANSRunner
from cans.config import CANSConfig
from cans.utils.logging import CANSLogger
from cans.utils.checkpointing import ModelCheckpointer, EarlyStopping
from cans.exceptions import TrainingError


class TestCANSRunner:
    """Test CANS training runner"""
    
    def test_runner_initialization(self, cans_model, test_config):
        """Test runner initialization"""
        optimizer = torch.optim.Adam(cans_model.parameters())
        
        runner = CANSRunner(
            model=cans_model,
            optimizer=optimizer,
            config=test_config
        )
        
        assert runner.model == cans_model
        assert runner.optimizer == optimizer
        assert runner.config == test_config
        assert runner.device in ['cpu', 'cuda']
        assert runner.global_step == 0
        assert runner.start_epoch == 0
    
    def test_runner_with_scheduler(self, cans_model, test_config):
        """Test runner with learning rate scheduler"""
        optimizer = torch.optim.Adam(cans_model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
        
        runner = CANSRunner(
            model=cans_model,
            optimizer=optimizer,
            config=test_config,
            scheduler=scheduler
        )
        
        assert runner.scheduler == scheduler
    
    def test_runner_device_handling(self, cans_model, test_config):
        """Test device handling"""
        optimizer = torch.optim.Adam(cans_model.parameters())
        
        # Test explicit device
        runner = CANSRunner(
            model=cans_model,
            optimizer=optimizer,
            config=test_config,
            device='cpu'
        )
        
        assert runner.device == 'cpu'
        # Check model is on correct device
        assert next(cans_model.parameters()).device.type == 'cpu'
    
    def test_predict_functionality(self, cans_model, test_config, simple_dataloader):
        """Test prediction functionality"""
        optimizer = torch.optim.Adam(cans_model.parameters())
        runner = CANSRunner(cans_model, optimizer, test_config)
        
        results = runner.predict(simple_dataloader)
        
        assert 'predictions' in results
        assert 'targets' in results
        assert 'treatments' in results
        assert 'mu0' in results
        assert 'mu1' in results
        
        # Check shapes are consistent
        n_samples = len(results['predictions'])
        assert len(results['targets']) == n_samples
        assert len(results['treatments']) == n_samples
        assert len(results['mu0']) == n_samples
        assert len(results['mu1']) == n_samples
    
    def test_evaluate_functionality(self, cans_model, test_config, simple_dataloader):
        """Test evaluation functionality"""
        optimizer = torch.optim.Adam(cans_model.parameters())
        runner = CANSRunner(cans_model, optimizer, test_config)
        
        metrics = runner.evaluate(simple_dataloader)
        
        # Check required metrics
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'ate' in metrics  # Average Treatment Effect
        
        # Check values are reasonable
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['rmse'] == pytest.approx(metrics['mse'] ** 0.5)
    
    def test_training_single_epoch(self, cans_model, test_config, simple_dataloader):
        """Test training for single epoch"""
        test_config.training.epochs = 1
        optimizer = torch.optim.Adam(cans_model.parameters())
        runner = CANSRunner(cans_model, optimizer, test_config)
        
        # Mock the progress bars to avoid output during testing
        import cans.pipeline.runner
        original_tqdm = cans.pipeline.runner.tqdm
        cans.pipeline.runner.tqdm = lambda x, **kwargs: x
        
        try:
            history = runner.fit(simple_dataloader)
            
            assert len(history) == 1
            assert 'train_loss' in history[0]
            assert runner.global_step > 0
        
        finally:
            cans.pipeline.runner.tqdm = original_tqdm
    
    def test_training_with_validation(self, cans_model, test_config, simple_dataloader):
        """Test training with validation"""
        test_config.training.epochs = 1
        optimizer = torch.optim.Adam(cans_model.parameters())
        runner = CANSRunner(cans_model, optimizer, test_config)
        
        # Mock progress bars
        import cans.pipeline.runner
        original_tqdm = cans.pipeline.runner.tqdm
        cans.pipeline.runner.tqdm = lambda x, **kwargs: x
        
        try:
            history = runner.fit(simple_dataloader, val_loader=simple_dataloader)
            
            assert len(history) == 1
            assert 'train_loss' in history[0]
            assert 'val_loss' in history[0]
        
        finally:
            cans.pipeline.runner.tqdm = original_tqdm
    
    def test_gradient_clipping(self, cans_model, test_config, simple_dataloader):
        """Test gradient clipping functionality"""
        test_config.training.gradient_clip_norm = 1.0
        test_config.training.epochs = 1
        optimizer = torch.optim.Adam(cans_model.parameters())
        runner = CANSRunner(cans_model, optimizer, test_config)
        
        # Mock progress bars and run training
        import cans.pipeline.runner
        original_tqdm = cans.pipeline.runner.tqdm
        cans.pipeline.runner.tqdm = lambda x, **kwargs: x
        
        try:
            # Should not raise any errors
            runner.fit(simple_dataloader)
        
        finally:
            cans.pipeline.runner.tqdm = original_tqdm
    
    def test_different_loss_functions(self, cans_model, test_config, simple_dataloader):
        """Test different loss functions"""
        optimizer = torch.optim.Adam(cans_model.parameters())
        
        # Mock progress bars
        import cans.pipeline.runner
        original_tqdm = cans.pipeline.runner.tqdm
        cans.pipeline.runner.tqdm = lambda x, **kwargs: x
        
        try:
            for loss_type in ['mse', 'huber', 'mae']:
                test_config.training.loss_type = loss_type
                test_config.training.epochs = 1
                
                runner = CANSRunner(cans_model, optimizer, test_config)
                
                # Should not raise errors
                history = runner.fit(simple_dataloader)
                assert len(history) == 1
        
        finally:
            cans.pipeline.runner.tqdm = original_tqdm
    
    def test_training_history(self, cans_model, test_config, simple_dataloader):
        """Test training history tracking"""
        test_config.training.epochs = 2
        optimizer = torch.optim.Adam(cans_model.parameters())
        runner = CANSRunner(cans_model, optimizer, test_config)
        
        # Mock progress bars
        import cans.pipeline.runner
        original_tqdm = cans.pipeline.runner.tqdm
        cans.pipeline.runner.tqdm = lambda x, **kwargs: x
        
        try:
            history = runner.fit(simple_dataloader)
            
            assert len(history) == 2
            assert all('train_loss' in epoch_metrics for epoch_metrics in history)
            
            # Test getter method
            retrieved_history = runner.get_training_history()
            assert retrieved_history == history
        
        finally:
            cans.pipeline.runner.tqdm = original_tqdm


class TestEarlyStopping:
    """Test early stopping functionality"""
    
    def test_early_stopping_initialization(self):
        """Test early stopping initialization"""
        early_stopping = EarlyStopping(patience=5, min_delta=0.01)
        
        assert early_stopping.patience == 5
        assert early_stopping.min_delta == 0.01
        assert early_stopping.patience_counter == 0
        assert not early_stopping.should_stop
    
    def test_early_stopping_improvement(self):
        """Test early stopping with improving metrics"""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, monitor='loss')
        
        # Improving losses
        assert not early_stopping({'loss': 1.0})
        assert not early_stopping({'loss': 0.8})
        assert not early_stopping({'loss': 0.6})
        
        assert early_stopping.patience_counter == 0
        assert not early_stopping.should_stop
    
    def test_early_stopping_no_improvement(self):
        """Test early stopping without improvement"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, monitor='loss')
        
        # Set initial best score
        early_stopping({'loss': 1.0})
        
        # No improvement
        assert not early_stopping({'loss': 1.05})  # Slightly worse, counter = 1
        assert early_stopping({'loss': 1.08})      # Still worse, counter = 2, should stop
        
        assert early_stopping.should_stop
    
    def test_early_stopping_state_dict(self):
        """Test early stopping state preservation"""
        early_stopping = EarlyStopping(patience=3)
        early_stopping({'loss': 1.0})
        early_stopping({'loss': 1.1})  # No improvement
        
        # Save state
        state = early_stopping.state_dict()
        
        # Create new instance and load state
        new_early_stopping = EarlyStopping(patience=3)
        new_early_stopping.load_state_dict(state)
        
        assert new_early_stopping.best_score == early_stopping.best_score
        assert new_early_stopping.patience_counter == early_stopping.patience_counter


class TestModelCheckpointer:
    """Test model checkpointing functionality"""
    
    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_checkpointer_initialization(self, temp_checkpoint_dir):
        """Test checkpointer initialization"""
        checkpointer = ModelCheckpointer(temp_checkpoint_dir)
        
        assert checkpointer.checkpoint_dir == Path(temp_checkpoint_dir)
        assert checkpointer.checkpoint_dir.exists()
        assert checkpointer.save_best_only == True
    
    def test_save_checkpoint(self, cans_model, test_config, temp_checkpoint_dir):
        """Test saving checkpoint"""
        optimizer = torch.optim.Adam(cans_model.parameters())
        checkpointer = ModelCheckpointer(temp_checkpoint_dir)
        
        metrics = {'loss': 0.5, 'accuracy': 0.8}
        
        checkpoint_path = checkpointer.save_checkpoint(
            cans_model, optimizer, None, epoch=1,
            metrics=metrics, config=test_config
        )
        
        assert checkpoint_path.exists()
        assert 'checkpoint_epoch_1' in checkpoint_path.name
    
    def test_load_checkpoint(self, cans_model, test_config, temp_checkpoint_dir):
        """Test loading checkpoint"""
        optimizer = torch.optim.Adam(cans_model.parameters())
        checkpointer = ModelCheckpointer(temp_checkpoint_dir)
        
        # Save checkpoint
        metrics = {'loss': 0.5}
        checkpoint_path = checkpointer.save_checkpoint(
            cans_model, optimizer, None, epoch=5,
            metrics=metrics, config=test_config
        )
        
        # Create new model instance
        from cans.models.gnn_modules import GCN
        new_gnn = GCN(in_dim=128, hidden_dim=64, output_dim=256)
        
        # Mock text encoder for new model
        class MockBertConfig:
            def __init__(self):
                self.hidden_size = 768
        
        class MockBert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockBertConfig()
        
        new_text_encoder = MockBert()
        new_model = type(cans_model)(new_gnn, new_text_encoder, fusion_dim=256)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        # Load checkpoint
        checkpoint_info = checkpointer.load_checkpoint(
            checkpoint_path, new_model, new_optimizer
        )
        
        assert checkpoint_info['epoch'] == 5
        assert checkpoint_info['metrics']['loss'] == 0.5
    
    def test_list_checkpoints(self, cans_model, test_config, temp_checkpoint_dir):
        """Test listing checkpoints"""
        optimizer = torch.optim.Adam(cans_model.parameters())
        checkpointer = ModelCheckpointer(temp_checkpoint_dir)
        
        # Save multiple checkpoints
        for epoch in range(3):
            checkpointer.save_checkpoint(
                cans_model, optimizer, None, epoch=epoch,
                metrics={'loss': 1.0 - epoch * 0.1}, config=test_config
            )
        
        checkpoints = checkpointer.list_checkpoints()
        
        assert len(checkpoints) == 3
        assert all('epoch' in cp for cp in checkpoints)
        assert all('metrics' in cp for cp in checkpoints)
        
        # Should be sorted by epoch
        epochs = [cp['epoch'] for cp in checkpoints]
        assert epochs == sorted(epochs)
    
    def test_cleanup_old_checkpoints(self, cans_model, test_config, temp_checkpoint_dir):
        """Test cleanup of old checkpoints"""
        optimizer = torch.optim.Adam(cans_model.parameters())
        checkpointer = ModelCheckpointer(temp_checkpoint_dir)
        
        # Save multiple checkpoints
        for epoch in range(5):
            checkpointer.save_checkpoint(
                cans_model, optimizer, None, epoch=epoch,
                metrics={'loss': 1.0}, config=test_config
            )
        
        # Cleanup, keep only last 2
        checkpointer.cleanup_old_checkpoints(keep_last_n=2)
        
        remaining_checkpoints = checkpointer.list_checkpoints()
        assert len(remaining_checkpoints) <= 2