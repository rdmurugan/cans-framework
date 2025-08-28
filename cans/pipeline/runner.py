import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from tqdm import tqdm
import time

from ..config import CANSConfig
from ..utils.logging import CANSLogger
from ..utils.checkpointing import ModelCheckpointer, EarlyStopping
from ..validation import DataValidator
from ..exceptions import TrainingError, ValidationError

class CANSRunner:
    def __init__(self, model, optimizer, config: CANSConfig, scheduler=None, 
                 logger: Optional[CANSLogger] = None, device=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logging
        self.logger = logger or CANSLogger(
            level=config.experiment.log_level,
            log_dir=config.experiment.log_dir if config.experiment.log_to_file else None
        )
        
        # Initialize checkpointing
        self.checkpointer = ModelCheckpointer(
            config.experiment.checkpoint_dir,
            save_best_only=config.experiment.save_best_only
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta
        )
        
        # Move model to device
        self.model.to(self.device)
        self.logger.info(f"Model moved to device: {self.device}")
        
        # Log model info
        if hasattr(self.model, 'get_model_info'):
            self.logger.log_model_info(self.model.get_model_info())
        
        # Training state
        self.global_step = 0
        self.start_epoch = 0
        self.training_history = []

    def fit(self, train_loader, val_loader=None, epochs: Optional[int] = None):
        """Enhanced training loop with logging, checkpointing, and early stopping"""
        try:
            epochs = epochs or self.config.training.epochs
            self.logger.info(f"Starting training for {epochs} epochs")
            
            self.model.train()
            
            for epoch in range(self.start_epoch, epochs):
                epoch_start_time = time.time()
                
                # Training phase
                train_metrics = self._train_epoch(train_loader, epoch)
                
                # Validation phase
                val_metrics = {}
                if val_loader is not None:
                    val_metrics = self._validate_epoch(val_loader, epoch)
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                self.training_history.append(epoch_metrics)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get('val_loss', train_metrics['train_loss']))
                    else:
                        self.scheduler.step()
                
                # Log epoch results
                epoch_time = time.time() - epoch_start_time
                self.logger.log_validation_results(epoch, **epoch_metrics, epoch_time=epoch_time)
                
                # Checkpointing
                is_best = self._is_best_epoch(epoch_metrics)
                if (epoch + 1) % self.config.experiment.save_every_n_epochs == 0 or is_best:
                    self.checkpointer.save_checkpoint(
                        self.model, self.optimizer, self.scheduler, epoch,
                        epoch_metrics, self.config, is_best
                    )
                
                # Early stopping
                if self.early_stopping(epoch_metrics):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            self.logger.info("Training completed successfully")
            return self.training_history
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise TrainingError(f"Training failed: {str(e)}")

    def predict(self, dataloader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions on a dataset"""
        self.model.eval()
        preds, trues, treatments = [], [], []
        mu0_preds, mu1_preds = [], []

        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Predicting"):
                    # Validate batch
                    DataValidator.validate_batch(batch)
                    
                    # Move to device
                    graph_data = batch['graph'].to(self.device)
                    text_input = {k: v.to(self.device) for k, v in batch['text'].items()}
                    treatment = batch['treatment'].to(self.device).float()
                    outcome = batch['outcome'].to(self.device).float()

                    # Forward pass
                    mu0, mu1 = self.model(graph_data, text_input, treatment)
                    y_pred = torch.where(treatment == 1, mu1, mu0).squeeze()

                    # Collect predictions
                    preds.extend(y_pred.cpu().numpy())
                    trues.extend(outcome.cpu().numpy())
                    treatments.extend(treatment.cpu().numpy())
                    mu0_preds.extend(mu0.squeeze().cpu().numpy())
                    mu1_preds.extend(mu1.squeeze().cpu().numpy())

            return {
                'predictions': np.array(preds),
                'targets': np.array(trues), 
                'treatments': np.array(treatments),
                'mu0': np.array(mu0_preds),
                'mu1': np.array(mu1_preds)
            }
            
        except Exception as e:
            raise TrainingError(f"Prediction failed: {str(e)}")

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Comprehensive evaluation with multiple metrics"""
        results = self.predict(dataloader)
        preds = results['predictions']
        trues = results['targets']
        
        # Regression metrics
        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)
        rmse = np.sqrt(mse)
        
        metrics = {
            'mse': mse,
            'mae': mae, 
            'rmse': rmse
        }
        
        # Classification metrics (if binary outcomes)
        unique_values = np.unique(trues)
        if len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values):
            preds_bin = (preds > 0.5).astype(int)
            trues_bin = trues.astype(int)
            
            try:
                metrics.update({
                    'f1': f1_score(trues_bin, preds_bin, zero_division=0),
                    'precision': precision_score(trues_bin, preds_bin, zero_division=0),
                    'recall': recall_score(trues_bin, preds_bin, zero_division=0),
                    'accuracy': np.mean(preds_bin == trues_bin)
                })
            except ValueError:
                # Handle edge cases in classification metrics
                pass
        
        # Causal inference metrics
        if 'mu0' in results and 'mu1' in results:
            ate = np.mean(results['mu1'] - results['mu0'])  # Average Treatment Effect
            metrics['ate'] = ate
        
        return metrics
    
    def _train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Validate batch
                DataValidator.validate_batch(batch)
                
                self.optimizer.zero_grad()
                
                # Move to device
                graph_data = batch['graph'].to(self.device)
                text_input = {k: v.to(self.device) for k, v in batch['text'].items()}
                treatment = batch['treatment'].to(self.device).float()
                outcome = batch['outcome'].to(self.device).float()
                
                # Forward pass
                mu0, mu1 = self.model(graph_data, text_input, treatment)
                y_pred = torch.where(treatment == 1, mu1, mu0).squeeze()
                
                # Compute loss
                if self.config.training.loss_type == 'mse':
                    loss = F.mse_loss(y_pred, outcome)
                elif self.config.training.loss_type == 'huber':
                    loss = F.huber_loss(y_pred, outcome)
                elif self.config.training.loss_type == 'mae':
                    loss = F.l1_loss(y_pred, outcome)
                else:
                    loss = F.mse_loss(y_pred, outcome)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_norm
                    )
                
                self.optimizer.step()
                
                total_loss += loss.item()
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
                # Log training step
                if self.global_step % 100 == 0:
                    self.logger.log_training_step(
                        epoch, self.global_step, loss.item()
                    )
                    
            except Exception as e:
                self.logger.error(f"Training step failed: {str(e)}")
                raise TrainingError(f"Training step failed: {str(e)}")
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def _validate_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1} Validation"):
                try:
                    # Move to device
                    graph_data = batch['graph'].to(self.device)
                    text_input = {k: v.to(self.device) for k, v in batch['text'].items()}
                    treatment = batch['treatment'].to(self.device).float()
                    outcome = batch['outcome'].to(self.device).float()
                    
                    # Forward pass
                    mu0, mu1 = self.model(graph_data, text_input, treatment)
                    y_pred = torch.where(treatment == 1, mu1, mu0).squeeze()
                    
                    # Compute loss
                    loss = F.mse_loss(y_pred, outcome)
                    total_loss += loss.item()
                    
                except Exception as e:
                    self.logger.error(f"Validation step failed: {str(e)}")
                    continue
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def _is_best_epoch(self, metrics: Dict[str, float]) -> bool:
        """Check if current epoch is the best so far"""
        val_loss = metrics.get('val_loss', metrics.get('train_loss', float('inf')))
        return val_loss < self.early_stopping.best_score
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       checkpoint_name: str = None) -> str:
        """Manually save a checkpoint"""
        return self.checkpointer.save_checkpoint(
            self.model, self.optimizer, self.scheduler, epoch,
            metrics, self.config, checkpoint_name=checkpoint_name
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a checkpoint and resume training state"""
        checkpoint_info = self.checkpointer.load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler, self.device
        )
        
        self.start_epoch = checkpoint_info['epoch'] + 1
        self.global_step = checkpoint_info.get('step', 0)
        
        self.logger.info(f"Resumed from checkpoint: epoch {checkpoint_info['epoch']}")
        return checkpoint_info
    
    def get_training_history(self) -> List[Dict[str, float]]:
        """Get training history"""
        return self.training_history
