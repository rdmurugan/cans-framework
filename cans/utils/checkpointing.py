"""Checkpointing utilities for the CANS framework"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import shutil

from ..config import CANSConfig
from ..exceptions import CANSError


class ModelCheckpointer:
    """Handles model checkpointing and loading"""
    
    def __init__(self, checkpoint_dir: str, save_best_only: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_metric = float('inf')
        self.best_checkpoint_path = None
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch: int, 
                       metrics: Dict[str, float], config: CANSConfig,
                       is_best: bool = False, step: Optional[int] = None):
        """Save model checkpoint with metadata"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if step is not None:
                checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}_{timestamp}.pt"
            else:
                checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
            
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            
            # Prepare checkpoint data
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'timestamp': timestamp,
                'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {},
                'config': {
                    'model': config.model.__dict__,
                    'training': config.training.__dict__,
                    'data': config.data.__dict__
                }
            }
            
            # Add scheduler state if available
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Handle best model tracking
            if is_best:
                self._save_best_checkpoint(checkpoint_path, metrics.get('loss', float('inf')))
            
            return checkpoint_path
            
        except Exception as e:
            raise CANSError(f"Failed to save checkpoint: {str(e)}")
    
    def _save_best_checkpoint(self, checkpoint_path: Path, metric_value: float):
        """Save best model checkpoint"""
        if metric_value < self.best_metric:
            self.best_metric = metric_value
            
            # Remove old best checkpoint
            if self.best_checkpoint_path and self.best_checkpoint_path.exists():
                self.best_checkpoint_path.unlink()
            
            # Copy current checkpoint as best
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy2(checkpoint_path, best_path)
            self.best_checkpoint_path = best_path
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer=None, 
                       scheduler=None, device='cpu'):
        """Load model checkpoint"""
        try:
            checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            return {
                'epoch': checkpoint.get('epoch', 0),
                'step': checkpoint.get('step', 0),
                'metrics': checkpoint.get('metrics', {}),
                'model_info': checkpoint.get('model_info', {}),
                'config': checkpoint.get('config', {})
            }
            
        except Exception as e:
            raise CANSError(f"Failed to load checkpoint: {str(e)}")
    
    def load_best_checkpoint(self, model, optimizer=None, scheduler=None, device='cpu'):
        """Load best model checkpoint"""
        best_path = self.checkpoint_dir / "best_model.pt"
        
        if not best_path.exists():
            raise FileNotFoundError("No best checkpoint found")
        
        return self.load_checkpoint(best_path, model, optimizer, scheduler, device)
    
    def list_checkpoints(self) -> list:
        """List available checkpoints"""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pt"):
            try:
                # Load basic info without loading the full checkpoint
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                checkpoints.append({
                    'path': checkpoint_file,
                    'epoch': checkpoint.get('epoch', 0),
                    'step': checkpoint.get('step', 0),
                    'metrics': checkpoint.get('metrics', {}),
                    'timestamp': checkpoint.get('timestamp', '')
                })
            except:
                continue
        
        # Sort by epoch and step
        checkpoints.sort(key=lambda x: (x['epoch'], x.get('step', 0)))
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Remove old checkpoints, keeping only the last N"""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last_n:
            return
        
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-keep_last_n]:
            try:
                Path(checkpoint['path']).unlink()
            except:
                continue


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 monitor: str = 'loss', mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        """Check if training should stop early"""
        if self.monitor not in metrics:
            return False
        
        current_score = metrics[self.monitor]
        
        if self._is_better(current_score):
            self.best_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        self.should_stop = self.patience_counter >= self.patience
        return self.should_stop
    
    def _is_better(self, score: float) -> bool:
        """Check if current score is better than best"""
        if self.mode == 'min':
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get early stopping state"""
        return {
            'best_score': self.best_score,
            'patience_counter': self.patience_counter,
            'should_stop': self.should_stop
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load early stopping state"""
        self.best_score = state.get('best_score', self.best_score)
        self.patience_counter = state.get('patience_counter', 0)
        self.should_stop = state.get('should_stop', False)