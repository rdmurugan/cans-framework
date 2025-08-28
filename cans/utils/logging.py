"""Logging utilities for the CANS framework"""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class CANSLogger:
    """Enhanced logger for CANS framework"""
    
    def __init__(self, name: str = "cans", level: str = "INFO", 
                 log_file: Optional[str] = None, log_dir: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file or log_dir:
            if log_dir:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                log_file = Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.error(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.debug(message)
    
    def log_model_info(self, model_info: dict):
        """Log model configuration information"""
        self.info("Model Configuration", **model_info)
    
    def log_training_step(self, epoch: int, step: int, loss: float, **metrics):
        """Log training step information"""
        self.info(f"Training Step", epoch=epoch, step=step, loss=loss, **metrics)
    
    def log_validation_results(self, epoch: int, **metrics):
        """Log validation results"""
        self.info(f"Validation Results", epoch=epoch, **metrics)
    
    def log_experiment_config(self, config: dict):
        """Log experiment configuration"""
        self.info("Experiment Configuration", config=config)