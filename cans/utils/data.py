"""
Enhanced data loading utilities for CANS framework.
Includes both legacy PHEME loader and new preprocessing pipeline.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import networkx as nx
import random
from typing import Optional, Tuple, List, Dict, Any
import json
from pathlib import Path

from .preprocessing import DataPreprocessor, CANSDataset
from ..config import DataConfig
from ..exceptions import DataError

def load_pheme_graphs(data_dir, tokenizer, max_len=128, batch_size=16):
    """
    Loads rumor graphs and text for CANS training.

    Args:
        data_dir (str): Path to PHEME dataset
        tokenizer: HuggingFace tokenizer (e.g., BERT tokenizer)
        max_len (int): Max sequence length
        batch_size (int): Batch size for loaders

    Returns:
        train_loader, test_loader
    """
    data_list = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".json"): continue

            # Simulate example features
            x = torch.rand((10, 128))  # 10 nodes, 128-dim features
            edge_index = torch.tensor([[0,1,2,3], [1,2,3,4]], dtype=torch.long)  # Dummy edges
            treatment = torch.tensor(random.choice([0, 1]), dtype=torch.float)
            outcome = torch.tensor(random.choice([0.0, 1.0]), dtype=torch.float)

            # Simulate text
            text = "Rumor text goes here."
            text_input = tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")

            data = {
                'graph': Data(x=x, edge_index=edge_index),
                'text': {k: v.squeeze(0) for k, v in text_input.items()},
                'treatment': treatment,
                'outcome': outcome
            }
            data_list.append(data)

    # Train/test split
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

    class CausalGraphDataset(torch.utils.data.Dataset):
        def __init__(self, data_list):
            self.data_list = data_list

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            return self.data_list[idx]

    train_loader = DataLoader(CausalGraphDataset(train_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(CausalGraphDataset(test_data), batch_size=batch_size)

    return train_loader, test_loader


def load_csv_dataset(
    csv_path: str,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    feature_columns: Optional[List[str]] = None,
    config: Optional[DataConfig] = None,
    text_model: str = "bert-base-uncased",
    max_text_length: int = 128
) -> Tuple[CANSDataset, CANSDataset, CANSDataset]:
    """
    Load dataset from CSV file with enhanced preprocessing
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of column containing text data
        treatment_column: Name of column containing treatment assignments
        outcome_column: Name of column containing outcomes
        feature_columns: List of columns to use as node features
        config: Data configuration (uses defaults if None)
        text_model: HuggingFace model for text tokenization
        max_text_length: Maximum text sequence length
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    try:
        # Load data
        data = pd.read_csv(csv_path)
        
        # Use default config if not provided
        if config is None:
            config = DataConfig()
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Process data
        dataset = preprocessor.process_tabular_data(
            data=data,
            text_column=text_column,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            feature_columns=feature_columns,
            text_model=text_model,
            max_text_length=max_text_length
        )
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = preprocessor.split_dataset(dataset)
        
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        raise DataError(f"Failed to load CSV dataset: {str(e)}")


def load_json_dataset(
    json_path: str,
    config: Optional[DataConfig] = None,
    text_model: str = "bert-base-uncased"
) -> Tuple[CANSDataset, CANSDataset, CANSDataset]:
    """
    Load dataset from JSON file with structured data
    
    Expected JSON format:
    {
        "data": [
            {
                "text": "sample text",
                "treatment": 0 or 1,
                "outcome": float,
                "features": [list of numerical features],
                "graph": {
                    "nodes": [...],
                    "edges": [[source, target], ...]
                }
            },
            ...
        ]
    }
    
    Args:
        json_path: Path to JSON file
        config: Data configuration
        text_model: HuggingFace model for text tokenization
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        if 'data' not in json_data:
            raise DataError("JSON file must contain 'data' key with list of samples")
        
        samples = json_data['data']
        
        # Use default config if not provided
        if config is None:
            config = DataConfig()
        
        # Extract components
        texts = [sample['text'] for sample in samples]
        treatments = torch.tensor([sample['treatment'] for sample in samples], dtype=torch.float)
        outcomes = torch.tensor([sample['outcome'] for sample in samples], dtype=torch.float)
        
        # Initialize text processor
        tokenizer = AutoTokenizer.from_pretrained(text_model)
        text_data = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Process graphs
        graph_data = []
        for sample in samples:
            if 'graph' in sample:
                # Use provided graph structure
                graph_info = sample['graph']
                node_features = torch.tensor(graph_info['nodes'], dtype=torch.float)
                edge_list = graph_info['edges']
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                # Use features to create single-node graph
                features = sample.get('features', [1.0])  # Default feature if none provided
                node_features = torch.tensor([features], dtype=torch.float)
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            graph = Data(x=node_features, edge_index=edge_index)
            graph_data.append(graph)
        
        # Create dataset
        dataset = CANSDataset(
            graph_data=graph_data,
            text_data={
                'input_ids': text_data['input_ids'],
                'attention_mask': text_data['attention_mask'],
                'token_type_ids': text_data.get('token_type_ids', torch.zeros_like(text_data['input_ids']))
            },
            treatments=treatments,
            outcomes=outcomes,
            validate_data=True
        )
        
        # Split dataset
        preprocessor = DataPreprocessor(config)
        train_dataset, val_dataset, test_dataset = preprocessor.split_dataset(dataset)
        
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        raise DataError(f"Failed to load JSON dataset: {str(e)}")


def create_sample_dataset(
    n_samples: int = 1000,
    n_features: int = 32,
    text_length: int = 50,
    config: Optional[DataConfig] = None,
    random_seed: int = 42
) -> Tuple[CANSDataset, CANSDataset, CANSDataset]:
    """
    Create synthetic dataset for testing and experimentation
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of node features
        text_length: Average text length in words
        config: Data configuration
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Use default config if not provided
    if config is None:
        config = DataConfig()
    
    # Generate synthetic data
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    treatments = np.random.binomial(1, 0.5, n_samples).astype(np.float32)
    
    # Generate outcomes with treatment effect
    treatment_effect = 2.0
    noise = np.random.normal(0, 0.5, n_samples)
    outcomes = (features.mean(axis=1) + treatment_effect * treatments + noise).astype(np.float32)
    
    # Generate synthetic texts
    words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'and', 'runs', 'fast',
             'data', 'science', 'machine', 'learning', 'artificial', 'intelligence', 'model', 'prediction']
    
    texts = []
    for i in range(n_samples):
        n_words = max(5, np.random.poisson(text_length))
        text_words = np.random.choice(words, size=n_words, replace=True)
        texts.append(' '.join(text_words))
    
    # Create DataFrame for processing
    data = pd.DataFrame({
        'text': texts,
        'treatment': treatments,
        'outcome': outcomes,
        **{f'feature_{i}': features[:, i] for i in range(n_features)}
    })
    
    # Process with enhanced pipeline
    preprocessor = DataPreprocessor(config)
    dataset = preprocessor.process_tabular_data(
        data=data,
        text_column='text',
        treatment_column='treatment',
        outcome_column='outcome',
        feature_columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = preprocessor.split_dataset(dataset)
    
    return train_dataset, val_dataset, test_dataset


def get_data_loaders(
    datasets: Tuple[CANSDataset, CANSDataset, CANSDataset],
    batch_size: int = 16,
    shuffle_train: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders from datasets
    
    Args:
        datasets: Tuple of (train, val, test) datasets
        batch_size: Batch size for data loaders
        shuffle_train: Whether to shuffle training data
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset, val_dataset, test_dataset = datasets
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


# Legacy support - keep original function for backward compatibility
def load_legacy_pheme_graphs(data_dir, tokenizer, max_len=128, batch_size=16):
    """Legacy PHEME loader - use load_pheme_graphs instead"""
    return load_pheme_graphs(data_dir, tokenizer, max_len, batch_size)
