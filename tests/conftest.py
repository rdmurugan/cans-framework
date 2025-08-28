"""Pytest configuration and fixtures for CANS framework tests"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import Dict, Any

from cans.models.cans import CANS
from cans.models.gnn_modules import GCN, GAT
from cans.config import CANSConfig
from cans.pipeline.runner import CANSRunner


@pytest.fixture
def device():
    """Get device for testing"""
    return torch.device('cpu')  # Use CPU for CI/testing


@pytest.fixture
def sample_graph_data():
    """Create sample graph data for testing"""
    x = torch.randn(10, 128)  # 10 nodes, 128 features
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


@pytest.fixture
def sample_text_data():
    """Create sample text data for testing"""
    batch_size = 4
    seq_len = 32
    
    return {
        'input_ids': torch.randint(1, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones((batch_size, seq_len), dtype=torch.long),
        'token_type_ids': torch.zeros((batch_size, seq_len), dtype=torch.long)
    }


@pytest.fixture
def sample_treatment_outcome():
    """Create sample treatment and outcome data"""
    batch_size = 4
    treatment = torch.randint(0, 2, (batch_size,)).float()
    outcome = torch.randn(batch_size)
    return treatment, outcome


@pytest.fixture
def sample_batch(sample_graph_data, sample_text_data, sample_treatment_outcome):
    """Create complete sample batch"""
    treatment, outcome = sample_treatment_outcome
    
    # Replicate graph data for batch
    batch_graphs = []
    for _ in range(4):
        batch_graphs.append(sample_graph_data)
    
    from torch_geometric.data import Batch
    batched_graph = Batch.from_data_list(batch_graphs)
    
    return {
        'graph': batched_graph,
        'text': sample_text_data,
        'treatment': treatment,
        'outcome': outcome
    }


@pytest.fixture
def gnn_module():
    """Create GNN module for testing"""
    return GCN(in_dim=128, hidden_dim=64, output_dim=256)


@pytest.fixture
def text_encoder():
    """Create text encoder for testing"""
    # Use a small BERT model for testing
    class MockBertConfig:
        def __init__(self):
            self.hidden_size = 768
    
    class MockBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockBertConfig()
            self.embeddings = nn.Embedding(1000, 768)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(768, 8, batch_first=True),
                num_layers=2
            )
        
        def forward(self, input_ids, attention_mask=None, **kwargs):
            embeddings = self.embeddings(input_ids)
            if attention_mask is not None:
                # Simple attention mask handling
                embeddings = embeddings * attention_mask.unsqueeze(-1)
            
            output = self.encoder(embeddings)
            
            # Mock BERT output structure
            class MockOutput:
                def __init__(self, last_hidden_state):
                    self.last_hidden_state = last_hidden_state
            
            return MockOutput(output)
    
    return MockBert()


@pytest.fixture
def cans_model(gnn_module, text_encoder):
    """Create CANS model for testing"""
    return CANS(gnn_module, text_encoder, fusion_dim=256)


@pytest.fixture
def test_config():
    """Create test configuration"""
    config = CANSConfig()
    
    # Override for testing
    config.training.epochs = 2
    config.training.batch_size = 4
    config.training.early_stopping_patience = 2
    config.experiment.log_level = "DEBUG"
    config.experiment.checkpoint_dir = "test_checkpoints"
    config.experiment.log_dir = "test_logs"
    
    return config


@pytest.fixture
def simple_dataloader(sample_batch):
    """Create simple dataloader for testing"""
    class SimpleDataset:
        def __init__(self, batch):
            self.batch = batch
        
        def __len__(self):
            return 2  # Return small number for testing
        
        def __getitem__(self, idx):
            return self.batch
    
    from torch.utils.data import DataLoader
    dataset = SimpleDataset(sample_batch)
    return DataLoader(dataset, batch_size=1, shuffle=False)


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_files():
    """Cleanup test files after session"""
    yield
    
    import shutil
    from pathlib import Path
    
    # Cleanup test directories
    test_dirs = ["test_checkpoints", "test_logs", "test_metrics"]
    for dir_name in test_dirs:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)