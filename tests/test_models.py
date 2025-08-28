"""Tests for CANS model components"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from cans.models.cans import CANS, GatedFusion, CFRNet
from cans.models.gnn_modules import GCN, GAT
from cans.exceptions import DimensionMismatchError, ModelError, ValidationError


class TestGatedFusion:
    """Test GatedFusion layer"""
    
    def test_fusion_initialization(self):
        """Test fusion layer initialization"""
        fusion = GatedFusion(gnn_dim=256, bert_dim=768, fused_dim=512)
        
        assert fusion.gnn_dim == 256
        assert fusion.bert_dim == 768
        assert fusion.fused_dim == 512
    
    def test_fusion_invalid_dimensions(self):
        """Test fusion with invalid dimensions"""
        with pytest.raises(ValueError):
            GatedFusion(gnn_dim=-1, bert_dim=768, fused_dim=512)
        
        with pytest.raises(ValueError):
            GatedFusion(gnn_dim=256, bert_dim=0, fused_dim=512)
    
    def test_fusion_forward_pass(self):
        """Test fusion forward pass"""
        batch_size = 4
        gnn_dim, bert_dim, fused_dim = 256, 768, 512
        
        fusion = GatedFusion(gnn_dim, bert_dim, fused_dim)
        
        gnn_emb = torch.randn(batch_size, gnn_dim)
        bert_emb = torch.randn(batch_size, bert_dim)
        
        output = fusion(gnn_emb, bert_emb)
        
        assert output.shape == (batch_size, fused_dim)
        assert not torch.isnan(output).any()
    
    def test_fusion_dimension_mismatch(self):
        """Test fusion with mismatched dimensions"""
        fusion = GatedFusion(gnn_dim=256, bert_dim=768, fused_dim=512)
        
        gnn_emb = torch.randn(4, 128)  # Wrong dimension
        bert_emb = torch.randn(4, 768)
        
        with pytest.raises(DimensionMismatchError):
            fusion(gnn_emb, bert_emb)
    
    def test_fusion_batch_size_mismatch(self):
        """Test fusion with mismatched batch sizes"""
        fusion = GatedFusion(gnn_dim=256, bert_dim=768, fused_dim=512)
        
        gnn_emb = torch.randn(4, 256)
        bert_emb = torch.randn(2, 768)  # Different batch size
        
        with pytest.raises(DimensionMismatchError):
            fusion(gnn_emb, bert_emb)


class TestCFRNet:
    """Test CFRNet component"""
    
    def test_cfrnet_initialization(self):
        """Test CFRNet initialization"""
        cfrnet = CFRNet(input_dim=512, hidden_dim=128, num_layers=3)
        
        assert cfrnet.input_dim == 512
        assert cfrnet.hidden_dim == 128
        assert cfrnet.num_layers == 3
    
    def test_cfrnet_invalid_parameters(self):
        """Test CFRNet with invalid parameters"""
        with pytest.raises(ValueError):
            CFRNet(input_dim=-1, hidden_dim=128)
        
        with pytest.raises(ValueError):
            CFRNet(input_dim=512, hidden_dim=0)
        
        with pytest.raises(ValueError):
            CFRNet(input_dim=512, hidden_dim=128, num_layers=0)
    
    def test_cfrnet_forward_pass(self):
        """Test CFRNet forward pass"""
        batch_size = 4
        input_dim = 512
        
        cfrnet = CFRNet(input_dim, hidden_dim=128)
        
        x = torch.randn(batch_size, input_dim)
        t = torch.randint(0, 2, (batch_size,)).float()
        
        mu0, mu1 = cfrnet(x, t)
        
        assert mu0.shape == (batch_size, 1)
        assert mu1.shape == (batch_size, 1)
        assert not torch.isnan(mu0).any()
        assert not torch.isnan(mu1).any()
    
    def test_cfrnet_treatment_reshaping(self):
        """Test treatment tensor reshaping"""
        cfrnet = CFRNet(input_dim=512)
        
        x = torch.randn(4, 512)
        t_1d = torch.randint(0, 2, (4,)).float()  # 1D treatment
        t_2d = torch.randint(0, 2, (4, 1)).float()  # 2D treatment
        
        # Both should work
        mu0_1, mu1_1 = cfrnet(x, t_1d)
        mu0_2, mu1_2 = cfrnet(x, t_2d)
        
        assert mu0_1.shape == mu0_2.shape
        assert mu1_1.shape == mu1_2.shape


class TestCANSModel:
    """Test complete CANS model"""
    
    def test_cans_initialization(self, gnn_module, text_encoder):
        """Test CANS model initialization"""
        model = CANS(gnn_module, text_encoder, fusion_dim=256)
        
        assert model.gnn_dim == gnn_module.output_dim
        assert model.bert_dim == text_encoder.config.hidden_size
        assert model.fusion_dim == 256
    
    def test_cans_model_info(self, cans_model):
        """Test model info retrieval"""
        info = cans_model.get_model_info()
        
        assert 'gnn_type' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert info['total_parameters'] > 0
    
    def test_cans_forward_pass(self, cans_model, sample_batch, device):
        """Test CANS forward pass"""
        cans_model.to(device)
        
        # Move batch to device
        graph_data = sample_batch['graph'].to(device)
        text_input = {k: v.to(device) for k, v in sample_batch['text'].items()}
        treatment = sample_batch['treatment'].to(device)
        
        mu0, mu1 = cans_model(graph_data, text_input, treatment)
        
        batch_size = treatment.size(0)
        assert mu0.shape == (batch_size, 1)
        assert mu1.shape == (batch_size, 1)
        assert not torch.isnan(mu0).any()
        assert not torch.isnan(mu1).any()
    
    def test_cans_gradient_flow(self, cans_model, sample_batch, device):
        """Test gradient flow through model"""
        cans_model.to(device)
        cans_model.train()
        
        # Move batch to device
        graph_data = sample_batch['graph'].to(device)
        text_input = {k: v.to(device) for k, v in sample_batch['text'].items()}
        treatment = sample_batch['treatment'].to(device)
        outcome = sample_batch['outcome'].to(device)
        
        # Forward pass
        mu0, mu1 = cans_model(graph_data, text_input, treatment)
        y_pred = torch.where(treatment == 1, mu1, mu0).squeeze()
        loss = torch.nn.functional.mse_loss(y_pred, outcome)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        has_gradients = False
        for param in cans_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients, "No gradients found in model parameters"
    
    def test_cans_invalid_inputs(self, cans_model, device):
        """Test CANS with invalid inputs"""
        cans_model.to(device)
        
        # Invalid graph data
        invalid_graph = Data(x=torch.randn(5, 64))  # Missing edge_index
        text_input = {'input_ids': torch.randint(1, 1000, (4, 32))}
        treatment = torch.randint(0, 2, (4,)).float()
        
        with pytest.raises(Exception):  # Should raise validation error
            cans_model(invalid_graph, text_input, treatment)


class TestGNNModules:
    """Test GNN modules"""
    
    def test_gcn_initialization(self):
        """Test GCN initialization"""
        gcn = GCN(in_dim=128, hidden_dim=256, output_dim=512)
        
        assert gcn.output_dim == 512
        assert hasattr(gcn, 'conv1')
        assert hasattr(gcn, 'conv2')
    
    def test_gcn_forward_pass(self):
        """Test GCN forward pass"""
        gcn = GCN(in_dim=128, hidden_dim=256, output_dim=512)
        
        x = torch.randn(10, 128)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        graph_data = Data(x=x, edge_index=edge_index)
        
        output = gcn(graph_data)
        
        assert output.shape == (10, 512)
        assert not torch.isnan(output).any()
    
    def test_gat_initialization(self):
        """Test GAT initialization"""
        gat = GAT(in_dim=128, hidden_dim=256, output_dim=512, heads=4)
        
        assert gat.output_dim == 512
        assert hasattr(gat, 'gat1')
        assert hasattr(gat, 'gat2')
    
    def test_gat_forward_pass(self):
        """Test GAT forward pass"""
        gat = GAT(in_dim=128, hidden_dim=256, output_dim=512, heads=4)
        
        x = torch.randn(10, 128)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        graph_data = Data(x=x, edge_index=edge_index)
        
        output = gat(graph_data)
        
        assert output.shape == (10, 512)
        assert not torch.isnan(output).any()