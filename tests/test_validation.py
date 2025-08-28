"""Tests for validation utilities"""

import pytest
import torch
from torch_geometric.data import Data

from cans.validation import DataValidator, ModelValidator, validate_propensity_overlap
from cans.exceptions import DataError, ValidationError


class TestDataValidator:
    """Test data validation utilities"""
    
    def test_valid_graph_data(self):
        """Test validation of valid graph data"""
        x = torch.randn(10, 128)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        graph_data = Data(x=x, edge_index=edge_index)
        
        assert DataValidator.validate_graph_data(graph_data) == True
    
    def test_invalid_graph_data_missing_x(self):
        """Test validation with missing node features"""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        graph_data = Data(edge_index=edge_index)
        
        with pytest.raises(DataError, match="missing node features"):
            DataValidator.validate_graph_data(graph_data)
    
    def test_invalid_graph_data_missing_edges(self):
        """Test validation with missing edge indices"""
        x = torch.randn(10, 128)
        graph_data = Data(x=x)
        
        with pytest.raises(DataError, match="missing edge indices"):
            DataValidator.validate_graph_data(graph_data)
    
    def test_invalid_graph_data_wrong_edge_shape(self):
        """Test validation with wrong edge index shape"""
        x = torch.randn(10, 128)
        edge_index = torch.tensor([0, 1, 2], dtype=torch.long)  # Wrong shape
        graph_data = Data(x=x, edge_index=edge_index)
        
        with pytest.raises(DataError, match="must be 2D"):
            DataValidator.validate_graph_data(graph_data)
    
    def test_invalid_graph_data_out_of_bounds_edges(self):
        """Test validation with out of bounds edge indices"""
        x = torch.randn(3, 128)  # Only 3 nodes
        edge_index = torch.tensor([[0, 1, 5], [1, 2, 0]], dtype=torch.long)  # Edge to node 5
        graph_data = Data(x=x, edge_index=edge_index)
        
        with pytest.raises(DataError, match="Edge index .* >= num_nodes"):
            DataValidator.validate_graph_data(graph_data)
    
    def test_invalid_graph_data_negative_edges(self):
        """Test validation with negative edge indices"""
        x = torch.randn(10, 128)
        edge_index = torch.tensor([[0, -1, 2], [1, 2, 0]], dtype=torch.long)  # Negative index
        graph_data = Data(x=x, edge_index=edge_index)
        
        with pytest.raises(DataError, match="cannot be negative"):
            DataValidator.validate_graph_data(graph_data)
    
    def test_valid_text_data(self):
        """Test validation of valid text data"""
        text_data = {
            'input_ids': torch.randint(1, 1000, (4, 32)),
            'attention_mask': torch.ones(4, 32)
        }
        
        assert DataValidator.validate_text_data(text_data) == True
    
    def test_invalid_text_data_missing_keys(self):
        """Test validation with missing required keys"""
        text_data = {
            'input_ids': torch.randint(1, 1000, (4, 32))
            # Missing attention_mask
        }
        
        with pytest.raises(DataError, match="missing required key"):
            DataValidator.validate_text_data(text_data)
    
    def test_invalid_text_data_wrong_type(self):
        """Test validation with wrong tensor types"""
        text_data = {
            'input_ids': [1, 2, 3],  # Not a tensor
            'attention_mask': torch.ones(4, 32)
        }
        
        with pytest.raises(DataError, match="must be torch.Tensor"):
            DataValidator.validate_text_data(text_data)
    
    def test_invalid_text_data_inconsistent_length(self):
        """Test validation with inconsistent sequence lengths"""
        text_data = {
            'input_ids': torch.randint(1, 1000, (4, 32)),
            'attention_mask': torch.ones(4, 16)  # Different seq length
        }
        
        with pytest.raises(DataError, match="Inconsistent sequence length"):
            DataValidator.validate_text_data(text_data)
    
    def test_valid_treatment_outcome(self):
        """Test validation of valid treatment and outcome"""
        treatment = torch.tensor([0, 1, 1, 0]).float()
        outcome = torch.randn(4)
        
        assert DataValidator.validate_treatment_outcome(treatment, outcome) == True
    
    def test_invalid_treatment_values(self):
        """Test validation with invalid treatment values"""
        treatment = torch.tensor([0, 1, 2, 0]).float()  # Invalid value 2
        outcome = torch.randn(4)
        
        with pytest.raises(DataError, match="Invalid treatment values"):
            DataValidator.validate_treatment_outcome(treatment, outcome)
    
    def test_mismatched_batch_sizes(self):
        """Test validation with mismatched batch sizes"""
        treatment = torch.tensor([0, 1, 1]).float()  # Size 3
        outcome = torch.randn(4)  # Size 4
        
        with pytest.raises(DataError, match="batch sizes don't match"):
            DataValidator.validate_treatment_outcome(treatment, outcome)
    
    def test_valid_complete_batch(self, sample_batch):
        """Test validation of complete batch"""
        assert DataValidator.validate_batch(sample_batch) == True
    
    def test_invalid_batch_missing_keys(self, sample_batch):
        """Test validation with missing batch keys"""
        del sample_batch['treatment']  # Remove required key
        
        with pytest.raises(DataError, match="missing required key"):
            DataValidator.validate_batch(sample_batch)


class TestModelValidator:
    """Test model validation utilities"""
    
    def test_valid_model_compatibility(self, gnn_module, text_encoder):
        """Test validation of compatible models"""
        assert ModelValidator.validate_model_compatibility(
            gnn_module, text_encoder, fusion_dim=256
        ) == True
    
    def test_invalid_gnn_no_output_dim(self, text_encoder):
        """Test validation with GNN missing output_dim"""
        class BadGNN:
            pass  # No output_dim attribute
        
        bad_gnn = BadGNN()
        
        with pytest.raises(ValidationError, match="must have 'output_dim'"):
            ModelValidator.validate_model_compatibility(
                bad_gnn, text_encoder, fusion_dim=256
            )
    
    def test_invalid_text_encoder_no_config(self, gnn_module):
        """Test validation with text encoder missing config"""
        class BadEncoder:
            pass  # No config attribute
        
        bad_encoder = BadEncoder()
        
        with pytest.raises(ValidationError, match="must have 'config'"):
            ModelValidator.validate_model_compatibility(
                gnn_module, bad_encoder, fusion_dim=256
            )
    
    def test_negative_dimensions(self, gnn_module, text_encoder):
        """Test validation with negative fusion dimension"""
        with pytest.raises(ValidationError, match="must be positive"):
            ModelValidator.validate_model_compatibility(
                gnn_module, text_encoder, fusion_dim=-1
            )
    
    def test_model_forward_validation(self, cans_model, sample_batch):
        """Test model forward pass validation"""
        assert ModelValidator.validate_model_forward(cans_model, sample_batch) == True
    
    def test_model_forward_validation_wrong_output_type(self, sample_batch):
        """Test model forward validation with wrong output type"""
        class BadModel:
            def eval(self):
                pass
            
            def __call__(self, *args):
                return "not a tensor", "also not a tensor"
        
        bad_model = BadModel()
        
        with pytest.raises(ValidationError, match="must return two torch.Tensors"):
            ModelValidator.validate_model_forward(bad_model, sample_batch)


class TestPropensityOverlap:
    """Test propensity overlap validation"""
    
    def test_good_overlap(self):
        """Test with good propensity overlap"""
        # Balanced treatment assignment
        treatment = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]).float()
        
        is_valid, stats = validate_propensity_overlap(treatment, threshold=0.1)
        
        assert is_valid == True
        assert stats['propensity_score'] == 0.5
        assert stats['overlap_satisfied'] == True
    
    def test_poor_overlap_too_few_treated(self):
        """Test with too few treated units"""
        # Very few treated units
        treatment = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1]).float()
        
        is_valid, stats = validate_propensity_overlap(treatment, threshold=0.2)
        
        assert is_valid == False
        assert stats['propensity_score'] == 0.125
        assert stats['overlap_satisfied'] == False
    
    def test_poor_overlap_too_few_control(self):
        """Test with too few control units"""
        # Very few control units
        treatment = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0]).float()
        
        is_valid, stats = validate_propensity_overlap(treatment, threshold=0.2)
        
        assert is_valid == False
        assert stats['propensity_score'] == 0.875
        assert stats['overlap_satisfied'] == False
    
    def test_edge_case_all_treated(self):
        """Test edge case with all treated units"""
        treatment = torch.ones(10).float()
        
        is_valid, stats = validate_propensity_overlap(treatment, threshold=0.1)
        
        assert is_valid == False
        assert stats['propensity_score'] == 1.0
        assert stats['min_group_size'] == 0
    
    def test_edge_case_all_control(self):
        """Test edge case with all control units"""
        treatment = torch.zeros(10).float()
        
        is_valid, stats = validate_propensity_overlap(treatment, threshold=0.1)
        
        assert is_valid == False
        assert stats['propensity_score'] == 0.0
        assert stats['min_group_size'] == 0