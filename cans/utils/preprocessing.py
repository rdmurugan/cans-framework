"""Enhanced data preprocessing utilities for CANS framework"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset
from transformers import AutoTokenizer
import json

from ..exceptions import DataError
from ..validation import DataValidator, validate_propensity_overlap
from ..config import DataConfig


class GraphBuilder:
    """Build graphs from various data sources"""
    
    def __init__(self, method: str = "knn", **kwargs):
        """
        Initialize graph builder
        
        Args:
            method: Graph construction method ('knn', 'similarity', 'custom')
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.params = kwargs
        
        if method == "knn":
            self.k = kwargs.get('k', 5)
        elif method == "similarity":
            self.threshold = kwargs.get('threshold', 0.5)
            self.metric = kwargs.get('metric', 'cosine')
    
    def build_from_features(self, features: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph from node features
        
        Args:
            features: Node features array [num_nodes, num_features]
            
        Returns:
            Tuple of (node_features, edge_index)
        """
        if self.method == "knn":
            return self._build_knn_graph(features)
        elif self.method == "similarity":
            return self._build_similarity_graph(features)
        else:
            raise ValueError(f"Unknown graph construction method: {self.method}")
    
    def _build_knn_graph(self, features: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build k-nearest neighbors graph"""
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='ball_tree').fit(features)
        distances, indices = nbrs.kneighbors(features)
        
        # Remove self-loops (first neighbor is always self)
        indices = indices[:, 1:]
        
        # Build edge list
        edges = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:
                edges.append([i, neighbor])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(features, dtype=torch.float)
        
        return node_features, edge_index
    
    def _build_similarity_graph(self, features: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build similarity-based graph"""
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        
        if self.metric == 'cosine':
            similarity_matrix = cosine_similarity(features)
        elif self.metric == 'euclidean':
            dist_matrix = euclidean_distances(features)
            # Convert distance to similarity
            similarity_matrix = 1 / (1 + dist_matrix)
        else:
            raise ValueError(f"Unknown similarity metric: {self.metric}")
        
        # Create edges based on threshold
        edges = []
        num_nodes = len(features)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # Avoid self-loops and duplicates
                if similarity_matrix[i, j] > self.threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # Make undirected
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(features, dtype=torch.float)
        
        return node_features, edge_index
    
    def build_from_networkx(self, G: nx.Graph, node_features: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph from NetworkX graph
        
        Args:
            G: NetworkX graph
            node_features: Optional node features [num_nodes, num_features]
            
        Returns:
            Tuple of (node_features, edge_index)
        """
        # Convert NetworkX to edge list
        edges = list(G.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Handle node features
        if node_features is not None:
            if len(node_features) != G.number_of_nodes():
                raise ValueError("Number of node features must match number of nodes")
            features = torch.tensor(node_features, dtype=torch.float)
        else:
            # Use node degrees as features if none provided
            degrees = [G.degree(node) for node in G.nodes()]
            features = torch.tensor(degrees, dtype=torch.float).unsqueeze(1)
        
        return features, edge_index


class TextPreprocessor:
    """Preprocess text data for BERT-like models"""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        """
        Initialize text preprocessor
        
        Args:
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def preprocess_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Preprocess list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with tokenized inputs
        """
        if not texts:
            raise ValueError("Text list cannot be empty")
        
        # Tokenize all texts
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'token_type_ids': encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids']))
        }
    
    def preprocess_single_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess single text"""
        return self.preprocess_texts([text])


class FeatureScaler:
    """Scale numerical features"""
    
    def __init__(self, method: str = "standard"):
        """
        Initialize feature scaler
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
        """
        if method == "standard":
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit scaler and transform features"""
        return self.scaler.fit_transform(features)
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        return self.scaler.transform(features)
    
    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """Inverse transform scaled features"""
        return self.scaler.inverse_transform(features)


class CANSDataset(Dataset):
    """Enhanced dataset class for CANS framework"""
    
    def __init__(self, 
                 graph_data: List[Data],
                 text_data: Dict[str, torch.Tensor],
                 treatments: torch.Tensor,
                 outcomes: torch.Tensor,
                 validate_data: bool = True):
        """
        Initialize CANS dataset
        
        Args:
            graph_data: List of PyTorch Geometric Data objects
            text_data: Dictionary with tokenized text data
            treatments: Treatment assignments
            outcomes: Outcome values
            validate_data: Whether to validate data consistency
        """
        super().__init__()
        
        self.graph_data = graph_data
        self.text_data = text_data
        self.treatments = treatments
        self.outcomes = outcomes
        
        if validate_data:
            self._validate_data()
    
    def _validate_data(self):
        """Validate data consistency"""
        n_graphs = len(self.graph_data)
        n_texts = self.text_data['input_ids'].size(0)
        n_treatments = self.treatments.size(0)
        n_outcomes = self.outcomes.size(0)
        
        if not (n_graphs == n_texts == n_treatments == n_outcomes):
            raise DataError(
                f"Inconsistent data sizes: graphs={n_graphs}, texts={n_texts}, "
                f"treatments={n_treatments}, outcomes={n_outcomes}"
            )
        
        # Validate individual components
        for i, graph in enumerate(self.graph_data):
            try:
                DataValidator.validate_graph_data(graph)
            except Exception as e:
                raise DataError(f"Invalid graph data at index {i}: {str(e)}")
        
        try:
            DataValidator.validate_text_data(self.text_data)
            DataValidator.validate_treatment_outcome(self.treatments, self.outcomes)
        except Exception as e:
            raise DataError(f"Data validation failed: {str(e)}")
    
    def __len__(self) -> int:
        return len(self.graph_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get single data item"""
        return {
            'graph': self.graph_data[idx],
            'text': {
                'input_ids': self.text_data['input_ids'][idx],
                'attention_mask': self.text_data['attention_mask'][idx],
                'token_type_ids': self.text_data.get('token_type_ids', torch.zeros_like(self.text_data['input_ids']))[idx]
            },
            'treatment': self.treatments[idx],
            'outcome': self.outcomes[idx]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        treatment_prop = self.treatments.float().mean().item()
        outcome_mean = self.outcomes.float().mean().item()
        outcome_std = self.outcomes.float().std().item()
        
        # Check propensity overlap
        overlap_valid, overlap_stats = validate_propensity_overlap(self.treatments)
        
        return {
            'size': len(self),
            'treatment_proportion': treatment_prop,
            'outcome_mean': outcome_mean,
            'outcome_std': outcome_std,
            'propensity_overlap_valid': overlap_valid,
            'propensity_stats': overlap_stats
        }


class DataPreprocessor:
    """Main data preprocessing pipeline"""
    
    def __init__(self, config: DataConfig):
        """Initialize with data configuration"""
        self.config = config
        
        # Initialize components
        self.graph_builder = GraphBuilder(
            method=config.graph_construction,
            k=config.knn_k,
            threshold=config.similarity_threshold
        )
        self.text_processor = None  # Will be initialized when needed
        self.feature_scaler = FeatureScaler() if config.scale_node_features else None
        self.outcome_scaler = FeatureScaler() if config.scale_outcomes else None
    
    def process_tabular_data(self,
                           data: pd.DataFrame,
                           text_column: str,
                           treatment_column: str,
                           outcome_column: str,
                           feature_columns: Optional[List[str]] = None,
                           text_model: str = "bert-base-uncased",
                           max_text_length: int = 128) -> CANSDataset:
        """
        Process tabular data into CANS format
        
        Args:
            data: Input dataframe
            text_column: Name of text column
            treatment_column: Name of treatment column
            outcome_column: Name of outcome column
            feature_columns: List of feature columns (if None, use all numeric columns)
            text_model: HuggingFace model for tokenization
            max_text_length: Maximum text sequence length
            
        Returns:
            CANSDataset object
        """
        try:
            # Initialize text processor
            self.text_processor = TextPreprocessor(text_model, max_text_length)
            
            # Extract components
            texts = data[text_column].tolist()
            treatments = torch.tensor(data[treatment_column].values, dtype=torch.float)
            outcomes = torch.tensor(data[outcome_column].values, dtype=torch.float)
            
            # Process node features
            if feature_columns is None:
                feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                # Remove treatment and outcome from features
                feature_columns = [col for col in feature_columns 
                                 if col not in [treatment_column, outcome_column]]
            
            if not feature_columns:
                raise DataError("No numerical features found for graph construction")
            
            features = data[feature_columns].values.astype(np.float32)
            
            # Handle missing values
            if np.isnan(features).any():
                features = pd.DataFrame(features).fillna(0).values
            
            # Scale features if requested
            if self.feature_scaler is not None:
                features = self.feature_scaler.fit_transform(features)
            
            # Scale outcomes if requested
            if self.outcome_scaler is not None:
                outcomes_np = outcomes.numpy().reshape(-1, 1)
                outcomes_scaled = self.outcome_scaler.fit_transform(outcomes_np)
                outcomes = torch.tensor(outcomes_scaled.flatten(), dtype=torch.float)
            
            # Build graphs
            node_features, edge_index = self.graph_builder.build_from_features(features)
            
            # Create individual graph objects (assuming each row is a separate graph)
            graph_data = []
            for i in range(len(data)):
                # For tabular data, each sample gets its own single-node graph
                # This is a simplified approach - could be enhanced for multi-node scenarios
                single_node_features = node_features[i:i+1]  # Single node
                single_edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges for single node
                
                graph = Data(x=single_node_features, edge_index=single_edge_index)
                graph_data.append(graph)
            
            # Process text
            text_data = self.text_processor.preprocess_texts(texts)
            
            # Create dataset
            dataset = CANSDataset(
                graph_data=graph_data,
                text_data=text_data,
                treatments=treatments,
                outcomes=outcomes,
                validate_data=True
            )
            
            return dataset
            
        except Exception as e:
            raise DataError(f"Failed to process tabular data: {str(e)}")
    
    def split_dataset(self, 
                     dataset: CANSDataset,
                     train_size: Optional[float] = None,
                     val_size: Optional[float] = None,
                     test_size: Optional[float] = None,
                     random_state: Optional[int] = None) -> Tuple[CANSDataset, CANSDataset, CANSDataset]:
        """
        Split dataset into train/val/test
        
        Args:
            dataset: Input dataset
            train_size: Training set proportion
            val_size: Validation set proportion  
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Use config defaults if not specified
        train_size = train_size or self.config.train_split
        val_size = val_size or self.config.val_split
        test_size = test_size or self.config.test_split
        random_state = random_state or self.config.random_seed
        
        # Validate splits sum to 1
        total = train_size + val_size + test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split proportions must sum to 1.0, got {total}")
        
        indices = list(range(len(dataset)))
        
        # First split: separate test set
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=self.config.shuffle_data
        )
        
        # Second split: separate train and validation
        val_proportion = val_size / (train_size + val_size)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_proportion,
            random_state=random_state,
            shuffle=self.config.shuffle_data
        )
        
        # Create subset datasets
        def create_subset(indices):
            return CANSDataset(
                graph_data=[dataset.graph_data[i] for i in indices],
                text_data={
                    key: tensor[indices] for key, tensor in dataset.text_data.items()
                },
                treatments=dataset.treatments[indices],
                outcomes=dataset.outcomes[indices],
                validate_data=False  # Skip validation for subsets
            )
        
        train_dataset = create_subset(train_indices)
        val_dataset = create_subset(val_indices)
        test_dataset = create_subset(test_indices)
        
        return train_dataset, val_dataset, test_dataset