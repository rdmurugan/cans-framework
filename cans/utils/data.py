# This is an exmaple file of loading Pheme dataset, you should follow similar procedure for test adn train dataframes.

import os
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
import networkx as nx
import random

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
