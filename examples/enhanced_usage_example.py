"""
Enhanced CANS Framework Usage Example

This example demonstrates the improved features including:
- Configuration management
- Enhanced error handling and validation
- Logging and checkpointing
- Comprehensive data preprocessing
- Advanced training pipeline
"""

import torch
import torch.optim as optim
from transformers import BertModel

# Import enhanced CANS components
from cans.config import CANSConfig
from cans.models import CANS
from cans.models.gnn_modules import GCN
from cans.pipeline.runner import CANSRunner
from cans.utils.data import create_sample_dataset, get_data_loaders
from cans.utils.logging import CANSLogger
from cans.utils.causal import simulate_counterfactual


def main():
    """Main example function"""
    
    # 1. Create and customize configuration
    print("=== Configuration Setup ===")
    config = CANSConfig()
    
    # Customize for our experiment
    config.model.gnn_type = "GCN"
    config.model.gnn_hidden_dim = 128
    config.model.fusion_dim = 256
    
    config.training.learning_rate = 0.001
    config.training.epochs = 10
    config.training.batch_size = 32
    config.training.early_stopping_patience = 5
    
    config.experiment.experiment_name = "enhanced_cans_demo"
    config.experiment.log_level = "INFO"
    
    # Validate configuration
    config.validate()
    print(f"✓ Configuration validated successfully")
    
    
    # 2. Create sample dataset with enhanced preprocessing
    print("\n=== Dataset Creation ===")
    train_dataset, val_dataset, test_dataset = create_sample_dataset(
        n_samples=1000,
        n_features=64,
        text_length=20,
        config=config.data,
        random_seed=42
    )
    
    # Print dataset statistics
    train_stats = train_dataset.get_statistics()
    print(f"Training set size: {train_stats['size']}")
    print(f"Treatment proportion: {train_stats['treatment_proportion']:.3f}")
    print(f"Propensity overlap valid: {train_stats['propensity_overlap_valid']}")
    
    
    # 3. Create data loaders
    datasets = (train_dataset, val_dataset, test_dataset)
    train_loader, val_loader, test_loader = get_data_loaders(
        datasets, 
        batch_size=config.training.batch_size,
        shuffle_train=True
    )
    
    print(f"✓ Data loaders created: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")
    
    
    # 4. Initialize enhanced model with configuration
    print("\n=== Model Setup ===")
    
    # Create GNN module
    gnn_module = GCN(
        in_dim=64,  # Match our synthetic features
        hidden_dim=config.model.gnn_hidden_dim,
        output_dim=config.model.gnn_output_dim
    )
    
    # Create text encoder (mock BERT for demo)
    class MockBert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {'hidden_size': 768})()
            self.embeddings = torch.nn.Embedding(1000, 768)
            self.encoder = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(768, 8, batch_first=True),
                num_layers=2
            )
        
        def forward(self, input_ids, attention_mask=None, **kwargs):
            embeddings = self.embeddings(input_ids)
            if attention_mask is not None:
                embeddings = embeddings * attention_mask.unsqueeze(-1)
            output = self.encoder(embeddings)
            return type('Output', (), {'last_hidden_state': output})()
    
    text_encoder = MockBert()
    
    # Create CANS model with enhanced configuration
    model = CANS(
        gnn_module=gnn_module,
        text_encoder=text_encoder,
        fusion_dim=config.model.fusion_dim,
        cfrnet_config={
            'hidden_dim': config.model.cfrnet_hidden_dim,
            'num_layers': config.model.cfrnet_num_layers,
            'dropout': config.model.cfrnet_dropout
        },
        fusion_config={
            'dropout': config.model.gnn_dropout
        }
    )
    
    # Print model information
    model_info = model.get_model_info()
    print(f"Model: {model_info['gnn_type']} + BERT")
    print(f"Parameters: {model_info['total_parameters']:,} total, {model_info['trainable_parameters']:,} trainable")
    
    
    # 5. Setup enhanced training pipeline
    print("\n=== Training Setup ===")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.training.epochs
    )
    
    # Initialize enhanced runner with logging and checkpointing
    runner = CANSRunner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device='cpu'  # Use CPU for demo
    )
    
    print("✓ Training pipeline initialized with logging and checkpointing")
    
    
    # 6. Train model with enhanced features
    print("\n=== Training ===")
    
    try:
        # Train with validation and early stopping
        training_history = runner.fit(
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        print(f"✓ Training completed successfully!")
        print(f"Training history: {len(training_history)} epochs")
        print(f"Final train loss: {training_history[-1]['train_loss']:.4f}")
        print(f"Final val loss: {training_history[-1]['val_loss']:.4f}")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return
    
    
    # 7. Comprehensive evaluation
    print("\n=== Evaluation ===")
    
    # Evaluate on test set
    test_metrics = runner.evaluate(test_loader)
    
    print("Test Set Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    
    # 8. Counterfactual analysis
    print("\n=== Counterfactual Analysis ===")
    
    try:
        # Simulate counterfactual outcomes
        cf_outcomes_control = simulate_counterfactual(model, test_loader, intervention=0)
        cf_outcomes_treatment = simulate_counterfactual(model, test_loader, intervention=1)
        
        # Calculate Average Treatment Effect (ATE)
        ate = torch.tensor(cf_outcomes_treatment).mean() - torch.tensor(cf_outcomes_control).mean()
        print(f"Estimated Average Treatment Effect (ATE): {ate:.4f}")
        
    except Exception as e:
        print(f"✗ Counterfactual analysis failed: {e}")
    
    
    # 9. Save final results
    print("\n=== Saving Results ===")
    
    try:
        # Save final checkpoint
        final_metrics = {**test_metrics, 'ate': ate.item() if 'ate' in locals() else 0.0}
        checkpoint_path = runner.save_checkpoint(
            epoch=len(training_history) - 1,
            metrics=final_metrics
        )
        print(f"✓ Final checkpoint saved: {checkpoint_path}")
        
        # Save configuration
        config.save("enhanced_experiment_config.json")
        print("✓ Configuration saved: enhanced_experiment_config.json")
        
    except Exception as e:
        print(f"Warning: Could not save results: {e}")
    
    
    print("\n=== Example Complete ===")
    print("Enhanced CANS framework demonstration completed successfully!")
    print("Key improvements showcased:")
    print("  ✓ Configuration management and validation")
    print("  ✓ Enhanced error handling throughout pipeline")
    print("  ✓ Comprehensive logging and checkpointing")
    print("  ✓ Robust data preprocessing with validation")
    print("  ✓ Advanced training with early stopping")
    print("  ✓ Comprehensive evaluation metrics")
    print("  ✓ Counterfactual analysis capabilities")


if __name__ == "__main__":
    main()