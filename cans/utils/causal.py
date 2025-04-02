import torch

def simulate_counterfactual(model, dataloader, intervention=0):
    """
    Simulates counterfactual outcomes for a given intervention.

    Args:
        model: Trained CANS model
        dataloader: DataLoader containing graph + text + treatment + outcome
        intervention: 0 or 1 (the desired counterfactual treatment)

    Returns:
        List of simulated counterfactual predictions
    """
    model.eval()
    device = next(model.parameters()).device
    cf_preds = []

    with torch.no_grad():
        for batch in dataloader:
            graph_data = batch['graph'].to(device)
            text_input = {k: v.to(device) for k, v in batch['text'].items()}
            treatment_cf = torch.full_like(batch['treatment'], intervention, dtype=torch.float).to(device)

            mu0, mu1 = model(graph_data, text_input, treatment_cf)
            y_cf = mu0 if intervention == 0 else mu1
            cf_preds.extend(y_cf.squeeze().cpu().numpy())

    return cf_preds

#how to use 
# cf_outcomes = simulate_counterfactual(model, test_loader, intervention=1)
