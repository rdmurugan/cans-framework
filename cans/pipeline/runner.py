import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

class CANSRunner:
    def __init__(self, model, optimizer, device=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, dataloader, epochs=5):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                self.optimizer.zero_grad()

                # Move to device
                graph_data = batch['graph'].to(self.device)
                text_input = {k: v.to(self.device) for k, v in batch['text'].items()}
                treatment = batch['treatment'].to(self.device).float()
                outcome = batch['outcome'].to(self.device).float()

                # Forward pass
                mu0, mu1 = self.model(graph_data, text_input, treatment)
                y_pred = torch.where(treatment == 1, mu1, mu0).squeeze()
                loss = F.mse_loss(y_pred, outcome)

                # Backward + optimize
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")

    def predict(self, dataloader):
        self.model.eval()
        preds, trues, treatments = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                graph_data = batch['graph'].to(self.device)
                text_input = {k: v.to(self.device) for k, v in batch['text'].items()}
                treatment = batch['treatment'].to(self.device).float()
                outcome = batch['outcome'].to(self.device).float()

                mu0, mu1 = self.model(graph_data, text_input, treatment)
                y_pred = torch.where(treatment == 1, mu1, mu0).squeeze()

                preds.extend(y_pred.cpu().numpy())
                trues.extend(outcome.cpu().numpy())
                treatments.extend(treatment.cpu().numpy())

        return preds, trues, treatments

    def evaluate(self, dataloader):
        preds, trues, _ = self.predict(dataloader)
        preds_bin = [1 if p > 0.5 else 0 for p in preds]
        trues_bin = [1 if t > 0.5 else 0 for t in trues]

        return {
            "F1": f1_score(trues_bin, preds_bin),
            "Precision": precision_score(trues_bin, preds_bin),
            "Recall": recall_score(trues_bin, preds_bin)
        }
