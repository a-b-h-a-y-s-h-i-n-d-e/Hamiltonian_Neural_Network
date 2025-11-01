import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:

    def __init__(self,
                model,

                learning_rate: float = 1e-3,
                device: str | None = None
            ):
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = model.to(self.device)

        self.learning_rate = learning_rate
        
        self.losses = []
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, x, y, epochs:int = 10, verbose: bool = False):
        x = x.to(self.device)
        y = y.to(self.device)
        
        for epoch in range(epochs):
            self.model.train()

            # forward pass
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

            # backward + optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.item())


            if verbose:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        
        min_loss = min(self.losses)
        print("--------------------------------------------------------")
        print("Summary : ")
        print(f"Total epochs : {epochs}, minimum loss : {min_loss:.4f}")

    def predict(self, x):
        self.model.eval()

        x = x.to(self.device)
        return self.model(x).cpu()

                
            