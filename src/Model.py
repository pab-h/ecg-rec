import torch.nn as nn

class ECGReconstructor(nn.Module):

    def __init__(self, latentDim, hiddenDim):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(12, 6, 5, padding = 2),
            nn.ReLU(),
            nn.Conv1d(6, hiddenDim, 5, padding = 2),
            nn.ReLU(),
            nn.Conv1d(hiddenDim, latentDim, 5, padding = 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(latentDim, hiddenDim, 5, padding = 2),
            nn.ReLU(),
            nn.Conv1d(hiddenDim, 6, 5, padding = 2),
            nn.ReLU(),
            nn.Conv1d(6, 12, 5, padding = 2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  

        encoded = self.encoder(x) 
        decoded = self.decoder(encoded)
        decoded = decoded.permute(0, 2, 1)

        return decoded
    