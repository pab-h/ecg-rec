import torch.nn as nn

class ECGRecV1(nn.Module):

    def __init__(self, latentDim, hiddenDim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(12, hiddenDim, 5, stride=2, padding=2),
            nn.BatchNorm1d(hiddenDim),
            nn.ReLU(),

            nn.Conv1d(hiddenDim, latentDim, 5, stride=2, padding=2),
            nn.BatchNorm1d(latentDim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latentDim, hiddenDim, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(hiddenDim, 12, 4, stride=2, padding=1),
        )

    def forward(self, x):
        x  = x.permute(0, 2, 1)
        z   = self.encoder(x)
        out = self.decoder(z)
        
        return out.permute(0, 2, 1)

class ECGRecV2(nn.Module):

    def __init__(self, latentDim):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(12, 32, 5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, latentDim, 5, stride=2, padding=2),
            nn.BatchNorm1d(latentDim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latentDim, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 12, 4, stride=2, padding=1),
        )

    def forward(self, x):
        x  = x.permute(0, 2, 1)
        z   = self.encoder(x)
        out = self.decoder(z)
        
        return out.permute(0, 2, 1)
    