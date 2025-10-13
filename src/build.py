import torch
import os

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics  import r2_score
from dotenv           import load_dotenv
from Dataset          import Code15RandomLeadsDataset
from Model            import ECGReconstructor

import matplotlib.pyplot as plt
import torch.nn          as nn

import logging
import logging.config


# Log config

logging.config.fileConfig('logging.conf')

logger = logging.getLogger()

# Loading env

load_dotenv()

# Training definitions

SEED               = int(os.environ.get("SEED"))
EPOCHS             = int(os.environ.get("EPOCHS"))
DIST_DIR           = os.environ.get("DIST_DIR")
BATCH_SIZE         = int(os.environ.get("BATCH_SIZE"))
DATA_FOLDER        = os.environ.get("DATA_FOLDER")
SAMPLING_FREQUENCY = int(os.environ.get("SAMPLING_FREQUENCY"))

    
logger.info("Define dataset and dataloaders")

randomLeadsDataset = Code15RandomLeadsDataset(
    hdf5Files  = os.listdir(DATA_FOLDER),
    seed       = SEED
)

logger.info(f"Dataset lenght is {len(randomLeadsDataset)}")

dataloader = DataLoader(
    dataset    = randomLeadsDataset,
    batch_size = BATCH_SIZE,
    shuffle    = False,
)

# Holdout dataset

generator = torch.Generator().manual_seed(SEED)

trainSize = int(0.80 * len(randomLeadsDataset))
testSize  = len(randomLeadsDataset) - trainSize

logger.info(f"Dataset train lenght is {trainSize}")
logger.info(f"Dataset test lenght is {testSize}")

trainSet, testSet = random_split(
    randomLeadsDataset, 
    [trainSize, testSize], 
    generator = generator
)

# Dataloaders

trainDataloader = DataLoader(
    dataset     = trainSet,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
)

testDataloader = DataLoader(
    dataset     = testSet,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
)

# Model definition

logger.info("Define and compiling the model")

model = ECGReconstructor(
    latentDim = 128,
    hiddenDim = 32
)
model = torch.compile(model)

# GPU things

logger.info("Checking if the GPU is available")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Device = {device}")

model = model.to(device)

# The Training

logger.info("Starting the training!!")

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

trainingLoss = []
r2Scores     = []

model.train()

for epoch in range(EPOCHS):

    loss    = 0
    r2Score = 0

    for X, Y in trainDataloader:

        X, Y = X.to(device), Y.to(device)

        prediction = model(X)
        batchLoss  = criterion(prediction, Y)

        loss += batchLoss.item()

        YFlat          = Y.detach().cpu().flatten(0, 1).numpy()
        predictionFlat = prediction.detach().cpu().flatten(0, 1).numpy()

        r2Score += r2_score(YFlat, predictionFlat) / trainSize

        optimizer.zero_grad()
        batchLoss.backward()
        optimizer.step()

    trainingLoss.append(loss)
    r2Scores.append(r2Score)

    logger.info(f"Train - epoch = {epoch} loss = {loss: .5f} r2 = {r2Score: .5f}")

# Generating the training plot

if not os.path.exists(DIST_DIR):
    logger.warning("The dist folder does not exist. Creating")
    os.makedirs(DIST_DIR)

figure, axes = plt.subplots(nrows = 1, ncols = 2)

## Loss x Epoch

axes[0].scatter(range(EPOCHS), trainingLoss, c = "blue", marker = "x")

axes[0].set_title("Training Loss")

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

axes[0].grid()

## r2Score x Epoch

axes[1].scatter(range(EPOCHS), trainingLoss, c = "red", marker = "x")

axes[1].set_title("Training R²")

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("R²")

axes[1].grid()

plt.tight_layout()

trainingPlotPath = os.path.join(DIST_DIR, "training.png") 

plt.savefig(trainingPlotPath)

logger.info(f"Saving Training plot on {trainingPlotPath}")

logger.info("Starting the evaluate!!")

# The validation

model.eval()

testLoss    = 0
testR2Score = 0

with torch.no_grad():
    for X, Y in testDataloader:
          
        X, Y =  X.to(device), Y.to(device)
          
        prediction =  model(X)
        batchLoss  =  criterion(prediction, Y)
          
        testLoss   += batchLoss.item()

        YFlat          = Y.detach().cpu().flatten(0, 1).numpy()
        predictionFlat = prediction.detach().cpu().flatten(0, 1).numpy()

        testR2Score += r2_score(YFlat, predictionFlat) / trainSize

logger.info(f"Validation - loss = {testLoss: .5f} r2 = {testR2Score: .5f}")

# Deploy the model

modelDist = os.path.join(DIST_DIR, "model.pth")

torch.save(model.state_dict(), modelDist)
