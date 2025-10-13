import os
import torch
import logging
import logging.config

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from Dataset          import Code15RandomLeadsDataset
from Model            import ECGReconstructor
from dotenv           import load_dotenv
from scipy.stats      import pearsonr
from sklearn.metrics  import r2_score
from utils            import plotECG
from utils            import methodComparativePlot
from utils            import comparativeFullEcgPlot

# Log config

logging.config.fileConfig('logging.conf')

logger = logging.getLogger()

# Loading env

load_dotenv()

SEED        = int(os.environ.get("SEED"))
DIST_DIR    = os.environ.get("DIST_DIR")
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE"))
DATA_FOLDER = os.environ.get("DATA_FOLDER")

# Define dataset

logger.info("Define dataset and dataloaders")

randomLeadsDataset = Code15RandomLeadsDataset(
    hdf5Files  = os.listdir(DATA_FOLDER),
    seed       = SEED
)

datasetLen = len(randomLeadsDataset)

dataloader = DataLoader(
    dataset     = randomLeadsDataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False
)

logger.info(f"Dataset lenght is {datasetLen}")

# Model definition

logger.info("Loading and compiling the model")

model = ECGReconstructor(
    latentDim = 128,
    hiddenDim = 32
)

model = torch.compile(model)

modelPath = os.path.join(DIST_DIR, "model.pth")

model.load_state_dict(
    torch.load(modelPath, weights_only = True)
)

# GPU things

logger.info("Checking if the GPU is available")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Device = {device}")

model = model.to(device)

# Plot configurations

ecgColumns = [
    "LI", 
    "LII", 
    "LIII", 
    "aVR", 
    "aVL",
    "aVF", 
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6"
]

ecgPlotColors = {
    "LI":   "seagreen",
    "aVR":  "black",
    "V1":   "gold",
    "V4":   "orangered",
    "LII":  "cornflowerblue",
    "aVL":  "seagreen",
    "V2":   "gold",
    "V5":   "crimson",
    "LIII": "cornflowerblue",
    "aVF":  "cornflowerblue",
    "V3":   "orangered",
    "V6":   "crimson"
}

# Evaluate

logger.info("Starting the evaluation")

r2Scores = pd.DataFrame(
    columns = ecgColumns,
    index   = range(datasetLen),
    data    = np.zeros((datasetLen, len(ecgColumns)))
)

correlations = pd.DataFrame(
    columns = ecgColumns,
    index   = range(datasetLen),
    data    = np.zeros((datasetLen, len(ecgColumns)))
)

model.eval()

with torch.no_grad():
    for i, (X, Y) in enumerate(dataloader):
        X, Y       =  X.to(device), Y.to(device)
        prediction =  model(X)

        prediction = prediction.cpu()[0]
        Y 		   = Y.cpu()[0]
        
        r2Row   	   = []
        correlationRow = []

        for j in range(len(ecgColumns)):

            yTrue = Y[:, j].numpy()
            yPred = prediction[:, j].numpy()

            r2 = r2_score(yTrue, yPred)

            if np.std(yTrue) == 0 or np.std(yPred) == 0:
                correlation = 0 
            else:
                correlation = pearsonr(yTrue, yPred).statistic

            r2Row.append(r2)
            correlationRow.append(correlation)

        r2Scores.iloc[i]     = r2Row
        correlations.iloc[i] = correlationRow

logger.info(F"Saving the results to {DIST_DIR}")

if not os.path.exists(DIST_DIR):

    logger.warning("The dist folder does not exist. Creating")
    os.makedirs(DIST_DIR, exist_ok = True)

if not os.path.exists(DIST_DIR + "/metrics"):

    logger.warning("The metrics folder does not exist. Creating")
    os.makedirs(DIST_DIR + "/metrics", exist_ok = True)

if not os.path.exists(DIST_DIR + "/exams"):

    logger.warning("The exams folder does not exist. Creating")
    os.makedirs(DIST_DIR + "/exams", exist_ok = True)

for derivation in ecgColumns:

    logger.info(f"Saving the results of {derivation}")

    corrPlotFigure = methodComparativePlot(correlations, derivation, "CORR")
    corrPlotPath   = os.path.join(DIST_DIR, 'metrics', f'CORR - {derivation}.png')
    corrPlotFigure.savefig(corrPlotPath)

    r2PlotFigure = methodComparativePlot(r2Scores, derivation, "R^2")
    r2PlotPath   = os.path.join(DIST_DIR, 'metrics', f'R2 - {derivation}.png')
    r2PlotFigure.savefig(r2PlotPath)

ecgChosen = np.random.choice(datasetLen, 3)

logger.info(F"Saving comparatives plots to {DIST_DIR}")


for ecgId in ecgChosen:

    logger.info(f"Saving the results of {ecgId}")
    sampleX, sampleY = randomLeadsDataset[ecgId]

    with torch.no_grad():
        prediction = model(sampleX.unsqueeze(0).to(device))\
            .squeeze(0)\
            .cpu()\
            .numpy()
        
    sampleECG              = pd.DataFrame(sampleY,    columns = ecgColumns)
    sampleRandomLeadECG    = pd.DataFrame(sampleX,    columns = ecgColumns)
    sampleECGReconstructed = pd.DataFrame(prediction, columns = ecgColumns)

    sampleECGFigure = plotECG(
        sampleECG, 
        ecgColumns, 
        ecgPlotColors
    )
    sampleECGFigure.savefig(f"{DIST_DIR}/exams/ECG - {ecgId}.png")

    sampleRandomLeadECGFigure = plotECG(
        sampleRandomLeadECG, 
        ecgColumns, 
        ecgPlotColors
    )
    sampleRandomLeadECGFigure.savefig(f"{DIST_DIR}/exams/ECG - {ecgId} - Random Lead.png")

    comparativeFullEcgPlotFigure = comparativeFullEcgPlot(
        sampleECG,
        sampleECGReconstructed,
        ecgColumns
    )
    comparativeFullEcgPlotFigure.savefig(f"{DIST_DIR}/exams/ECG - {ecgId} - Comparative.png")
