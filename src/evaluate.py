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
from sklearn.metrics  import mean_absolute_error
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

print(BATCH_SIZE)

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
    "aVR",
    "V1",
    "V4",
    "LII",
    "aVL",
    "V2",
    "V5",
    "LIII",
    "aVF",
    "V3",
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

maeScores = pd.DataFrame(
    columns = ecgColumns,
    index   = range(datasetLen),
    data    = np.zeros((datasetLen, len(ecgColumns)))
)

model.eval()

sampleIdx = 0 

with torch.no_grad():
    for i, (X, Y) in enumerate(dataloader):

        X,          Y = X.to(device),   Y.to(device)
        prediction, Y = model(X).cpu(), Y.cpu()

        for j in range(X.size(0)): 

            yTrue = Y[j].numpy()
            yPred = prediction[j].numpy()

            r2Row   = []
            maeRow  = []
            corrRow = []

            for k in range(len(ecgColumns)):

                derivationTrue = yTrue[:, k]
                derivationPred = yPred[:, k]

                r2 = r2_score(derivationTrue, derivationPred)

                mae = mean_absolute_error(derivationTrue, derivationPred)

                if np.std(derivationTrue) == 0 or np.std(derivationPred) == 0:
                    correlation = 0
                else:
                    correlation = pearsonr(derivationTrue, derivationPred).statistic

                r2Row.append(r2)
                maeRow.append(mae)
                corrRow.append(correlation)

            r2Scores.iloc[sampleIdx]     = r2Row
            maeScores.iloc[sampleIdx]    = maeRow
            correlations.iloc[sampleIdx] = corrRow

            sampleIdx += 1

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

    maePlotFigure = methodComparativePlot(maeScores, derivation, "MAE")
    maePlotPath   = os.path.join(DIST_DIR, 'metrics', f'MAE - {derivation}.png')
    maePlotFigure.savefig(maePlotPath)

logger.info("Saving boxplots for each metric")

logger.info("Saving violin plots for each metric")

metrics = {
    "MAE": maeScores,
    "R2": r2Scores,
    "CORR": correlations
}

for metricName, metricDF in metrics.items():

    plt.figure(figsize=(12, 6))

    data = [metricDF[col].dropna() for col in ecgColumns]

    parts = plt.violinplot(
        data,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )

    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    if "cmedians" in parts:
        parts['cmedians'].set_color('red')
    if "cmeans" in parts:
        parts['cmeans'].set_color('green')

    plt.xticks(
        ticks=range(1, len(ecgColumns) + 1),
        labels=ecgColumns,
        rotation=45
    )

    plt.title(f'{metricName} - Violin Plot por Derivação')
    plt.ylabel(metricName)
    plt.xlabel("Derivações")

    violinPath = os.path.join(DIST_DIR, 'metrics', f"{metricName} - Violinplot.png")
    plt.tight_layout()
    plt.savefig(violinPath)
    plt.close()


ecgChosen = np.random.choice(datasetLen, 5)

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
