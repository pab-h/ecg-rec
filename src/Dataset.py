import os
import h5py
import torch

import numpy  	    as np
import scipy.signal as signal

from torch.utils.data import Dataset
from dotenv           import load_dotenv

# Loading env

load_dotenv()

DATA_FOLDER        = os.environ.get("DATA_FOLDER")
SAMPLING_FREQUENCY = int(os.environ.get("SAMPLING_FREQUENCY"))

class Code15RandomLeadsDataset(Dataset):

    def __init__(self, hdf5Files, seed = None):
        super().__init__()

        self.hdf5Files = hdf5Files
        self.indexMap  = []  
        self.features  = np.arange(0, 12)
        self.target    = np.arange(0, 12)

        self.nLeads      = 12  
        self.leadCounts  = np.zeros(self.nLeads)

        for fileIndex, path in enumerate(self.hdf5Files):

            dataFile = os.path.join(DATA_FOLDER, f"{path}")

            with h5py.File(dataFile, "r") as file:

                samplesCount = file['exam_id'].shape[0]
                self.indexMap.extend([(fileIndex, i) for i in range(samplesCount)])

        if seed:
            np.random.seed(seed)

    def transform(self, ecg):
        b, a = signal.butter(
            N     = 1, 
            Wn    = 1, 
            btype = 'high', 
            fs    = SAMPLING_FREQUENCY
        )
        
        ecgFiltred  = signal.filtfilt(b, a, ecg, axis = 0)
        ecgWithGain = 5 * ecgFiltred
        ecgClean    = ecgWithGain[600: -600, :]

        ecgMean = np.mean(ecgClean, axis = 0, keepdims = True)
        ecgStd  = np.std(ecgClean,  axis = 0, keepdims = True) + 1e-8

        ecgNormalized = (ecgClean - ecgMean) / ecgStd

        return ecgNormalized

    def __len__(self):
        return len(self.indexMap)

    def __getitem__(self, idx):
        
        fileIndex, examIdx = self.indexMap[idx]
        hdf5File           = self.hdf5Files[fileIndex]

        nLeadsToPick = np.random.randint(3, 9)

        odds =  1 / (self.leadCounts + 1)
        odds /= odds.sum()

        chosenLeads = np.random.choice(
            a       = self.nLeads,
            size    = nLeadsToPick,
            replace = False,
            p       = odds
        )

        self.leadCounts[chosenLeads] += 1

        hdf5File = os.path.join(DATA_FOLDER, hdf5File)

        with h5py.File(hdf5File, "r") as file:
            tracing = np.array(file['tracings'][examIdx]) 

        tracing = self.transform(tracing)

        X = np.zeros_like(tracing)
        X[:, chosenLeads] = tracing[:, chosenLeads]

        X = X[:, self.features]
        X = torch.tensor(X, dtype = torch.float32)

        Y = tracing[:, self.target]
        Y = torch.tensor(Y, dtype = torch.float32)

        return X, Y
