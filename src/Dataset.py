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
            N     = 2, 
            Wn    = 1, 
            btype = 'high', 
            fs    = SAMPLING_FREQUENCY
        )
        ecgFiltred = signal.filtfilt(b, a, ecg, axis = 0)


        start_idx = int(0.5 * SAMPLING_FREQUENCY)
        end_idx   = int(-0.5 * SAMPLING_FREQUENCY)
        
        ecgCroped = ecgFiltred[start_idx:end_idx, :]


        ecgMean = np.mean(ecgCroped, axis = 0, keepdims = True)
        ecgStd  = np.std(ecgCroped,  axis = 0, keepdims = True) + 1e-8

        ecgNormalized = (ecgCroped - ecgMean) / ecgStd


        target_samples = 4096
        current_samples = ecgNormalized.shape[0]

        pad_amount = target_samples - current_samples
        pad_before = pad_amount // 2
        pad_after  = pad_amount - pad_before
        
        ecgZeroPadding = np.pad(
            ecgNormalized, 
            pad_width       = ((pad_before, pad_after), (0, 0)), 
            mode            = 'constant', 
            constant_values = 0
        )

        return ecgZeroPadding

    def __len__(self):
        return len(self.indexMap)

    def __getitem__(self, idx):
        
        fileIndex, examIdx = self.indexMap[idx]
        hdf5File           = self.hdf5Files[fileIndex]

        nLeadsToPick = np.random.randint(3, 9 + 1)

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
            exam_id = int(file['exam_id'][examIdx])

        tracing = self.transform(tracing)

        X = np.zeros_like(tracing)
        X[:, chosenLeads] = tracing[:, chosenLeads]

        X = X[:, self.features]
        X = torch.tensor(X, dtype = torch.float32)

        Y = tracing[:, self.target]
        Y = torch.tensor(Y, dtype = torch.float32)

        return X, Y, exam_id, nLeadsToPick
