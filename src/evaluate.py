import os
import h5py
import logging
import logging.config

import pandas            as pd
import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt

from Dataset          import Code15RandomLeadsSequence
from Model            import ECGRecV2

from dotenv           import load_dotenv

from scipy.stats      import pearsonr
from sklearn.metrics  import r2_score
from sklearn.metrics  import mean_absolute_error

from utils import plotECG
from utils import methodComparativePlot
from utils import comparativeFullEcgPlot

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

def setup_logger():
    logging.config.fileConfig("logging.conf")
    return logging.getLogger()


def load_config():

    load_dotenv()

    return {
        "seed":        int(os.environ["SEED"]),
        "dist_dir":    os.environ["DIST_DIR"],
        "batch_size":  int(os.environ["BATCH_SIZE"]),
        "data_folder": os.environ["DATA_FOLDER"],
    }

def create_dataset(data_folder, seed, batch_size):

    evaluate_file = sorted(
        f for f in os.listdir(data_folder)
        if f.endswith(".hdf5")
    )[-1]

    return Code15RandomLeadsSequence(
        hdf5Files  = [evaluate_file],
        batch_size = batch_size,
        seed       = seed
    )

def load_model(dist_dir):
    model_path = os.path.join(
        dist_dir,
        "best_model.keras"
    )

    model = tf.keras.models.load_model(model_path)

    return model

def create_output_folders(dist_dir):

    metrics_dir = os.path.join(dist_dir, "metrics")
    exams_dir   = os.path.join(dist_dir, "exams")

    os.makedirs(metrics_dir, exist_ok = True)
    os.makedirs(exams_dir, exist_ok = True)

    return metrics_dir, exams_dir

def calculate_metrics(y_true, y_pred):

    r2_row   = []
    mae_row  = []
    corr_row = []

    for k in range(y_true.shape[1]):
        derivation_true = y_true[:, k]
        derivation_pred = y_pred[:, k]

        r2_row.append(
            r2_score(
                derivation_true,
                derivation_pred
            )
        )

        mae_row.append(
            mean_absolute_error(
                derivation_true,
                derivation_pred
            )
        )

        if (
            np.std(derivation_true) == 0 or
            np.std(derivation_pred) == 0
        ):
            corr = 0
        else:
            corr = pearsonr(
                derivation_true,
                derivation_pred
            ).statistic

        corr_row.append(corr)

    return r2_row, mae_row, corr_row

def evaluate_dataset(
    model,
    dataset,
    ecg_columns,
    output_h5_path
):
    dataset_len = len(dataset) * dataset.batch_size
    
    r2_scores = pd.DataFrame(
        np.zeros((dataset_len, len(ecg_columns))), columns=ecg_columns
    )

    mae_scores = pd.DataFrame(
        np.zeros((dataset_len, len(ecg_columns))), columns=ecg_columns
    )

    correlations = pd.DataFrame(
        np.zeros((dataset_len, len(ecg_columns))), columns=ecg_columns
    )

    sample_idx = 0

    with h5py.File(output_h5_path, "w") as h5f:
        
        h5_predictions = None
        h5_targets = None
        h5_exam_ids = h5f.create_dataset(
            "exam_id", shape=(dataset_len,), dtype="i8"
        )
        h5_n_derivations = h5f.create_dataset(
            "n_derivations", shape=(dataset_len,), dtype="i4"
        )

        h5f.attrs["ecg_columns"] = [str(col) for col in ecg_columns]

        exam_id_column       = []
        n_derivations_column = []

        for batch_idx in range(len(dataset)):
            X, Y = dataset[batch_idx]
            
            prediction = model.predict(X, verbose=0)
            
            batch_size = X.shape[0]

            if h5_predictions is None:
                signal_shape = prediction.shape[1:]
                chunk_shape = (dataset.batch_size, *signal_shape)

                h5_predictions = h5f.create_dataset(
                    "predictions",
                    shape=(dataset_len, *signal_shape),
                    dtype="f4",
                    compression="gzip",
                    compression_opts=4,
                    chunks=chunk_shape,
                )

                h5_targets = h5f.create_dataset(
                    "targets",
                    shape=(dataset_len, *signal_shape),
                    dtype="f4",
                    compression="gzip",
                    compression_opts=4,
                    chunks=chunk_shape,
                )

            end_idx = sample_idx + batch_size

            h5_predictions[sample_idx:end_idx] = prediction
            h5_targets[sample_idx:end_idx]     = Y

            for i in range(batch_size):
                global_idx = batch_idx * dataset.batch_size + i
                if global_idx >= len(dataset.indexMap):
                    break
                    
                fileIndex, examIdx = dataset.indexMap[global_idx]
                hdf5File = dataset.hdf5Files[fileIndex]
                hdf5FilePath = os.path.join(os.environ.get("DATA_FOLDER"), hdf5File)

                with h5py.File(hdf5FilePath, "r") as file:
                    exam_id = int(file['exam_id'][examIdx])

                n_derivations = dataset.nLeads  

                exam_id_column.append(exam_id)
                n_derivations_column.append(n_derivations)

                h5_exam_ids[sample_idx + i] = exam_id
                h5_n_derivations[sample_idx + i] = n_derivations

                r2_row, mae_row, corr_row = calculate_metrics(
                    Y[i], prediction[i]
                )

                r2_scores.iloc[sample_idx + i]    = r2_row
                mae_scores.iloc[sample_idx + i]   = mae_row
                correlations.iloc[sample_idx + i] = corr_row

            sample_idx += batch_size

        r2_scores     = r2_scores.iloc[:sample_idx]
        mae_scores   = mae_scores.iloc[:sample_idx]
        correlations = correlations.iloc[:sample_idx]

        r2_scores.insert(0, "exam_id", exam_id_column)
        mae_scores.insert(0, "exam_id", exam_id_column)
        correlations.insert(0, "exam_id", exam_id_column)

        r2_scores.insert(1, "n_derivations", n_derivations_column)
        mae_scores.insert(1, "n_derivations", n_derivations_column)
        correlations.insert(1, "n_derivations", n_derivations_column)

        return r2_scores, mae_scores, correlations

def save_metric_plots(
    metric_df,
    metric_name,
    ecg_columns,
    metrics_dir
):
    for derivation in ecg_columns:
        figure = methodComparativePlot(
            metric_df,
            derivation,
            metric_name
        )

        output = os.path.join(
            metrics_dir,
            f"{metric_name} - {derivation}.png"
        )

        figure.savefig(output)


def save_best_and_worst_ecgs(
    model,
    dataset,
    r2_scores,
    ecg_columns,
    ecg_colors,
    exams_dir,
    top_k = 5
):
    mean_r2 = r2_scores.drop(
        columns=["exam_id", "n_derivations"]
    ).mean(axis=1)

    best_ids  = mean_r2.nlargest(top_k).index.tolist()
    worst_ids = mean_r2.nsmallest(top_k).index.tolist()

    selected_ecgs = [
        ("BEST", ecg_id)
        for ecg_id in best_ids
    ] + [
        ("WORST", ecg_id)
        for ecg_id in worst_ids
    ]

    for category, ecg_id in selected_ecgs:

        batch_idx = ecg_id // dataset.batch_size
        item_idx = ecg_id % dataset.batch_size

        X_batch, Y_batch = dataset[batch_idx]
        sample_x = X_batch[item_idx]
        sample_y = Y_batch[item_idx]

        fileIndex, examIdx = dataset.indexMap[ecg_id]
        hdf5File = dataset.hdf5Files[fileIndex]
        hdf5FilePath = os.path.join(os.environ.get("DATA_FOLDER"), hdf5File)
        with h5py.File(hdf5FilePath, "r") as file:
            exam_id = int(file['exam_id'][examIdx])

        prediction = model.predict(np.expand_dims(sample_x, axis=0), verbose=0)[0]

        sample_ecg = pd.DataFrame(
            sample_y,
            columns = ecg_columns
        )

        random_leads = pd.DataFrame(
            sample_x,
            columns = ecg_columns
        )

        reconstructed = pd.DataFrame(
            prediction,
            columns = ecg_columns
        )

        mean_score = mean_r2.iloc[ecg_id]

        sample_ecg.to_csv(
            f"{exams_dir}/{category} - ECG {exam_id} - Original.csv",
            index = False
        )

        reconstructed.to_csv(
            f"{exams_dir}/{category} - ECG {exam_id} - Reconstructed.csv",
            index = False
        )

        plotECG(
            sample_ecg,
            ecg_columns,
            ecg_colors
        ).savefig(
            f"{exams_dir}/{category} - ECG {ecg_id} - Original.png"
        )

        plotECG(
            random_leads,
            ecg_columns,
            ecg_colors
        ).savefig(
            f"{exams_dir}/{category} - ECG {ecg_id} - Input.png"
        )

        comparativeFullEcgPlot(
            sample_ecg,
            reconstructed,
            ecg_columns
        ).savefig(
            f"{exams_dir}/{category} - ECG {ecg_id} - Comparative - R2={mean_score:.4f}.png"
        )

def main():
    
    logger = setup_logger()
    config = load_config()

    dataset = create_dataset(
        config["data_folder"],
        config["seed"],
        config["batch_size"]
    )

    model = load_model(
        config["dist_dir"]
    )

    metrics_dir, exams_dir = (
        create_output_folders(
            config["dist_dir"]
        )
    )

    r2_scores, mae_scores, correlations = (
        evaluate_dataset(
            model,
            dataset,
            ecgColumns,
            f"{config['dist_dir']}/ecg.reconstructions.h5"
        )
    )

    r2_scores.to_csv(
        f"{config['dist_dir']}/r2.scores.csv", 
        index = False
    )

    mae_scores.to_csv(
        f"{config['dist_dir']}/mae.scores.csv", 
        index = False
    )

    correlations.to_csv(
        f"{config['dist_dir']}/corr.scores.csv", 
        index = False
    )

    save_metric_plots(
        correlations,
        "CORR",
        ecgColumns,
        metrics_dir
    )

    save_metric_plots(
        r2_scores,
        "R2",
        ecgColumns,
        metrics_dir
    )

    save_metric_plots(
        mae_scores,
        "MAE",
        ecgColumns,
        metrics_dir
    )

    save_best_and_worst_ecgs(
        model,
        dataset,
        r2_scores,
        ecgColumns,
        ecgPlotColors,
        exams_dir
    )

if __name__ == "__main__":
    main()