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

def create_dataset(data_folder, seed):

    evaluate_file = os.listdir(data_folder)[-1]

    return Code15RandomLeadsDataset(
        hdf5Files = [evaluate_file],
        seed      = seed
    )


def create_dataloader(dataset, batch_size):
    return DataLoader(
        dataset    = dataset,
        batch_size = batch_size,
        shuffle    = False
    )

def load_model(dist_dir, device):

    model = ECGReconstructor(
        latentDim = 128,
        hiddenDim = 32
    )

    model = torch.compile(model)

    model_path = os.path.join(
        dist_dir,
        "best_model.pth"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model.to(device)

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
    dataloader,
    dataset_len,
    ecg_columns,
    device
):

    r2_scores = pd.DataFrame(
        np.zeros((dataset_len, len(ecg_columns))),
        columns = ecg_columns
    )

    mae_scores = pd.DataFrame(
        np.zeros((dataset_len, len(ecg_columns))),
        columns = ecg_columns
    )

    correlations = pd.DataFrame(
        np.zeros((dataset_len, len(ecg_columns))),
        columns = ecg_columns
    )

    sample_idx = 0

    model.eval()

    with torch.no_grad():

        for X, Y in dataloader:

            X = X.to(device)

            prediction = model(X).cpu()
            Y          = Y.cpu()

            for i in range(X.size(0)):

                r2_row, mae_row, corr_row = (
                    calculate_metrics(
                        Y[i].numpy(),
                        prediction[i].numpy()
                    )
                )

                r2_scores.iloc[sample_idx]    = r2_row
                mae_scores.iloc[sample_idx]   = mae_row
                correlations.iloc[sample_idx] = corr_row

                sample_idx += 1

    return (
        r2_scores,
        mae_scores,
        correlations
    )

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

def save_violin_plots(
    metrics,
    ecg_columns,
    metrics_dir
):

    for metric_name, metric_df in metrics.items():

        plt.figure(figsize=(12, 6))

        data = [
            metric_df[col].dropna()
            for col in ecg_columns
        ]

        plt.violinplot(
            data,
            showmeans   = True,
            showmedians = True,
            showextrema = True
        )

        plt.xticks(
            range(1, len(ecg_columns) + 1),
            ecg_columns,
            rotation = 45
        )

        plt.title(
            f"{metric_name} - Violin Plot por Derivação"
        )

        plt.tight_layout()

        output = os.path.join(
            metrics_dir,
            f"{metric_name} - Violinplot.png"
        )

        plt.savefig(output)
        plt.close()

def save_best_and_worst_ecgs(
    model,
    dataset,
    r2_scores,
    device,
    ecg_columns,
    ecg_colors,
    exams_dir,
    top_k = 5
):

    mean_r2   = r2_scores.mean(axis=1)

    best_ids  = mean_r2.nlargest(top_k).index.tolist()
    worst_ids = mean_r2.nsmallest(top_k).index.tolist()

    selected_ecgs = [
        ("BEST", ecg_id)
        for ecg_id in best_ids
    ] + [
        ("WORST", ecg_id)
        for ecg_id in worst_ids
    ]

    model.eval()

    for category, ecg_id in selected_ecgs:

        sample_x, sample_y = dataset[ecg_id]

        with torch.no_grad():

            prediction = (
                model(
                    sample_x.unsqueeze(0)
                    .to(device)
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

        sample_ecg = pd.DataFrame(
            sample_y,
            columns=ecg_columns
        )

        random_leads = pd.DataFrame(
            sample_x,
            columns=ecg_columns
        )

        reconstructed = pd.DataFrame(
            prediction,
            columns=ecg_columns
        )

        mean_score = mean_r2.iloc[ecg_id]

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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = create_dataset(
        config["data_folder"],
        config["seed"]
    )

    dataloader = create_dataloader(
        dataset,
        config["batch_size"]
    )

    model = load_model(
        config["dist_dir"],
        device
    )

    metrics_dir, exams_dir = (
        create_output_folders(
            config["dist_dir"]
        )
    )

    r2_scores, mae_scores, correlations = (
        evaluate_dataset(
            model,
            dataloader,
            len(dataset),
            ecgColumns,
            device
        )
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

    save_violin_plots(
        {
            "MAE": mae_scores,
            "R2": r2_scores,
            "CORR": correlations
        },
        ecgColumns,
        metrics_dir
    )

    save_best_and_worst_ecgs(
        model,
        dataset,
        r2_scores,
        device,
        ecgColumns,
        ecgPlotColors,
        exams_dir
    )

if __name__ == "__main__":
    main()