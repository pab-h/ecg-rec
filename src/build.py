import os
import logging
import logging.config

import torch
import torch.nn          as nn
import matplotlib.pyplot as plt

from dotenv           import load_dotenv
from sklearn.metrics  import r2_score
from torch.utils.data import DataLoader, random_split

from Dataset import Code15RandomLeadsDataset
from Model   import ECGReconstructor

def setup_logger():

    logging.config.fileConfig("logging.conf")

    return logging.getLogger()

def load_config():

    load_dotenv()

    return {
        "seed":               int(os.environ["SEED"]),
        "epochs":             int(os.environ["EPOCHS"]),
        "batch_size":         int(os.environ["BATCH_SIZE"]),
        "data_folder":        os.environ["DATA_FOLDER"],
        "dist_dir":           os.environ["DIST_DIR"],
        "sampling_frequency": int(os.environ["SAMPLING_FREQUENCY"]),
    }

def create_dataset(data_folder, seed):

    dataset = Code15RandomLeadsDataset(
        hdf5Files=os.listdir(data_folder),
        seed=seed
    )

    return dataset


def create_dataloaders(dataset, batch_size, seed):

    generator = torch.Generator().manual_seed(seed)

    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size

    train_set, test_set = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

def create_model(device):

    model = ECGReconstructor(
        latentDim=128,
        hiddenDim=32
    )

    model = torch.compile(model)

    return model.to(device)

def compute_r2(y_true, y_pred):

    y_true = y_true.detach().cpu().flatten(0, 1).numpy()
    y_pred = y_pred.detach().cpu().flatten(0, 1).numpy()

    return r2_score(y_true, y_pred)

def train_epoch(model, dataloader, optimizer, criterion, device):

    model.train()

    epoch_loss = 0
    epoch_r2   = 0

    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)

        prediction = model(X)

        loss = criterion(prediction, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_r2   += compute_r2(Y, prediction)

    epoch_loss /= len(dataloader)
    epoch_r2   /= len(dataloader)

    return epoch_loss, epoch_r2

def train(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epochs,
    logger
):
    
    losses    = []
    r2_scores = []

    for epoch in range(epochs):

        loss, r2 = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        losses.append(loss)
        r2_scores.append(r2)

        logger.info(
            f"Train - epoch={epoch} loss={loss:.5f} r2={r2:.5f}"
        )

    return losses, r2_scores

def evaluate(model, dataloader, criterion, device):

    model.eval()

    loss = 0
    r2   = 0

    with torch.no_grad():
        for X, Y in dataloader:

            X, Y = X.to(device), Y.to(device)

            prediction = model(X)

            batch_loss = criterion(prediction, Y)

            loss += batch_loss.item()
            r2   += compute_r2(Y, prediction)

    loss /= len(dataloader)
    r2   /= len(dataloader)

    return loss, r2

def save_training_plot(losses, r2_scores, epochs, dist_dir):
    os.makedirs(dist_dir, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 8),
        sharex=True
    )

    axes[0].scatter(range(epochs), losses, marker="x")
    axes[0].set_title("Training Loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid()

    axes[1].scatter(range(epochs), r2_scores, marker="x")
    axes[1].set_title("Training R²")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("R²")
    axes[1].grid()

    plt.tight_layout()

    output = os.path.join(dist_dir, "training.png")
    plt.savefig(output)

    return output

def main():

    logger = setup_logger()
    config = load_config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = create_dataset(
        config["data_folder"],
        config["seed"]
    )

    train_loader, test_loader = create_dataloaders(
        dataset,
        config["batch_size"],
        config["seed"]
    )

    model = create_model(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    losses, r2_scores = train(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        config["epochs"],
        logger
    )

    plot_path = save_training_plot(
        losses,
        r2_scores,
        config["epochs"],
        config["dist_dir"]
    )

    logger.info(f"Training plot saved at {plot_path}")

    test_loss, test_r2 = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    logger.info(
        f"Validation - loss={test_loss:.5f} r2={test_r2:.5f}"
    )

    model_path = os.path.join(
        config["dist_dir"],
        "model.pth"
    )

    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()