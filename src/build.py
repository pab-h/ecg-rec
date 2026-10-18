import os
import logging
import logging.config

import tensorflow        as tf
import matplotlib.pyplot as plt

from dotenv  import load_dotenv
from Dataset import Code15RandomLeadsSequence
from Model   import ECGRecV2
from utils   import R2ScoreMetric

from sklearn.metrics import r2_score

def setup_logger():

    os.makedirs("logs", exist_ok=True)
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

def create_sequences(data_folder, batch_size, seed):

    train_test_files = os.listdir(data_folder)
    train_test_files = train_test_files[:-1]

    split_idx   = int(0.8 * len(train_test_files))
    train_files = train_test_files[:split_idx]
    val_files   = train_test_files[split_idx:]

    train_seq = Code15RandomLeadsSequence(
        hdf5Files  = train_files,
        batch_size = batch_size,
        seed       = seed
    )

    val_seq = Code15RandomLeadsSequence(
        hdf5Files  = val_files,
        batch_size = batch_size,
        seed       = seed
    )

    return train_seq, val_seq

def create_model():

    model = ECGRecV2(latentDim=128)

    return model

def save_training_plot(history, dist_dir):

    os.makedirs(dist_dir, exist_ok=True)

    losses     = history.history['loss']
    val_losses = history.history['val_loss']
    r2_scores  = history.history['r2_score']
    val_r2     = history.history['val_r2_score']
    epochs     = range(len(losses))

    fig, axes = plt.subplots(
        nrows   = 2,
        ncols   = 1,
        figsize = (10, 8),
        sharex  = True
    )

    axes[0].plot(epochs, losses, label='Train Loss', marker="x")
    axes[0].plot(epochs, val_losses, label='Val Loss', marker="o")
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(epochs, r2_scores, label='Train R²', marker="x")
    axes[1].plot(epochs, val_r2, label='Val R²', marker="o")
    axes[1].set_title("Training and Validation R²")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("R²")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()

    output = os.path.join(dist_dir, "training.png")
    plt.savefig(output)
    plt.close()

    return output

def main():

    logger = setup_logger()
    config = load_config()

    train_seq, val_seq = create_sequences(
        config["data_folder"],
        config["batch_size"],
        config["seed"]
    )

    model = create_model()

    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss      = tf.keras.losses.MeanSquaredError(),
        metrics   = [R2ScoreMetric()]
    )

    os.makedirs(config["dist_dir"], exist_ok = True)
    checkpoint_path = os.path.join(config["dist_dir"], "best_model.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor              = 'val_r2_score',
            mode                 = 'max',
            patience             = 10,
            min_delta            = 1e-3,
            restore_best_weights = True,
            verbose              = 1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = checkpoint_path,
            monitor        = 'val_r2_score',
            mode           = 'max',
            save_best_only = True,
            verbose        = 1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join("logs", "training_history.csv")
        )
    ]

    history = model.fit(
        train_seq,
        validation_data = val_seq,
        epochs          = config["epochs"],
        callbacks       = callbacks,
        verbose         = 1
    )

    plot_path = save_training_plot(
        history,
        config["dist_dir"]
    )

    logger.info(f"Training plot saved at {plot_path}")

if __name__ == "__main__":
    main()