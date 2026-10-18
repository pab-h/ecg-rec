import matplotlib.pyplot as plt
import numpy             as np
import tensorflow        as tf

from keras.metrics   import Metric
from sklearn.metrics import r2_score

import tensorflow as tf

class R2ScoreMetric(tf.keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        
        super().__init__(name=name, **kwargs)

        self.ss_res   = self.add_weight(shape=(12,), initializer='zeros', name='ss_res')
        self.sum_y    = self.add_weight(shape=(12,), initializer='zeros', name='sum_y')
        self.sum_y_sq = self.add_weight(shape=(12,), initializer='zeros', name='sum_y_sq')
        self.count    = self.add_weight(shape=(), initializer='zeros', name='count')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        input_shape = tf.shape(y_true)
        y_true_flat = tf.reshape(y_true, [-1, input_shape[-1]])
        y_pred_flat = tf.reshape(y_pred, [-1, input_shape[-1]])

        batch_ss_res = tf.reduce_sum(tf.square(y_true_flat - y_pred_flat), axis=0)
        self.ss_res.assign_add(batch_ss_res)

        self.sum_y.assign_add(tf.reduce_sum(y_true_flat, axis=0))
        self.sum_y_sq.assign_add(tf.reduce_sum(tf.square(y_true_flat), axis=0))

        n_samples = tf.cast(tf.shape(y_true_flat)[0], tf.float32)
        self.count.assign_add(n_samples)

    def result(self):

        mean_y = self.sum_y / tf.maximum(self.count, 1.0)
        
        ss_tot = self.sum_y_sq - (self.sum_y * mean_y)
        
        ss_tot = tf.where(ss_tot == 0.0, 1e-8, ss_tot)

        r2_per_column = 1.0 - (self.ss_res / ss_tot)
        
        return tf.reduce_mean(r2_per_column)

    def reset_state(self):
        self.ss_res.assign(tf.zeros(12))
        self.sum_y.assign(tf.zeros(12))
        self.sum_y_sq.assign(tf.zeros(12))
        self.count.assign(0.0)
        
def comparativeFullEcgPlot(ecgOring, ecgRec, headers):

    figure, axes = plt.subplots(
        nrows   = 3,
        ncols   = 4,
        sharex  = True,
        figsize = (16, 9)
    )

    figure.suptitle("Comparison: ECG 12-Lead")
    figure.supxlabel("Sample")
    figure.supylabel("Dpp")

    axes = axes.flatten()

    for idx, header in enumerate(headers):
        corr = np.round(ecgOring[header].corr(ecgRec[header]), 3)
        r2   = np.round(r2_score(ecgOring[header], ecgRec[header]), 3)

        axes[idx].plot(
            ecgOring[header], 
            color = "blue", 
            alpha = 0.75
        )
        axes[idx].plot(
            ecgRec[header], 
            color = "red", 
            alpha = 0.75
        )

        axes[idx].set_title(f"{header} CORR = {corr} r2 = {r2}")
    

    plt.tight_layout(pad = 1.5)
    plt.close()

    return figure

def plotECG(ecg, headers, colors): 

    figure, axes = plt.subplots(
        nrows   = 3,
        ncols   = 4,
        sharex  = True,
        figsize = (16, 9)
    )

    figure.suptitle("ECG 12-Lead")
    figure.supxlabel("Sample")
    figure.supylabel("Dpp")

    axes = axes.flatten()

    for idx, header in enumerate(headers):
        axes[idx].plot(ecg[header], color = colors[header])
        axes[idx].set_title(f"{header}")
    

    plt.tight_layout(pad = 1.5)

    plt.close()

    return figure

def methodComparativePlot(df, derivation, method):

    dfMean = np.mean(df[derivation])
    dfMean = np.round(dfMean, 3)

    figure, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))

    axes[0].set_title(f"{method}($ {derivation} $, $ {derivation}_{{rec}} $)")
    axes[1].set_title(f"Histograma - {method}($ {derivation} $, $ {derivation}_{{rec}} $)")

    axes[0].set_xlabel("n")
    axes[0].set_ylabel(f"{method}")
    
    axes[1].set_xlabel(f"{method}")
    axes[1].set_ylabel("Frequência")


    axes[0].scatter(
        df.index, 
        df[derivation]
    )
    axes[0].axhline(
        dfMean, 
        color     = 'r', 
        linestyle = '--', 
        label     = f"Média = {dfMean}"
    )


    counts, bins = np.histogram(df[derivation], 50)
    axes[1].stairs(counts / len(df[derivation]), bins, fill = True)
    axes[1].axvline(
        dfMean, 
        color     = 'r', 
        linestyle = '--', 
        label     = f"Média = {dfMean}"
    )

    axes[1].legend()
    axes[0].legend()

    plt.close()

    return figure
