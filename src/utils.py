import matplotlib.pyplot as plt
import numpy             as np

from sklearn.metrics import r2_score

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
