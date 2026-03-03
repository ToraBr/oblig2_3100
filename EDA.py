import pandas as pd
import matplotlib.pyplot as plt

def stat_describe(df, print_out= False):
    """
    genererer beskrivende statestik (mean, median, std, min, and max) på et Dataframe)
    Args:
        df (pd.DataFrame): input dataframe av datasett.
        print(boolean, optional): hvis True, printer resultatet, default False.

        returns: pd.Dataframe: et dataframe hvor hver ras represeterer variabler,
        og hver kolonne inneholder mean, median, std, min og max


        """

    pd.set_option('display.max_rows', None)  # viser alle rader
    pd.set_option('display.max_columns', None)  # viser alle kolonner

    stats = pd.DataFrame()
    stats['mean'] = df.mean()
    stats['median'] = df.median()
    stats['std'] = df.std()
    stats['min'] = df.min()
    stats['max'] = df.max()


    if print_out:
        print(stats)
    return stats


def plot_histograms(df, columns, bins=20, figsize=(8,5), savefig = False):
    """
    Lager histogrammer for kolonne(r) i et datasett.

    Args:
        df (pd.DataFrame): Dataframe som inneholder kolonnene.
        columns (list of str): Liste over kolonnenavn som skal plottes.
        bins (int, optional): Antall bins i histogrammet. Default 20.
        figsize (tuple, optional): Figurstørrelse for hver plot. Default (8,5).
        savefig(boolean, optional): Hvis True lagres figur, default False.
    Returns:
        None (viser plottene)
    """
    for col in columns:
        plt.figure(figsize=figsize)
        plt.hist(df[col], bins=bins, color='skyblue', edgecolor='black')
        plt.title(f'{col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(axis='y', alpha=0.75)
        if savefig: plt.savefig(f'{col}.png', bbox_inches='tight')
        plt.show()


def plot_scatter(df, x_cols, y_col, figsize=(6,5), savefig=False):
    """
    Genererer scatterplot(s) med spesifiserte x- verdier og y verdi. .

    Args:
        df (pd.DataFrame): Dataframe som innehodler dataene.
        x_cols (list of str): Liste av kolonnenavn for x variabler.
        y_col (str): et kolonnenavn, utfallsvariabel, y variabel.
        figsize (tuple, optional): Figure str. Default (6,5).
        savefig (bool, optional): hvis True, lagrer figur, default False..

    Returns:
        None
    """

    for x in x_cols:
            plt.figure(figsize=figsize)
            plt.scatter(df[x], df[y_col], alpha=0.6)
            plt.xlabel(x)
            plt.ylabel(y_col)
            plt.title(f"{y_col} vs {x}")
            plt.grid(True)
            if savefig:
                plt.savefig(f'{x}_scatter.png', bbox_inches='tight')
            plt.show()