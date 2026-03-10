import pandas as pd
from itertools import combinations

def encode_categorical(df, column_names= None):
    #metode for dummyvariabler
    df = pd.get_dummies(df, columns=column_names, drop_first=True)
    # Konverter alle boolske kolonner til int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df



def map_status_and_tariff(df):
    """
    Spesifikk for datasett Irarian churn
    Konverterer Status og Tariff Plan fra 1/2 til 0/1.

    Status: 1 = active -> 1, 2 = non-active -> 0
    Tariff Plan: 1 = Pay as you go -> 0, 2 = contractual -> 1
    """
    if 'Status' in df.columns:
        df['Status'] = df['Status'].map({1: 1, 2: 0})
    if 'Tariff Plan' in df.columns:
        df['Tariff Plan'] = df['Tariff Plan'].map({1: 0, 2: 1})
    return df

def clean_data(df):
    #dropper nan verdier
    df = df.dropna()
    return df
def load_data(path):
    #leser datafila
    return pd.read_csv(path)

def drop_unused_columns(df, column_names = None):
    #fjerner user_id
    df = df.drop(columns= column_names)
    return df

def load_n_clean_data(path, clean=True, columns_to_remove = None, columns_to_dummy= None, map_stat_and_tariff = True):
    #samlet funskjon, laster inn renser, fjerner og omkoder variabler.
    df = load_data(path)
    df = drop_unused_columns(df, columns_to_remove)
    df.columns = df.columns.str.strip()
    if columns_to_dummy is not None:
        df = encode_categorical(df, columns_to_dummy)
    if map_stat_and_tariff:
        #spesifikk for datasett Iranian Churn
        df = map_status_and_tariff(df)
    if clean:
        df = clean_data(df)
    return df


def variable_combinations(x_cols, max_vars=None):
    """
    Returnerer alle kombinasjoner av x_cols som en liste av lister.
    Hver kombinasjon er en liste av kolonnenavn.
    max_vars: maks antall variabler i en kombinasjon (default: alle)
    """
    all_combinations = []

    if max_vars is None:
        max_vars = len(x_cols)

    for r in range(1, max_vars + 1):
        for comb in combinations(x_cols, r):
            all_combinations.append(list(comb))  # konverter tuple til liste

    return all_combinations