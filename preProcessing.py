import pandas as pd

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
    if columns_to_dummy is not None:
        df = encode_categorical(df, columns_to_dummy)
    if map_stat_and_tariff:
        #spesifikk for datasett Iranian Churn
        df = map_status_and_tariff(df)
    if clean:
        df = clean_data(df)
    return df


def stat_describe(df, print_out= False):
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