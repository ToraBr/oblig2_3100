import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold


def evaluate_models(df, feature_sets, target="Churn"):
    """
    Kryssevaluerer logistiske regresjonsmodeller med ulike variabelsett ved å dele i k-fold flere ganger,
    med  F1 score som kriterie.
    param : df: pandas dataframe
    param : feature_set: liste av ulike variable for ulike modeller
    param :  target: utfallsvariabel, default: Churn

    """


    y = df[target]

    # deler opp i folds
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10, #total blir det 50 froskjellige folds
        random_state=42 # seed for å få like resultater hver gjennomgang
    )

    results = []

    for features in feature_sets:

        X = df[features]

        #sk.learn sin modell metode, for lett tilgang på F1 score.
        # Bruker penalty None for å få like parametre som med statsmodel
        model = LogisticRegression(penalty=None,max_iter=1000)

        scores = cross_val_score(
            model,
            X,
            y,
            scoring="f1",
            cv=cv
        )

        results.append({
            "variabler": features,
            "Gjennomsnitt_f1": scores.mean()
            #"std_f1": scores.std()
        })

    results_df = pd.DataFrame(results)

    return results_df.sort_values(by="Gjennomsnitt_f1", ascending=False)