import statsmodels.api as sm


def simple_logit_model(df, x_col, y_col):
    """
    Lager en enkel logistisk regresjonsmodell for én forklaringsvariabel.

    Args:
        df (pd.DataFrame)
        x_col (str): utfallsvariabel
        y (str): utfallsvariabel

    Returns:
        result (LogitResults): model.fit
    """
    X = sm.add_constant(df[x_col])
    y = df[y_col]
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result


def simple_logit_models(df, x_cols, y_col, print_out = False):
    """
    metode for  enkel logistisk regresjonmodell for hver x kolonne mot y.
    args: df(pd.DataFrame),
    x_cols(list of strings): forklaringsvariable,
    y(string): utfallsvariabel

    returns: results(dictionary): Kolonnenavn(forklaringsvariabel) som nøkkel og model.fit som verdi
    """
    results = {}
    for col in x_cols:
        result = simple_logit_model(df, col, y_col)
        results[col] = result

        if print_out:
            print(f"Resultat {col}:")
            print(result.summary())
    return results


def model_custom_summary(results, print_out=False):
    """
    Custom sammendrag med de verdier spesielt nyttige for Oblig 2
    Args:
        results (dict): Nøkkel = variabel, verdi = fitted Logit-resultat
        print_out (bool): print summary
    Returns:
        list of lists: [Variable, Intercept, Coef, Std Err, P-val, CI Low, CI High, Pseudo-R2]
    """
    data = []
    for v, res in results.items():
        beta0 = res.params['const']
        beta1 = res.params[v]
        std_b = res.bse[v]
        pval = res.pvalues[v]
        ci_low, ci_high = res.conf_int().loc[v]
        pr2 = res.prsquared  # pseudo-R²
        aic = res.aic

        data.append([
            v,
            beta0,
            beta1,
            std_b,
            pval,
            ci_low,
            ci_high,
            pr2,
            aic

        ])

    if print_out:
        for row in data:
            rounded_row = [row[0]] + [round(x, 4) for x in row[1:]]
            print(rounded_row)

    return data


def find_lowest_aic_forw_select (df, x_columns, selected , y_column ) :
    results = {}
    for x in x_columns:
        x_vars = selected + [x]
        X = sm.add_constant(df[x_vars])
        y = df[y_column]
        model = sm.Logit(y, X).fit(disp = False)
        results [x] = model.aic
    return min(results, key = results.get), min(results.values () )

def logit_reg_forw(df, x_columns, y_column ):
    results = {}
    remaining_vars =list(x_columns) # ny liste som jeg kanendre
    selected = [] # tom liste jeg kan legge til variabler
    var_to_add, last_aic = find_lowest_aic_forw_select(df, remaining_vars, selected, y_column)
    print(f"{var_to_add}: {last_aic:.3f}")
    selected.append(var_to_add)
    results[var_to_add] = last_aic
    remaining_vars.remove(var_to_add)

    while True :
        # stopper når det ikke er forbedring i aic verdi
        var_to_add , aic = find_lowest_aic_forw_select (df, remaining_vars, selected, y_column)
        if aic < last_aic:
            print(f"{var_to_add}: {aic:.3f}")
            selected.append ( var_to_add )
            remaining_vars.remove(var_to_add)
            results [var_to_add] = aic
            last_aic = aic
        else:
            break
    return results