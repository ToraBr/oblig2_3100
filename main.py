from preProcessing import load_n_clean_data, variable_combinations
from EDA import *
from model import *
from evaluation import *
import pandas as pd

pd.set_option('display.max_colwidth', None)
df = load_n_clean_data('data/Customer Churn.csv', columns_to_remove=['Age'])

stat_describe(df, print_out= True)
numeric_var = ['Call  Failure', 'Subscription  Length', 'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers', 'Customer Value' ]
plot_histograms(df, numeric_var)
plot_histograms(df, ['Age Group'], bins=5)
plot_histograms(df, ['Charge  Amount'], bins=10)
plot_scatter(df, numeric_var, 'Churn')
plot_scatter(df, ['Age Group'], 'Churn')#plot_scatter(df, ['Charge  Amount'], 'Churn')

x_cols = df.columns.drop('Churn')
models = simple_logit_models(df, x_cols, 'Churn')
model_custom_summary(models, print_out = True)

scatter_XO(df, 'Customer Value', 'Frequency of use', 'Churn')
scatter_XO(df, 'Seconds of Use', 'Customer Value', 'Churn')
scatter_XO(df, 'Seconds of Use', 'Frequency of use', 'Churn')
model_list = logit_reg_forw(df, x_cols, ['Churn'])
model = list(model_list.values())[-1]

residuals(model)

x_vars1 = ['Status', 'Complains', 'Customer Value', 'Frequency of use','Call  Failure', 'Frequency of SMS']
feature_sets_1 = variable_combinations(x_vars1)
results_df_1 = evaluate_models(df, feature_sets_1)  # tren og evaluer modellene
print(results_df_1)

cv_best_vars_1 = ['Status', 'Complains', 'Customer Value', 'Frequency of use', 'Frequency of SMS']
result_cv_1 = simple_logit_model(df,cv_best_vars_1,'Churn')
print(result_cv_1.summary())
print(f'Aic: {result_cv_1.aic:.3f}')
print(f'devians:  {(2 * result_cv_1.llf):.3f}')

x_vars_2 = ['Status', 'Complains', 'Customer Value', 'Frequency of use', 'Frequency of SMS','Call  Failure','Charge  Amount', 'Distinct Called Numbers' ]
feature_sets_2 = variable_combinations(x_vars_2)

results_df_2 = evaluate_models(df, feature_sets_2)
print(results_df_2)

cv_best_vars_2 = ['Status', 'Complains', 'Customer Value', 'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers']
result_cv_2 = simple_logit_model(df,cv_best_vars_2,'Churn', print_out=True)
print(f'Aic: {result_cv_2.aic:.3f}')
print(f'devians:  {(2 * result_cv_2.llf):.3f}')

