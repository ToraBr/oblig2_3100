from preProcessing import load_n_clean_data
from EDA import *
from model import *
df = load_n_clean_data('data/Customer Churn.csv', columns_to_remove=['Age'])
#stat_describe(df, print_out= True)
numeric_var = ['Call  Failure', 'Subscription  Length', 'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers', 'Customer Value' ]
#plot_histograms(df, numeric_var, savefig= True)
#plot_histograms(df, ['Age Group'], bins=5)
#plot_histograms(df, ['Charge  Amount'], bins=10, savefig= True)
#plot_scatter(df, numeric_var, 'Churn')
#plot_scatter(df, ['Age Group'], 'Churn')
#plot_scatter(df, ['Charge  Amount'], 'Churn')
x_cols = df.columns.drop('Churn')
models = simple_logit_models(df, x_cols, 'Churn')

#model_custom_summary(models, print_out = True)

#frequency of use og ocustomer value er kontinuerlige variabler med høyste pseudo R^2
#scatter_XO(df, 'Customer Value', 'Frequency of use', 'Churn')
#scatter_XO(df, 'Seconds of Use', 'Customer Value', 'Churn')
#scatter_XO(df, 'Seconds of Use', 'Frequency of use', 'Churn')
logit_reg_forw(df, x_cols, ['Churn'])