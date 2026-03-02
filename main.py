from preProcessing import *

df = load_n_clean_data('data/Customer Churn.csv', columns_to_remove=['Age'])
stat_describe(df, print_out= True)
