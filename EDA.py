import pandas as pd
import numpy as np
from data_read import extract_df
from preprocess import del_blank_value_cols, process
import matplotlib.pyplot as plt; plt.rcdefaults()
from supported_data import DType, numeric_features
import seaborn as sns
%matplotlib inline



data = extract_df()
#data['id'] = range(data.shape[0]+1)[1:]
#test = extract_df18Q1()

# Response of data

data['loan_status'].value_counts()


data['loan_status'].value_counts()
#Fully Paid     34116
#Charged Off     5670


###############

data = del_blank_value_cols(data)
# (42535, 55)

###############

data['int_rate'] = data['int_rate'].str[:-1].astype(float)

grouped = data.groupby('grade')['int_rate'].mean().reset_index()

plt.bar(grouped['grade'], grouped['int_rate'], align='center', alpha=0.5, color='g')

##


## removing desc because it's having so many na values in the data
description = data['desc']; del data['desc']
# can be used as a NLP task

# dropping na rows since the ratio is only 90%
data = data.dropna()

###### checking for correlation
data_num=data[numeric_features].copy()

## Remove zero variance features
data_var = data.var()
for x in list(data_var[data_var <= 0.0].index):
    print(x)
    del data_num[x]

#These features are removed from zero variance filter
#out_prncp
#out_prncp_inv
#policy_code


data_cor = data_num.corr()
# plot the heatmap
sns.heatmap(data_cor, 
        xticklabels=data_cor.columns,
        yticklabels=data_cor.columns)

data_num = remove_correlation(data_num, 0.8)
    
# these features were removed for high correlated features
#funded_amnt
#funded_amnt_inv
#installment
#annual_inc
#total_pymnt
#total_pymnt_inv
#total_rec_prncp
#collection_recovery_fee

data_num_cor = data_num.corr()

sns.heatmap(data_num_cor, 
        xticklabels=data_num_cor.columns,
        yticklabels=data_num_cor.columns)


## Categorical features analysis

char_features = list(set(data.columns.tolist()) - set(numeric_features))

data_char = data[char_features].copy()

dc_dict = dict(data_char.nunique())

filt_cols = [key for key in dc_dict.keys() if dc_dict[key]<2]
filt_cols




grouped = data.groupby('grade')['int_rate'].mean().reset_index()

plt.bar(grouped['grade'], grouped['int_rate'], align='center', alpha=0.5, color='g')
