import pandas as pd
from supported_data import DType, sel_cols



def extract_df(**kwargs):
    return pd.read_csv(r'E:\Akshay\ML\data\LoanStats3a.csv', dtype=DType)
    

def extract_df18Q1(**kwargs):
    return pd.read_csv(r'E:\Akshay\ML\data\LoanStats_2018Q1.csv', usecols=sel_cols,dtype=DType)
