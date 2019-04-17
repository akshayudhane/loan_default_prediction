import pandas as pd
import numpy as np
from data_read import extract_df
from numpy import dtype
from supported_data import DType, numeric_features, char_features
from sklearn.preprocessing import Binarizer
from sklearn import preprocessing
import scipy.stats as scs

le = preprocessing.LabelEncoder()
bn = Binarizer(threshold=0.9)


# Deleting the data with 97.5% missing values
def del_blank_value_cols(data):
    data_miss = data.isna().sum()
    data = data.drop(columns=[col for col in data_miss[data_miss>=data.shape[0]*0.90].index])
    return data
    

def remove_correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
                    print(colname)
    return dataset
    


def remove_chi_ind(data,threshold):
    def categories(series):
        return range(int(series.min()), int(series.max()) + 1)


    def chi_square_of_df_cols(df, col1, col2):
        df_col1, df_col2 = df[col1], df[col2]

        result = [[sum((df_col1 == cat1) & (df_col2 == cat2))
                   for cat2 in categories(df_col2)]
                  for cat1 in categories(df_col1)]
    
        return scs.chi2_contingency(result)

    col_corr = set()
    dcc_dict = dict(data.nunique())
    filt_col = [key for key in dcc_dict.keys() if dcc_dict[key]<25]
    data = data[filt_col].copy()
    
    chi_ind = pd.DataFrame(0, columns=data.columns.tolist(),
                   index=data.columns.tolist())

    for col1 in data.columns.tolist():
        for col2 in data.columns.tolist():
            if col1!=col2:
                pval = chi_square_of_df_cols(data,col1,col2)[1]
                if pval >= threshold:
                    chi_ind.loc[col1,col2]= pval
    for i in range(len(chi_ind.columns)):
        for j in range(i):
            if (chi_ind.iloc[i, j] >= threshold) and (chi_ind.columns[j] not in col_corr):
                colname = chi_ind.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in data.columns:
                    del data[colname] # deleting the column from the dataset
                    print(colname)
    return data
  

def process(data):
    #data = data[data['loan_status'].isin(['Fully Paid', 'Charged Off'])].reset_index()
    #
    data = del_blank_value_cols(data)
    data = data.loc[data['loan_status'].isin(['Fully Paid', 'Charged Off']),:]

    ## removing desc because it's having so many na values in the data
    description = data['desc']; del data['desc']
    
    # convert months to years and make 7 bins including na's
    data['yrs_since_last_delinq'] = (data['mths_since_last_delinq'].astype(float)/12).round(0)
    data.loc[data['yrs_since_last_delinq']>=5.00, 'yrs_since_last_delinq'] = '5_plus'
    data['yrs_since_last_delinq'].fillna('year_miss', inplace=True)
    data['yrs_since_last_delinq'] = data['yrs_since_last_delinq'].astype('str')
    
    del data['mths_since_last_delinq']
    
    # converting emp_length into bins
    data.loc[data['emp_length'].isin(['1 year', '3 years']), 'emp_length'] = 'one_to_three'
    data.loc[data['emp_length'].isin(['4 years', '5 years', '5_plus', '6 years']), 'emp_length'] = 'four_to_six'
    data.loc[data['emp_length'].isin(['7 years', '8 years', '9 years']), 'emp_length'] = 'seven_to_nine'
    data['emp_length'] = data['emp_length'].replace({'< 1 year': 'lt 1 year', '10+ years': 'gt 10 years'})
    data['emp_length'].fillna('miss_exp', inplace=True)
    # dropping na rows since he ratio is only 90%
    
    
    data['revol_util'] = data['revol_util'].str[:-1].astype(float)
    data['revol_util'].fillna(data['revol_util'].mean(), inplace=True)
    
    data.drop(['emp_title', 'title', 'last_pymnt_d', 'chargeoff_within_12_mths', 'collections_12_mths_ex_med',
               'pub_rec_bankruptcies', 'tax_liens'], axis=1, inplace=True)
    
    #data = data.dropna()
    
    data_num = data[numeric_features].copy()
    data_char = data[char_features].copy()
    
    # percentage columns
    data_num['int_rate'] = data_num['int_rate'].str[:-1].astype(float)
    
    # Remove near zero variance features from numeric variables
    data_var = data_num.var()
    for x in list(data_var[data_var <= 0.9].index):
        del data_num[x]
    
    # Remove highly correlated features 
    data_num = remove_correlation(data_num, 0.8)
    
#    for col in data_num.columns.tolist():
#        data_num[col].plot.box()
    
    ## applying cube root transformation for numeric features
    for col in data_num.columns.tolist():
        if col not in ['dti','total_rec_late_fee', 'recoveries','collection_recovery_fee']:
            data_num[col] = data_num[col] ** (1/3)

#    for col in data_num.columns.tolist():
#            data_num[col].plot.box()

    ## Binarization for numerica transformed features
    
    features_to_binary = ['total_rec_late_fee','collection_recovery_fee']
    
    for cols in features_to_binary:
        if cols in data_num.columns.tolist():
            data_num[cols] = bn.transform([data_num[cols]])[0]
    
    # remove columns which have only 1 category    
    dcc_dict = dict(data_char.nunique())
    drop_cols = [key for key in dcc_dict.keys() if dcc_dict[key]<2]
    
    # deleting features with only single category
    data_c = data_char.drop(columns=drop_cols).copy()
    
    # Convert the categorical columns to codes
    for col in data_c.columns.tolist():
        data_c[col] = le.fit_transform(data_c[col])
        #data_c[col] = data_c[col].astype(str)
        
    ##### Chi-square association filter
    data_c = remove_chi_ind(data_c, 0.05)
    
    print(data_c.shape, data_num.shape)
    #data_num = data_num.iloc[data_c.index.tolist(),:]    
    
    data_combine = pd.concat([data_c, data_num], axis=1)
    #data_combine.reset_index(inplace=True)
    y = data_combine['loan_status']; del data_combine['loan_status']
    return data_combine, y


def apply_model(data, req_cols):
    data = data.loc[:,req_cols].copy()
    
    data['yrs_since_last_delinq'] = (data['mths_since_last_delinq'].astype(float)/12).round(0)
    data.loc[data['yrs_since_last_delinq']>=5.00, 'yrs_since_last_delinq'] = '5_plus'
    data['yrs_since_last_delinq'].fillna('year_miss', inplace=True)
    data['yrs_since_last_delinq'] = data['yrs_since_last_delinq'].astype('str')
    
    del data['mths_since_last_delinq']

    
    data = data.dropna()
    
    data = data.loc[data['loan_status'].isin(['Fully Paid', 'Charged Off']),:]
    
    num_cols =list(set(req_cols) & set(numeric_features))
    
    data_num = data.loc[:,num_cols].copy()
    
    cat_col = list(set(req_cols+['yrs_since_last_delinq']) & set(char_features))
    data_char = data.loc[:,cat_col].copy()
    
    if('revol_util' in req_cols):
        data_num['revol_util'] = data_num['revol_util'].str[:-1].astype(float)
        data_num['revol_util'].fillna(data_num['revol_util'].mean(), inplace=True)

    if('int_rate' in req_cols):
        data_num['int_rate'] = data_num['int_rate'].str[:-1].astype(float)
        
    ## applying cube root transformation for numeric features
    for col in data_num.columns.tolist():
        if col not in ['dti','total_rec_late_fee', 'recoveries','collection_recovery_fee']:
            data_num[col] = data_num[col] ** (1/3)
    
    ## Binarization for numerica transformed features
    
    features_to_binary = ['total_rec_late_fee','collection_recovery_fee']
    
    for cols in features_to_binary:
        if cols in data_num.columns.tolist():
            data_num[cols] = bn.transform([data_num[cols]])[0]

    # char feature eng
    if 'emp_length' in data_char.columns.tolist():
        # converting emp_length into bins
        data_char.loc[data_char['emp_length'].isin(['1 year', '3 years']), 'emp_length'] = 'one_to_three'
        data_char.loc[data_char['emp_length'].isin(['4 years', '5 years', '5_plus', '6 years']), 'emp_length'] = 'four_to_six'
        data_char.loc[data_char['emp_length'].isin(['7 years', '8 years', '9 years']), 'emp_length'] = 'seven_to_nine'
        data_char['emp_length'] = data_char['emp_length'].replace({'< 1 year': 'lt 1 year', '10+ years': 'gt 10 years'})
        data_char['emp_length'].fillna('miss_exp', inplace=True)
    
    data_char = data_char.loc[data_char['loan_status'].isin(['Fully Paid', 'Charged Off']),:]

    for col in data_char.columns.tolist():
        data_char[col] = le.fit_transform(data_char[col])

    print(data_char.shape, data_num.shape)
    #data_num = data_num.iloc[data_c.index.tolist(),:]    
        
    data_combine = pd.concat([data_char, data_num], axis=1)
    #data_combine.reset_index(inplace=True)
    y = data_combine['loan_status']; del data_combine['loan_status']
    return data_combine, y



#for col in data_combine.columns.tolist():
#    print(col, chi_square_of_df_cols(data_combine, col, 'responce'))


##### preprocessing for feature selection of categorical information
#temp = data_c.copy()
#
#for col in temp.columns.tolist():
#    temp[col] = temp[col].astype(str)
#    
#    
#y = temp['loan_status']; del temp['loan_status']
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(temp, y, test_size=0.2, random_state=0)
#
#
#from sklearn.ensemble import RandomForestClassifier
#rfc = RandomForestClassifier(criterion ='entropy')
#
#rfc.fit(X_train,y_train)
#
#y_pred = rfc.predict(X_test)
#
#from sklearn.metrics import accuracy_score
#
#accuracy_score(y_test, y_pred)
#
#feat_labels = temp.columns.tolist()
#
#for feature in zip(feat_labels, rfc.feature_importances_):
#    print(feature)
#    
#from sklearn.feature_selection import SelectFromModel
#
#sfm = SelectFromModel(rfc, threshold=0.05)
#
#sfm.fit(temp, y)
#
#imp_features = []
#for feature_list_index in sfm.get_support(indices=True):
#    print(feat_labels[feature_list_index])
#    imp_features.append(feat_labels[feature_list_index])
#
### model with updated features
##imp_cat_features = ['title', 'last_pymnt_d', 'delinq_2yrs', 'emp_title', 'purpose']
#
#temp_new = temp[['title', 'last_pymnt_d', 'delinq_2yrs', 'emp_title', 'purpose']].copy()
#
#    
#X_train, X_test, y_train, y_test = train_test_split(temp_new, y, test_size=0.2, random_state=0)
#
#
#from sklearn.ensemble import RandomForestClassifier
#rfc = RandomForestClassifier()
#
#rfc.fit(X_train,y_train)
#
#y_pred = rfc.predict(X_test)
#
#from sklearn.metrics import accuracy_score
#
#accuracy_score(y_test, y_pred)
#
#
#######
#

##
#['inq_last_6mths', 'pub_rec', 'purpose', 'pub_rec_bankruptcies',
#       'home_ownership', 'verification_status', 'grade', 'emp_length', 'loan_status']
#    
#['loan_amnt', 'revol_bal', 'revol_util', 'int_rate', 'dti',
#       'total_rec_int', 'total_rec_late_fee', 'last_pymnt_amnt']