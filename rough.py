from preprocess import process
from data_read import extract_df
from supported_data import DType, numeric_features

data = extract_df()
#data['id'] = range(data.shape[0]+1)[1:]

data_combine, y = process(data)


data_num = data_combine[numeric_features].copy()



##############################################################


########################## Numeric features engineering

# loan_amnt - positively skewed - hence qube root transformation

data_num['loan_amnt'] = data_num['loan_amnt'] ** (1/3)


from sklearn.tree import DecisionTreeClassifier 

dtc = DecisionTreeClassifier()


from sklearn.tree import _tree

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print( "{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))
            
    recurse(0, 1)

cols = ['loan_amnt', 'revol_bal', 'annual_inc', 'dti', 'total_pymnt',
       'total_rec_int', 'last_pymnt_amnt']

tree_to_code(dtc, data_num.columns.tolist())
#
#from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
#import pydotplus
#from IPython.display import Image 
#
#dot_data = StringIO()



## checking the distribution of data
for col in data_num.columns.tolist():
    if col not in ['dti','total_rec_late_fee', 'recoveries',
       'collection_recovery_fee']:
        data_num[col] = data_num[col] ** (1/3)

## Binarization for numerica transformed features

features_to_binary = ['total_rec_late_fee', 'recoveries',
       'collection_recovery_fee']

from sklearn.preprocessing import Binarizer
bn = Binarizer(threshold=0.9)

data_num['collection_recovery_fee'] = bn.transform([data_num['collection_recovery_fee']])[0]


import matplotlib.pyplot as plt
['loan_amnt', 'revol_bal', 'annual_inc', 'dti', 'total_pymnt',
       'total_rec_int', 'total_rec_late_fee', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_amnt']

