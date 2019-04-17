import pandas as pd
from preprocess import process, apply_model
from data_read import extract_df, extract_df18Q1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from matplotlib import pyplot


data = extract_df()
#data['id'] = range(data.shape[0]+1)[1:]


data_combine, y = process(data)

clf = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(data_combine, y, test_size=0.2, random_state=0)

clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)
#y_pred = [1 if val>0.8 else 0 for val in probs]
y_pred = clf.predict(X_test)

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)


# generate 2 class dataset
probs = probs[:, 1]
# predict class values
yhat = [1 if val>0.35 else 0 for val in probs]
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, yhat)
# calculate F1 score
f1 = f1_score(y_test, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(y_test, probs)

ar = recall.mean()

print('f1=%.3f auc=%.3f ap=%.3f ar=%.3f' % (f1, auc, ap, ar))



#############################################################################

#### Model prediction on 2018Q1

test = extract_df18Q1()
req_cols = data_combine.columns.tolist() + ['loan_status'] + ['mths_since_last_delinq']
req_cols = [val for val in req_cols if val!='yrs_since_last_delinq']
X_test, y_test = apply_model(test,req_cols)

probs = clf.predict_proba(X_test)[:,1]
#y_pred = [1 if val>0.8 else 0 for val in probs]
y_hat = clf.predict(X_test)

# generate 2 class dataset
#probs = probs[:, 1]
# predict class values
yhat = [1 if val>0.35 else 1 for val in probs]
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, yhat)
# calculate precision-recall AUC
auc_test = auc(recall, precision)
# calculate average precision score
ap_test = average_precision_score(y_test, probs)
ar_test = recall.mean()

print('auc=%.3f ap=%.3f ar= %.3f' %(auc_test, ap_test, ar_test))
accuracy_score(y_test, yhat)



#############################################################################


def k_fold_test(train_index, test_index):
    tr, ts = pd.Series(train_index), pd.Series(test_index)
    X_train, X_test = data_combine.iloc[tr,:], data_combine.iloc[ts,:]
    y_train, y_test = y[tr], y[ts]
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    probs = probs[:, 1]
    # predict class values
    yhat = [1 if val>0.35 else 0 for val in probs]
    precision, recall, thresholds = precision_recall_curve(y_test, yhat)
    acc = accuracy_score(y_test, y_pred)
    ap = average_precision_score(y_test, probs)
    ar = recall.mean()
    
    return ar, ap, acc


## K-Fold cross validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)


for train_index, test_index in kf.split(data_combine):
    #tr, ts = pd.Series(train_index), pd.Series(test_index)
    #print(train_index, test_index)
    recall, precision, accuracy = k_fold_test(train_index, test_index)
    print(recall, precision, accuracy)


#############################################################################



