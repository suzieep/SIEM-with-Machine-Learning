#READ FILE

from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits
import warnings
from IPython.display import Image
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

filename = '/Users/soojinlee/repo/Capstone/Data/csv/firewall_raw.csv'

raw_df = pd.read_csv(filename, encoding="ISO-8859-1")  # data frame

p_df = raw_df.drop(['createtime', 'reqtime', 'msg', 'src'], axis=1)

arr_total_rank = raw_df.groupby(
    "src")["msg"].count().sort_values(ascending=False)[:7]

p_df = p_df.replace({'action': 'Deny'}, {'action': 1})
p_df = p_df.replace({'action': 'Permit'}, {'action': 0})

#raw_df.info()

url_raw = p_df['url']

ex = []

for u in url_raw:
    u = str(u)
    ex.append(u.split('.')[-1])


data_raw = p_df['data']

met = []
size = []

for d in data_raw:
    d = str(d)
    met.append(d.split(' ')[0])
    size.append(len(d))

#print(met[1200])

df_p = pd.DataFrame(ex)
df_p = pd.get_dummies(df_p)
#print(df_p)

df_m = pd.DataFrame(met)
df_m = pd.get_dummies(df_m)
#print(df_m)


pd_size = pd.DataFrame(size, columns=['size'])
p_df = pd.concat([p_df, df_p, df_m, pd_size], axis=1)

del p_df['url']
del p_df['domain']
del p_df['data']

del df_p, df_m


x_data = p_df.copy()
x_data.drop('isattack', axis=1, inplace=True)
y_label = p_df['isattack']

del p_df

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_label, test_size=0.2, random_state=1)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=0)
dt_clf = DecisionTreeClassifier()
ada_clf = AdaBoostClassifier(n_estimators=100)

lr_final = LogisticRegression(C=10)

rf_clf.fit(x_train, y_train)
#dt_clf.fit(x_train,y_train)
#ada_clf.fit(x_train,y_train)

rf_pred = rf_clf.predict(x_test)
#dt_pred = dt_clf.predict(x_test)
#ada_pred = ada_clf.predict(x_test)

print('random forest accuracy', format(accuracy_score(y_test, rf_pred)))
#print('decision tree accuracy',format(accuracy_score(y_test,dt_pred)))
#print('adaboost accuracy',format(accuracy_score(y_test,ada_pred)))


print('random forest')
print(classification_report(y_test, rf_pred,
                            target_names=['0', '1'], digits=2))
print('decision tree')
#print(classification_report(y_test, dt_pred, target_names=['0','1'],digits=2))
print('adaboost')
#print(classification_report(y_test, ada_pred, target_names=['0','1'],digits=2))

warnings.filterwarnings('ignore')


cm = confusion_matrix(y_test, rf_pred)

print('True Negative : ', cm[0][0])
print('False Positive : ', cm[0][1])
print('False Negative : ', cm[1][0])
print('True Positive : ', cm[1][1])


def find_TP(y_true, y_pred):
    arr_TP = []
    for i in y_true.index:
        j = 0
        if((y_true[i] == 1) & (y_pred[j] == 1)):
            print("tp index : ", i)
            #print(raw_df.values[i])
            arr_TP.append(raw_df.values[i])
            j += 1
    return sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):
    arr_FN = []
    print('FN------------------------------')
    j = 0
    k = 0
    for i in y_true.index:
        if((y_true[i] == 1) & (y_pred[j] == 0)):
            print("FN ", j, "index : ", i)
            print(raw_df.values[i][2], raw_df.values[i][3])
            arr_FN.append(raw_df.values[i])
            print(arr_FN[k][2], arr_FN[k][3])
            k = k+1
        j = j+1
    return sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):
    arr_FP = []
    print('FP------------------------------')
    j = 0
    k = 0
    for i in y_true.index:
        if((y_true[i] == 0) & (y_pred[j] == 1)):
            print("FP ", j, "index : ", i)
            print(raw_df.values[i][2], raw_df.values[i][3])
            arr_FP.append(raw_df.values[i])
            print(arr_FP[k][2], arr_FP[k][3])
            k = k+1
        j = j+1
    return sum((y_true == 0) & (y_pred == 1))


def find_TN(y_true, y_pred):
    arr_TN = []
    j = 0
    for i in y_true.index:
        if((y_true[i] == 0) & (y_pred[j] == 0)):
            print("tn index : ", i)
            #print(raw_df.values[i])
            arr_TN.append(raw_df.values[i])
        j = j+1
    return sum((y_true == 0) & (y_pred == 0))


#print('TP:',find_TP(y_test, rf_pred))
print('FN:', find_FN(y_test, rf_pred))
print('FP:', find_FP(y_test, rf_pred))
#print('TN:',find_TN(y_test, rf_pred))
