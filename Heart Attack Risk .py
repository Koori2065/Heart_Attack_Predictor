import numpy as np 
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.linear_model import LogisticRegression


df=pd.read_csv("F:\DATA\MAIN\/Heart Attack Risk.csv")

x=df.drop(['output'] ,  axis= 1 )
y=df['output']

msk=np.random.rand(len(df)) < 0.8

train_x = x[msk]
test_x = x[~msk]
train_y = y[msk]
test_y = y[~msk]

# Modeling

LR=LogisticRegression(C=1 , solver='liblinear').fit(train_x , train_y)

# Predict

test_y_=LR.predict(test_x)
prob_y=LR.predict_proba(test_x)   # between 0 , 1 and up to 0.5 is 1


conf_matrix = confusion_matrix(test_y, test_y_)
print("Confusion Matrix:\n", conf_matrix)

print (classification_report(test_y, test_y_))
