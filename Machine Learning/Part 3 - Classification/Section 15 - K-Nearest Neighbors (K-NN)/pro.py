import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, neighbors

df = pd.read_csv('breast-cancer.data')
df.replace('?', -99999, inplace=True)

X = np.array(df.drop(['irradiat'], 1))
y = np.array(df[['irradiat']])

 

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(acc)