import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Low'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forcast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)

forcast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forcast_col].shift(-forcast_out)
df.dropna(inplace=True)


X = np.array(df.drop(['label'], 1))
Y = np.array(df['label'])

X = preprocessing.scale(X)
# X = X[:-forcast_out+1]
# df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size=0.2)
clf = LinearRegression()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)

print(accuracy)