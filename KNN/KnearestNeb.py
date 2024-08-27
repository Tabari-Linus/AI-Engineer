import numpy as np 
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("knn/breast-cancer-wisconsin.data")
df.replace('?', -99999, inplace=True)

df.drop(['id'],inplace=True,axis=1)

X = np.array(df.drop(['class'], axis=1))
Y = np.array(df['class'])

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accurracy = clf.score(x_test,y_test)

print(accurracy)

sample_meansures = np.array([4,2,1,1,1,2,3,2,1])
sample_meansures = sample_meansures.reshape(1,-1)
prediction = clf.predict(sample_meansures)
print(prediction)
