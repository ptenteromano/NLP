from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('spambase.data').values
np.random.shuffle(data)

X = data[:,:48]
Y = data[:, -1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

modelNB = MultinomialNB()
modelNB.fit(Xtrain, Ytrain)
print("NB score:", modelNB.score(Xtest, Ytest))


modelAB = AdaBoostClassifier()
modelAB.fit(Xtrain, Ytrain)

print("AdaBoost score:", modelAB.score(Xtest, Ytest))
