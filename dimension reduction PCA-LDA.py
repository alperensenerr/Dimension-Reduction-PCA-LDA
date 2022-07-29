#1. libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix

# data set
veriler = pd.read_csv('Wine.csv',sep = ',')
X = veriler.iloc[:, 0:13].values
y = veriler.iloc[:, 13].values

# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA

pca = PCA(n_components = 2)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#LDA 
lda = LDA(n_components = 2)

X_train_lda = lda.fit_transform(X_train,y_train)
X_test_lda = lda.transform(X_test)

#LR before dimension reduction

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#LR after pca
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train_pca,y_train)

#LR after LDA 
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

#predictions
y_pred = classifier.predict(X_test) #without Dimension Reduction

y_pred_pca = classifier2.predict(X_test_pca) #pred with PCA

y_pred_lda = classifier_lda.predict(X_test_lda) #pred with LDA


#actual / without Dimension reduction
print('gercek / DRsız')
cm = confusion_matrix(y_test,y_pred)
print(cm)

#actual / with PCA 
print("gercek / pca ile")
cm2 = confusion_matrix(y_test,y_pred_pca)
print(cm2)

#actual / with LDA 
print('lda ve orijinal')
cm3 = confusion_matrix(y_test,y_pred_lda)
print(cm3)











