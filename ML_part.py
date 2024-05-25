# -*- coding: utf-8 -*-
#from opals import Import, Grid, Algebra, AddInfo, pyDM
#import opals
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import xgboost as xgb

#change directories:
#Felix:
#path = r'C:\Users\felix\OneDrive\Dokumente\TU Wien\Applied Earth Observation\2_2_Urban_Full_Waveform_Classification'
#Bettina:
path = r'C:\Users\betti\OneDrive\STUDIUM\SS24\Applied Earth Observation\2_2_Urban_Full_Waveform_Classification'
#Theresa:
#path = r'Bitte den Pfad angeben'
#Max:
#path = r'Bitte den Pfad angeben'

#change directory
os.chdir(path)

# create test data bc real one is not finished
# data = pd.DataFrame({"Normalized_Z": [1,2,5,2,2,1,4,0,0,0,1,2,3,1,2], 
#                      "Reflectance": [0.4,0.3,0,1,1,0.6,0.7,0.1,0.2,0.1,0.3,0.1,0.1,0.4,0],
#                      "Amplitude": [12,32,23,34,45,41,12,9,6,45,43,23,32,41,2],
#                      "Classification": [1,2,1,3,2,3,2,1,2,3,2,1,3,2,2]})

# feature_nams = ["Normalized_Z", "Reflectance", "Amplitude"]#, "EchoRatio", "EchoWidth",
#                 #"NormalSigma0"] #Backscatter Coefficient

# read all_data (all points) and subdata (classified points)
# use subdata to get train and test sets, then use model on all_data

#data = pd.read_csv("all_data.csv")
subdata = pd.read_csv("subdata.csv")

subdata = subdata.loc[~subdata["Classification"].isin([2, 10])]

feature_nams = ["EchoWidth", "Reflectance", "Amplitude"]#, "EchoRatio", "Normalized_Z",
                #"NormalSigma0"] #Backscatter Coefficien

feat_x = subdata[feature_nams].values
feat_x_std = StandardScaler().fit_transform(feat_x)

feat_y = (subdata[["Classification"]].values).ravel()

train_x, test_x, train_y, test_y = train_test_split(feat_x_std, feat_y, test_size=0.30, stratify=feat_y)


# Decision Tree
clf = DecisionTreeClassifier(max_depth=5) #max_depth=2
clf = clf.fit(train_x, train_y)

pred_y = clf.predict(test_x)
classification_report(test_y, pred_y)
print(f"{classification_report(test_y, pred_y)}\n")

disp = ConfusionMatrixDisplay.from_predictions(test_y, pred_y) #, display_labels=[]
disp.figure_.suptitle("Confusion Matrix")
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
plot_tree(clf, ax=ax, impurity=False, node_ids=False, feature_names=feature_nams) #, class_names=[]
plt.show()


# Random Forest 
rf = RandomForestClassifier()
rf.fit(train_x, train_y)

pred_y = clf.predict(test_x)
print(f"{classification_report(test_y, pred_y)}\n")

disp = ConfusionMatrixDisplay.from_predictions(test_y, pred_y) #, display_labels=[]
disp.figure_.suptitle("Confusion Matrix")
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
plot_tree(rf.estimators_[0], ax=ax, impurity=False, node_ids=False, feature_names=feature_nams) #, class_names=[]
plt.show()


# XGBoost
le = LabelEncoder()
feat_y_encoded = le.fit_transform(feat_y)

train_x, test_x, train_y, test_y = train_test_split(feat_x_std, feat_y_encoded, test_size=0.30, stratify=feat_y_encoded, random_state=42)

clf = xgb.XGBClassifier(max_depth=5, n_estimators=100, random_state=42)
clf.fit(train_x, train_y)

pred_y_encoded = clf.predict(test_x)
pred_y = le.inverse_transform(pred_y_encoded)

test_y_original = le.inverse_transform(test_y)

print(classification_report(test_y_original, pred_y))

disp = ConfusionMatrixDisplay.from_predictions(test_y_original, pred_y)
disp.figure_.suptitle("Confusion Matrix")
plt.show()

fig, ax = plt.subplots(figsize=(20, 20))
xgb.plot_tree(clf, num_trees=0, ax=ax)
plt.show()


### PCA Analysis #erstmals alle reingeben
pca = PCA(n_components=None, whiten=True).fit(feat_x_std)
feat_x_pca = pca.transform(feat_x_std)

pca_evar = pca.explained_variance_ratio_
print("Explained variance per component:")
print(pca_evar*100)

pca_evar_csum = np.cumsum(pca.explained_variance_ratio_)
print("\nCumulative explained variance:")
print(pca_evar_csum*100)

var_perc = 0.95
pca_dim = np.max(np.argwhere(pca_evar_csum <= var_perc))+1
print("\n# dimensions explaining ~%i%% of the total variance: %i" % (var_perc*100, pca_dim))

# Mit 95% Varianz EV nochmal DT
feat_x_std = feat_x_pca[:,0:1]

feat_y = (subdata[["Classification"]].values).ravel()

train_x, test_x, train_y, test_y = train_test_split(feat_x_std, feat_y, test_size=0.30, stratify=feat_y)

clf = DecisionTreeClassifier() #max_depth=2
clf = clf.fit(train_x, train_y)

pred_y = clf.predict(test_x)
print(f"{classification_report(test_y, pred_y)}\n")

features = feature_nams

# Erstelle ein DataFrame, um die Koeffizienten besser zu visualisieren
first_eigenvector = feat_x_pca[:,0]
coefficients = pd.DataFrame({'Feature': features, 'Coefficient': first_eigenvector[:3]})  # [:3] weil du nur drei Features hast

# Sortiere nach der Magnitude der Koeffizienten
coefficients['AbsCoefficient'] = coefficients['Coefficient'].abs()
coefficients_sorted = coefficients.sort_values(by='AbsCoefficient', ascending=False)

print(coefficients_sorted)
