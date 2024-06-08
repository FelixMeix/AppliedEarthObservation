# -*- coding: utf-8 -*-
#from opals import Import, Grid, Algebra, AddInfo, pyDM
#import opals
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import xgboost as xgb
import shap


# define directories --------------------------------------------------------------------------------------------------

#Felix:
#path = r'C:\Users\felix\OneDrive\Dokumente\TU Wien\Applied Earth Observation\2_2_Urban_Full_Waveform_Classification'
#Bettina:
path = r'C:\Users\betti\OneDrive\STUDIUM\SS24\Applied Earth Observation\2_2_Urban_Full_Waveform_Classification'
#Theresa:
#path = r'Bitte den Pfad angeben'
#Max:
#path = r'Bitte den Pfad angeben'


# change directory  ---------------------------------------------------------------------------------------------------

os.chdir(path)


# read all_data (all points) and subdata (classified points) ----------------------------------------------------------
# use subdata to get train and test sets, then use model on all_data

data = pd.read_csv("all_data.csv")
subdata = pd.read_csv("subdata.csv")

subdata = subdata.loc[~subdata["Classification"].isin([2, 10])]
subdata['Classification'].loc[subdata["Classification"].isin([4])] = 5
subdata['_NormalSigma0'].loc[np.isnan(subdata["_NormalSigma0"])] = 0

data = data.loc[~data["Classification"].isin([2, 10])]
data['Classification'].loc[data["Classification"].isin([4])] = 5
data['_NormalSigma0'].loc[np.isnan(data["_NormalSigma0"])] = 0


#drop class 7 (others) ------------------------------------------------------------------------------------------------

#subdata = subdata[subdata['Classification'] != 7]

feature_nams = ["EchoWidth", "Reflectance", "Amplitude", "EchoRatio", "NormalizedZ",
                "_NormalSigma0", "_IncidenceAngle", "NrOfEchos", "_RAomega","Range"]


# describe the subdata ------------------------------------------------------------------------------------------------

subdata_describe = subdata.groupby("Classification").describe()
subdata_describe.to_csv('subdata_statistic.csv')

feat_x = subdata[feature_nams].values
feat_y = (subdata[["Classification"]].values).ravel()

train_x, test_x, train_y, test_y = train_test_split(feat_x, feat_y, test_size=0.30, stratify=feat_y)


# Decision Tree -------------------------------------------------------------------------------------------------------

# clf = DecisionTreeClassifier(max_depth=5) #max_depth=2
# clf = clf.fit(train_x, train_y)

# pred_y = clf.predict(test_x)
# classification_report(test_y, pred_y)
# print(f"{classification_report(test_y, pred_y)}\n")

# disp = ConfusionMatrixDisplay.from_predictions(test_y, pred_y) #, display_labels=[]
# disp.figure_.suptitle("Confusion Matrix")
# plt.show()

#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
#plot_tree(clf, ax=ax, impurity=False, node_ids=False, feature_names=feature_nams) #, class_names=[]
#plt.show()


# # Random Forest -----------------------------------------------------------------------------------------------------

# rf = RandomForestClassifier()
# rf.fit(train_x, train_y)

# pred_y = clf.predict(test_x)
# print(f"{classification_report(test_y, pred_y)}\n")

# disp = ConfusionMatrixDisplay.from_predictions(test_y, pred_y) #, display_labels=[]
# disp.figure_.suptitle("Confusion Matrix")
# plt.show()

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
# plot_tree(rf.estimators_[0], ax=ax, impurity=False, node_ids=False, feature_names=feature_nams) #, class_names=[]
# plt.show()


# XGBoost -------------------------------------------------------------------------------------------------------------

le = LabelEncoder()
feat_y_encoded = le.fit_transform(feat_y)

model = xgb.XGBClassifier(use_label_encoder=True)

train_x, test_x, train_y, test_y = train_test_split(feat_x, feat_y_encoded, test_size=0.30, 
						stratify=feat_y_encoded, random_state=42)

# parameter optimisation
# params = {
#     'max_depth': [6, 10, 15, 20],
#     'learning_rate': [0.001, 0.01, 0.1, 0.2, 0, 3],
#     'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#     'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#     'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
#     'gamma': [0, 0.25, 0.5, 1.0],
#     'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
#     'n_estimators': [100,200,300,400,500,600,700,800,900,1000]}

eval_set = [(train_x, train_y), (test_x, test_y)]

# kfold = KFold(n_splits=2, shuffle=True)
# rs_xgbc = RandomizedSearchCV(model, param_distributions=params, n_iter=2,
#                               n_jobs=1, verbose=3, cv=kfold, scoring='neg_log_loss', refit=True, random_state=42)

# #search_time_start = time.time()
# rs_xgbc.fit(train_x, train_y, eval_metric=[
#      "mlogloss", 'merror'], eval_set=eval_set, verbose=False)

# best_params = rs_xgbc.best_params_
# print("Best params: ")
# for param_name in sorted(best_params.keys()):
#     print('%s: %r' % (param_name, best_params[param_name]))

# model = xgb.XGBClassifier(max_depth=6,
#                           learning_rate=0.1,
#                           subsample=0.7,
#                           colsample_bytree=0.7,
#                           colsample_bylevel=0.8,
#                           min_child_weight=7.0,
#                           gamma=0.5,
#                           reg_lambda=0.1,
#                           n_estimators=300)


# XGBoost model with optimised parameters

model = xgb.XGBClassifier(max_depth=6,
                          learning_rate=0.1,
                          subsample=0.8,
                          colsample_bytree=0.7,
                          colsample_bylevel=0.8,
                          min_child_weight=7.0,
                          gamma=0.5,
                          reg_lambda=0.1,
                          n_estimators=300)#, use_label_encoder=True)


# model.fit(train_x, train_y, eval_metric=[
#         "mlogloss", 'merror'], eval_set=eval_set, verbose=False)

# #y_pred = model.predict(test_x)

# #clf = xgb.XGBClassifier(max_depth=5, n_estimators=100, random_state=42)
# #clf.fit(train_x, train_y)

# pred_y_encoded = model.predict(test_x)
# pred_y = le.inverse_transform(pred_y_encoded)

# test_y_original = le.inverse_transform(test_y)

# print(classification_report(test_y_original, pred_y))

# disp = ConfusionMatrixDisplay.from_predictions(test_y_original, pred_y)
# disp.figure_.suptitle("Confusion Matrix")
# plt.show()


#Feature importance (shap)

# fig, ax = plt.subplots()
# shap_values = shap.TreeExplainer(model).shap_values(feat_x)
# shap.summary_plot(shap_values, feat_x, show=False)
# #shap.dependence_plot("min_fixation_duration", shap_values[0], X, show=False)

# plt.xlabel('SHAP value: average impact on model magnitude')
# fig.tight_layout()
# plt.show()


#fig, ax = plt.subplots(figsize=(20, 20))
#xgb.plot_tree(clf, num_trees=0, ax=ax)
#plt.show()


# train model on entire subdata -----------------------------------------------------------------------------------------

train_x, test_x, train_y, test_y = train_test_split(feat_x, feat_y_encoded, test_size=0.0001, 
							stratify=feat_y_encoded, random_state=42)

model.fit(train_x, train_y, eval_metric=[
         "mlogloss", 'merror'], eval_set=eval_set, verbose=False)


# predict on entire unclassified data -----------------------------------------------------------------------------------

unclass_data = data.loc[data["Classification"] == 0]

feat_x = unclass_data[feature_nams].values

pred_y_encoded = model.predict(feat_x)
pred_y = le.inverse_transform(pred_y_encoded)

unclass_data["Classification"] = list(pred_y)

unclass_data.to_csv('predicted_data_without_stripes.csv')


# describe the predicted data -------------------------------------------------------------------------------------------

data_describe = unclass_data.groupby("Classification").describe()
data_describe.to_csv('predicted_data_statistic.csv')


# merge all data --------------------------------------------------------------------------------------------------------

#all_data = subdata.merge(unclass_data)
all_data = pd.concat([subdata, unclass_data])
all_data.to_csv("all_data_with_classes.csv")


# PCA -------------------------------------------------------------------------------------------------------------------

pca = PCA(n_components=None, whiten=True).fit(feat_x)
feat_x_pca = pca.transform(feat_x)

pca_evar = pca.explained_variance_ratio_
print("Explained variance per component:")
print(pca_evar*100)

pca_evar_csum = np.cumsum(pca.explained_variance_ratio_)
print("\nCumulative explained variance:")
print(pca_evar_csum*100)

var_perc = 0.95
pca_dim = np.max(np.argwhere(pca_evar_csum <= var_perc))+1
print("\n# dimensions explaining ~%i%% of the total variance: %i" % (var_perc*100, pca_dim))


# those with 95% cumulative variance again for ML model

feat_x = feat_x_pca[:,0:pca_dim+1]

train_x, test_x, train_y, test_y = train_test_split(feat_x, feat_y_encoded, test_size=0.30, 
							stratify=feat_y_encoded, random_state=42)

model = xgb.XGBClassifier()

model.fit(train_x, train_y)

pred_y_encoded = model.predict(test_x)
pred_y = le.inverse_transform(pred_y_encoded)

test_y_original = le.inverse_transform(test_y)

print(classification_report(test_y_original, pred_y))

features = feature_nams

# dataframe for top three eigenvectors

first_eigenvector = feat_x_pca[:,0]
second_eigenvector = feat_x_pca[:,1]
third_eigenvector = feat_x_pca[:,2]

eigenvectors = [first_eigenvector, second_eigenvector, third_eigenvector]

for eigenvec in eigenvectors:
    coefficients = pd.DataFrame({'Feature': features, 'Coefficient': eigenvec[:10]})

    # sort by feature magnitude
    coefficients['AbsCoefficient'] = coefficients['Coefficient'].abs()
    coefficients_sorted = coefficients.sort_values(by='AbsCoefficient', ascending=False)

    print(coefficients_sorted)
