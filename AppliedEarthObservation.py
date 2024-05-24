from opals import Import, Grid, Algebra, AddInfo, pyDM
import opals
import numpy as np
import os
import pandas as pd

#change directories:
#Felix:
path = r'C:\Users\felix\OneDrive\Dokumente\TU Wien\Applied Earth Observation\2_2_Urban_Full_Waveform_Classification'
#Bettina:
#path = r'C:\Users\betti\OneDrive\STUDIUM\SS24\Applied Earth Observation\2_2_Urban_Full_Waveform_Classification'
#Theresa:
#path = r'Bitte den Pfad angeben'
#Max:
#path = r'Bitte den Pfad angeben'

#change directory
os.chdir(path)

#Merge odm-files:
files = ['strip1_AoI.odm', 'strip2_AoI.odm', 'strip3_AoI.odm']

#filter the region of interest:
#if os.path.isfile('strip_region.odm') == False:
 #   Import.Import(inFile=files, outFile='strip_region.odm', filter='region[Region_Filter.shp]').run()


#load all point and their classification
odm = 'strip_region_test.odm'

dm = pyDM.Datamanager.load(odm, False, False)
limit = dm.getLimit()
queryWin = pyDM.Window(limit.xmin, limit.ymin, limit.xmax, limit.ymax)

lf = pyDM.AddInfoLayoutFactory()
type, inDM = lf.addColumn(dm, "Id", True); assert inDM == True
type, inDM = lf.addColumn(dm, "Classification", True); assert inDM == True #unsere Klassen
type, inDM = lf.addColumn(dm, "Amplitude", True); assert inDM == True
type, inDM = lf.addColumn(dm, "EchoWidth", True); assert inDM == True
#type, inDM = lf.addColumn(dm, "CrossSection", True); assert inDM == True #keine Werte?
type, inDM = lf.addColumn(dm, "EchoNumber", True); assert inDM == True
type, inDM = lf.addColumn(dm, "NrOfEchos", True); assert inDM == True
type, inDM = lf.addColumn(dm, "RGIndex", True); assert inDM == True
type, inDM = lf.addColumn(dm, "Reflectance", True); assert inDM == True
type, inDM = lf.addColumn(dm, "Range", True); assert inDM == True
#type, inDM = lf.addColumn(dm, "_IncidenceAngle", True); assert inDM == True
#type, inDM = lf.addColumn(dm, "NormalizedZ", True); assert inDM == True #Errorrrr
#type, inDM = lf.addColumn(dm, "NormalSigma0", True); assert inDM == True
#type, inDM = lf.addColumn(dm, "_Gamma", True); assert inDM == True
#type, inDM = lf.addColumn(dm, "_Sigma", True); assert inDM == True # keine Werte?
layout = lf.getLayout()

result = pyDM.NumpyConverter.searchPoint(dm, queryWin, layout, withCoordinates=True)


#Maschine learning
import numpy as np

# import seaborn as sns
#
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from sklearn.metrics import classification_report, ConfusionMatrixDisplay
# from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.ensemble import RandomForestClassifier
# from matplotlib import pyplot as plt

#np.set_printoptions(suppress=True)

data = pd.DataFrame(result)
subdata = data.loc[data["Classification"] != 0]  #subdata with classes to train and test DT

#data.to_csv('all_data.csv', index=False)
#subdata.to_csv('subdata.csv', index=False)

#i=0
#pd_penguins = sns.load_dataset("penguins")
#pd_penguins = pd_penguins.dropna()

# feature_nams = ["Normalized_Z", "NrOfEchos", "Reflectance", "Amplitude"]
#
# feat_x = data[feature_nams].values
# feat_x_std = StandardScaler().fit_transform(feat_x)
#
# feat_y = (data[["Classification"]].values).ravel()
#
# train_x, test_x, train_y, test_y = train_test_split(feat_x, feat_y, test_size=0.30, stratify=feat_y)
#
# clf = DecisionTreeClassifier() #max_depth=2
# clf = clf.fit(train_x, train_y)
#
# pred_y = clf.predict(test_x)
# print(f"{classification_report(test_y, pred_y)}\n")
#
# disp = ConfusionMatrixDisplay.from_predictions(test_y, pred_y) #, display_labels=["Adelie", "Gentoo", "Chinstrap"]
# disp.figure_.suptitle("Confusion Matrix")
# plt.show()
#
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
# plot_tree(clf, ax=ax, impurity=False, node_ids=False, feature_names=feature_nams) #, class_names=["Adelie", "Gentoo", "Chinstrap"]
# plt.show()
#
# ### PCA Analysis
# from sklearn.decomposition import PCA
# pca = PCA(n_components=None, whiten=True).fit(feat_x)
# feat_x_pca = pca.transform(feat_x)
#
# pca_evar = pca.explained_variance_ratio_
# print("Explained variance per component:")
# print(pca_evar*100)
#
# pca_evar_csum = np.cumsum(pca.explained_variance_ratio_)
# print("\nCumulative explained variance:")
# print(pca_evar_csum*100)
#
# var_perc = 0.95
# pca_dim = np.max(np.argwhere(pca_evar_csum <= var_perc))+1
# print("\n# dimensions explaining ~%i%% of the total variance: %i" % (var_perc*100, pca_dim))
#
#
# i=0
#


#unclassify all points and write them back in the odm:
# classArray = result['Classification']
# classArray.fill(0)

setObj = {}
setObj['Id'] = result['Id']
setObj['Classification'] = result['Classification']

pyDM.NumpyConverter.setById(setObj, dm, layout)
dm.save()

i=0

# if os.path.isfile('strip1_AoI_dem.tif') == False and os.path.isfile('strip2_AoI_dem.tif') == False and os.path.isfile('strip3_AoI_dem.tif') == False:
#      for file in files:
#          name = file.split('.')
#          grid_name = name[0] + '_dem.tif'
#          Grid.Grid(inFile=file, outFile=grid_name, interpolation=opals.Types.GridInterpolator.movingPlanes, gridSize=0.5).run()
#
#
#
# #Claculate the normalizedz:
# AddInfo.AddInfo(inFile=files[0], gridFile='strip1_AoI_dem.tif', attribute='normalizedz=z-r[0]').run()
#
#
# #Add a userdefined attribute
# AddInfo.AddInfo(inFile=files[0], gridFile='strip1_AoI_dem.tif', attribute= '_newClass = normalizedz > 5 ? 2 : _newClass').run()