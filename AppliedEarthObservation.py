#Opals Standard Module
from opals import Import, Grid, Algebra, AddInfo, pyDM
#Opals addon-Module
from opals.workflows import clfTreeModelTrain, clfTreeModelApply, preAttribute
#import preTiling
import opals
import numpy as np
import os

#change directories:
#Felix:
path = r'C:\Users\felix\OneDrive\Dokumente\TU Wien\Applied Earth Observation\2_2_Urban_Full_Waveform_Classification'
#Bettina:
path = r'C:\Users\betti\OneDrive\STUDIUM\SS24\Applied Earth Observation\2_2_Urban_Full_Waveform_Classification'
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
#dm.sizePoint()

lf = pyDM.AddInfoLayoutFactory()
type, inDM = lf.addColumn(dm, "Id", True); assert inDM == True
type, inDM = lf.addColumn(dm, "Classification", True); assert inDM == True #ist das unsere Klasse? @Felix
type, inDM = lf.addColumn(dm, "Amplitude", True); assert inDM == True
type, inDM = lf.addColumn(dm, "EchoWidth", True); assert inDM == True
type, inDM = lf.addColumn(dm, "CrossSection", True); assert inDM == True
type, inDM = lf.addColumn(dm, "EchoNumber", True); assert inDM == True
type, inDM = lf.addColumn(dm, "NrOfEchos", True); assert inDM == True
type, inDM = lf.addColumn(dm, "RGIndex", True); assert inDM == True
type, inDM = lf.addColumn(dm, "Reflectance", True); assert inDM == True
type, inDM = lf.addColumn(dm, "Range", True); assert inDM == True
type, inDM = lf.addColumn(dm, "_IncidenceAngle", True); assert inDM == True
type, inDM = lf.addColumn(dm, "NormalizedZ", True); assert inDM == True
type, inDM = lf.addColumn(dm, "NormalSigma0", True); assert inDM == True
type, inDM = lf.addColumn(dm, "Gamma", True); assert inDM == True
type, inDM = lf.addColumn(dm, "Sigma", True); assert inDM == True
layout = lf.getLayout()

result = pyDM.NumpyConverter.searchPoint(dm, queryWin, layout, withCoordinates=True)

#Maschine learning

#test numpy array



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