#Opals Standard Module
from opals import Import, Grid, Algebra, AddInfo
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
#path = r'Bitte den Pfad angeben'
#Theresa:
#path = r'Bitte den Pfad angeben'
#Max:
#path = r'Bitte den Pfad angeben'

#change directory
os.chdir(path)

#Merge odm-files:
files = ['strip1_AoI.odm', 'strip2_AoI.odm', 'strip3_AoI.odm']

if os.path.isfile('strip1_AoI_dem.tif') == False and os.path.isfile('strip2_AoI_dem.tif') == False and os.path.isfile('strip3_AoI_dem.tif') == False:
     for file in files:
         name = file.split('.')
         grid_name = name[0] + '_dem.tif'
         Grid.Grid(inFile=file, outFile=grid_name, interpolation=opals.Types.GridInterpolator.movingPlanes, gridSize=0.5).run()



#Claculate the normalizedz:
AddInfo.AddInfo(inFile=files[0], gridFile='strip1_AoI_dem.tif', attribute='normalizedz=z-r[0]').run()


#Add a userdefined attribute
AddInfo.AddInfo(inFile=files[0], gridFile='strip1_AoI_dem.tif', attribute= '_newClass = normalizedz > 5 ? 2 : _newClass').run()