"""
    PROIECT
    
    Colorarea imaginilor folosind autoencoder si invatarea automata
    
    Badea Adrian Catalin, grupa 334, anul III, FMI
"""

import pdb
from DataSet import *
from AeModel import *


data_set: DataSet = DataSet()
data_set.scene_name = 'forest'
ae_model: AeModel = AeModel(data_set)

ae_model.define_the_model()
ae_model.compile_the_model()
ae_model.train_the_model()
ae_model.evaluate_the_model()



data_set: DataSet = DataSet()
data_set.scene_name = 'coast'
ae_model: AeModel = AeModel(data_set)

ae_model.define_the_model()
ae_model.compile_the_model()
ae_model.train_the_model()
ae_model.evaluate_the_model()

