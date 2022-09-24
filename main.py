"""
#model 
    1: 1D convolution
    2: center loss
    3: DNN
"""
from keras.layers import Input
from keras.models import Model
import argparse

import numpy as np
import pickle

from sklearn.utils import shuffle
from models import *
from utils import *

ap=argparse.ArgumentParser()
ap.add_argument('--data-path', required=True)
ap.add_argument('--test', action='store_true')
ap.add_argument('--model', required=True)

args=ap.parse_args()

def saveToPickle(args):

    data, labels=loadData(args)

    trainX, testX, trainy, testy=splitTrainTestData(data, labels)
    with open("testX.pkl", 'wb') as f:
        pickle.dump(testX, f, pickle.HIGHEST_PROTOCOL)
    
    with open("testy.pkl", 'wb') as f:
        pickle.dump(testy, f, pickle.HIGHEST_PROTOCOL)
    
    return trainX, trainy
    
def loadFromPkl():
    with open("testX.pkl", 'rb') as f:
        testX=pickle.load(f)
    with open("testy.pkl", 'rb') as f:
        testy=pickle.load(f)    

    return testX, testy

if args.model=='2':
    inputs=Input((988,1))      
    aux_input=Input((3,))
    
    output, side= get_centermodel_output(inputs, aux_input, 3)
    
    if not args.test:

        vanilla_model=Model([inputs, aux_input], [output, side])
        
        vanilla_model.compile(loss=['categorical_crossentropy', zero_loss],
                loss_weights=[1, 0.1], optimizer='adam', metrics=['acc'])
        vanilla_model.save_weights('vanilla_weights.h5')

        vanilla_model.evaluate(testX, testy)
    else:
        pass
#when model is not 2            
else: 
    dim=988
    if args.data_path=='emotions.csv':
        dim=2548
    
    inputs=Input((dim, 1))
    if args.model=='3':
        inputs=Input((dim,))

    
    output= create(args.model, inputs, 3)

    if args.test:
        testX, testy= loadFromPkl()
        vanilla_model=Model(inputs, output)
        
        vanilla_model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['acc'])
        vanilla_model.load_weights('vanilla_weights.h5')

        vanilla_model.evaluate(testX, testy)
    else:
            
        trainX, trainy = saveToPickle(args)
        vanilla_model=Model(inputs, output)

        vanilla_model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['acc'])
        vanilla_model.fit(trainX, trainy, epochs=30, batch_size=64)
        vanilla_model.save_weights('vanilla_weights.h5')















