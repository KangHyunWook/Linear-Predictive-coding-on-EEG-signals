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
import os
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from models import *
from utils import loadData, splitTrainTestData

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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

from sklearn.metrics import f1_score, confusion_matrix

def loadFromPkl():
    with open("testX.pkl", 'rb') as f:
        testX=pickle.load(f)
    with open("testy.pkl", 'rb') as f:
        testy=pickle.load(f)

    return testX, testy

import pandas as pd
import seaborn as sns

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
    if args.data_path.split(os.path.sep)[-1]=='emotions.csv':
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

        preds=vanilla_model.predict(testX)
        correct=0
        y_true=[]
        y_pred=[]
        for i in range(len(preds)):
            true=testy[i].argmax()
            pred = preds[i].argmax()
            y_true.append(true)
            y_pred.append(pred)
            if true == pred:
                correct+=1
        print('acc:',correct/len(preds))
        print('f1_score:', f1_score(y_true, y_pred, average='weighted'))
        class_names=['negative','neutral', 'positive']
        c=confusion_matrix(y_true, y_pred)

        df_cm = pd.DataFrame(
            c, index=class_names, columns=class_names,
        )


        fig = plt.figure(figsize=(10,7))
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
        if args.model=='1':
            name='1d-CNN'
        elif args.model=='3':
            name='DNN'
        plt.title(f'{name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        # vanilla_model.evaluate(testX, testy)
    else:

        trainX, trainy = saveToPickle(args)
        vanilla_model=Model(inputs, output)

        vanilla_model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['acc'])
        vanilla_model.fit(trainX, trainy, epochs=30, batch_size=64)
        vanilla_model.save_weights('vanilla_weights.h5')







#
