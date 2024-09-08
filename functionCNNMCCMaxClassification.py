import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

import warnings

warnings.filterwarnings('ignore')
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization,Convolution2D
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense

#from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils.vis_utils import plot_model
from glob import glob
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import *


class CNNFunction:
    def __init__(self, X_train, X_test, y_train, y_test, D, intParams, bounds, no_classes,batch_size,
                 early_stop=False):
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        self.D = D
        self.intParams = intParams
        self.bounds = bounds
        self.no_classes = no_classes
        self.early_stop = early_stop
        self.batch_size = batch_size

        # pomocna promenljiva
        self.feature_size = self.X_train.shape[1]

        # ovo nam koristi za solution
        self.y_test_length = len(y_test)

        #sada reshape X_train i X_test za klasifikaciju
        self.X_train = self.X_train.reshape(len(self.X_train), self.X_train.shape[1], 1)
        self.X_test = self.X_test.reshape(len(self.X_test), self.X_test.shape[1], 1)

        # pomocne promenljvie
        self.y_train1 = np.zeros(shape=len(y_train))
        self.y_test1 = np.zeros(shape=len(y_test))

        for i in range(len(self.y_test)):
            self.y_test1[i] = np.argmax(self.y_test[i])

        for i in range(len(self.y_train1)):
            self.y_train1[i] = np.argmax(self.y_train[i])

        #ovo je zbog frameworka
        self.features = [1,1,1,1,1,1]

        #self.y_test_length = len(self.test_data.classes)

        # postavjanje niza sa granicama vrednosti parametara
        # D je ukupna duzina niza i prosledjuje se iz glavnog koda

        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.name = "CNN Function"

        self.lb[0] = self.bounds['lb_lr']  # lower bound za learning rate, float
        self.ub[0] = self.bounds['ub_lr']  # lower bound za learning rate, float

        self.lb[1] = self.bounds['lb_dropout']  # lower bound za dropout
        self.ub[1] = self.bounds['ub_dropout']  # upper bound za dropout

        self.lb[2] = self.bounds['lb_epochs']  # lower bound za number of epochs, int
        self.ub[2] = self.bounds['ub_epochs']  # upper bound za number of epochs, int

        self.lb[3] = self.bounds['lb_layers_cnn']  # lower bound za broj CNN slojeva, int
        self.ub[3] = self.bounds['ub_layers_cnn']  # upper bound za broj CNN slojeva, int

        self.lb[4] = self.bounds['lb_layers_dense']  # lower bound za broj dense slojeva, int
        self.ub[4] = self.bounds['ub_layers_dense']  # upper bound za broj dense slojeva, int

        # sada sve do kraja ovog niza postavljamo vrednost za donju i gornju granicu broja neurona po svim slojevima, ukljucujuci i attention layer
        # pa zato ide do kraja, a duzina D zavisi od ub_layers, pa ako je ub_layers npr. 2, onda je duzina niza D=4 (osnovna 4 parametra) + 2*2 = 8

        for i in range(5, D):  # za D dodamo 1, posto je poslednja vrednost broj neurona u dense sloju
            self.lb[i] = self.bounds['lb_nn']  # lower bound za broj neurona po slojevima, int
            self.ub[i] = self.bounds['ub_nn']  # upper bound za broj neurona po slojevima, int, poslednji je za attention layer

        # ostalo su za pocetak fiksni parametri
        #self.loss = 'categorical_crossentropy'
        if self.no_classes==2:
            self.loss = 'categorical_crossentropy'
        else:
            self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        self.epochs = 200
        self.dropout = 0.01
        self.activation = 'relu'
        #self.activation = 'tanh'
        self.learning_rate = 0.001

        '''
        Parametri za optimizaciju:
           broj cnn slojeva, broj neurona, dropout, learning rate, broj epoha
           idu cnn layeri sa maxpooling, posle na kraju ide dropout, flatten, dense
           i na kraju output (dense) sa softmax funkcijom.
       '''

    def function(self, x):
        # list asa brojem neurona
        # swarm jedinka je vec lista i onda ne treba nista menjati
        # uzecemo samo prvi element liste u slucaju da imamo vise parametara
        # print(type(x))
        # print(x)
        # konvertujemo ponovo u listu, jer sa vise hiperparametara se komplikuje

        #self.test_data.reset()

        print('Solution:',x)

        learning_rate = x[0]
        dropout = x[1]  # ovo je za dropout layer
        epochs = int(x[2])  # epoch je isto integer
        layers_cnn = int(x[3])  # broj cnn layera
        layers_dense = int(x[4])  # broj dense layera
        # sad pravimo listu neurona, za svaki LSTM layer po jedan neuron plus jedan na kraju broj neurona za attention layer
        # idemo od 5-tog ideksa pa do kraja koliko imamo layers za CNN i dense
        nn_list_cnn = x[5:(layers_cnn + 5)]
        # konvertujemo sve u int
        nn_list_cnn = [int(x) for x in nn_list_cnn]
        nn_list_dense = x[(5+layers_cnn):(5+layers_cnn+layers_dense)]

        nn_list_dense = [int(x) for x in nn_list_dense]

        #print('Layers con', layers_cnn)
        #print('Layers dense', layers_dense)
        #print('CNN nn list:', nn_list_cnn)
        #print('Dense nn list:', nn_list_dense)

        # kreiranje CNN modela, treniranje i predikcija
        CNNModel = self.createCNNModel(layers_cnn,layers_dense,nn_list_cnn,nn_list_dense,learning_rate,dropout)

        self.trainCNN(CNNModel,epochs)



        objective, error, y_proba, y = self.model_predict(CNNModel)

        print('MCC:', (1-objective), ' Error:', error)

        return (objective, error, y_proba, y, self.feature_size, CNNModel)

        '''
        # sada predvidjamo i uzimamo rezultata, ovo je sa prob distribution

        results = CNNModel.predict(self.X_test) #,steps = self.test_data.n/self.batch_size
        # sada konvertujemo u klase
        print('results',results)
        results_predicted = np.argmax(results, axis=1)

        #print('Results',results)


        # print(results)
        # print(len(results))
        # print("Duzina labela:", len(self.labels))
        # print("Duzina predicted rezultata", len(results_predicted))

        print('y_test',self.y_test)
        print('results predicted',results_predicted)

        acc = accuracy_score(self.y_test,results_predicted) #ovo daje lose rezultate
        #evaluation = CNNModel.evaluate_generator(self.test_data)
        #acc = evaluation[1] #ovo je accuracy i daje dobre rezultate, ali ne vraca probabilities

        print("Results predicted:",results_predicted)

        # sada uzimamo error
        # print('Accuracy',acc)
        error = round(1 - acc, 30)
        # print("ERROR",error)
        # print("Results pred:",results_predicted)
        # print("Result pred len:",len(results_predicted))

        # sada racunamo cohen kappa score

        print('Acc:',acc)
        print('Err:',error)

        cohen_kappa = cohen_kappa_score(results_predicted, self.test_data.classes)

        # sada vracamo rezultate
        return error, cohen_kappa, results, results_predicted, self.features, CNNModel

        # ((1 - cohen_kappa), error, y_proba, y, feature_size, xgb_clf)
    '''

    def createCNNModel(self, layers_cnn, layers_dense, nn_list_cnn, nn_list_dense, learning_rate, dropout):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        CNNModel = keras.models.Sequential()

        for i in range(len(nn_list_cnn)):
            if i == 0:  # ako je prvi, dodajemo input shape
                CNNModel.add(Conv1D(nn_list_cnn[i], kernel_size=6, input_shape=(self.X_train.shape[1], 1),
                                    activation=self.activation))
                CNNModel.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))

            CNNModel.add(Conv1D(nn_list_cnn[i], kernel_size=6, activation=self.activation))
            CNNModel.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
        CNNModel.add(Flatten())
        for i in range(len(nn_list_dense)):
            CNNModel.add(Dense(nn_list_dense[i], activation=self.activation))
            # CNNModel.add(BatchNormalization())
            CNNModel.add(Dropout(dropout))

            # sada dodajemo poslednji moel sa aktivacionom funkcijom

        CNNModel.add(Dense(self.no_classes))
        CNNModel.add(Activation('softmax'))

        CNNModel.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)

        return CNNModel







    '''
    # Pomocna funkcija za kreiranje CNN modela
    def createCNNModel(self, layers_cnn, layers_dense, nn_list_cnn, nn_list_dense, learning_rate, dropout):

        #optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        # optimzer = 'adam'

        CNNModel = keras.models.Sequential()

        # sada idemo kroz listu neurona
        for i in range(len(nn_list_cnn)):
            # poslednji sloj nema return_sequences, svi ostali imaju
            # u prvom sloju samo dajemo batch size, u ostalim ne dajemo

            if i == 0:  # ako je prvi, dodajemo input shape
                #CNNModel.add(Conv2D(nn_list_cnn[i], (3,3), input_shape=(self.target_size[0], self.target_size[1],3),
                if(self.color_mode=='rgb'):
                 CNNModel.add(Conv2D(nn_list_cnn[i], (3, 3), input_shape=(self.target_size[0], self.target_size[1],3),
                             activation=self.activation))  # 3 na kraju za rgb slik
                 CNNModel.add(MaxPooling2D())
                else:
                    CNNModel.add(
                        Conv2D(nn_list_cnn[i], (3, 3), input_shape=(self.target_size[0], self.target_size[1], 1),
                               activation=self.activation))  # 3 na kraju za rgb slik
                    CNNModel.add(MaxPooling2D())


            CNNModel.add(Conv2D(nn_list_cnn[i],(3,3),activation=self.activation))
            CNNModel.add(MaxPooling2D())

        CNNModel.add(Flatten())

        for i in range(len(nn_list_dense)):
            CNNModel.add(Dense(nn_list_dense[i], activation=self.activation))
            #CNNModel.add(BatchNormalization())
            CNNModel.add(Dropout(dropout))

        # sada dodajemo poslednji moel sa aktivacionom funkcijom

        CNNModel.add(Dense(self.no_classes))
        CNNModel.add(Activation('softmax'))

        CNNModel.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)

        return CNNModel
    '''
    '''
    def trainCNN(self, model, epochs):
        if (self.early_stop):
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=False, patience=epochs / 3)
            model.fit_generator(generator=self.train_data,epochs=epochs,validation_data=self.test_data,callbacks=[es])

        else:
            model.fit_generator(generator=self.train_data, epochs=epochs,validation_data=self.test_data)
            #steps_per_epoch=self.train_data.n / self.batch_size,validation_steps=self.test_data.n/self.batch_size)
'''

    def trainCNN(self, model, epochs):
        if (self.early_stop):
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=False, patience=epochs / 3)
            model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=self.batch_size,
          validation_data=(self.X_test, self.y_test),callbacks=[es],verbose=0)
        else:
            model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=self.batch_size,
                      validation_data=(self.X_test, self.y_test),verbose=0)

    #pomocna funkcija za predvidjanje
    def model_predict(self,model):
        #self.model.fit(self.x_train, self.y_train, batch_size=self.BATCH_SIZE, epochs=self.NB_EPOCHS, verbose=0)
        # predivdja probabilities za sve klase
        y_proba = model.predict(self.X_test,verbose=0)
        correct = 0
        total = y_proba.shape[0]
        for i in range(total):
            predicted = np.argmax(y_proba[i])
            test = np.argmax(self.y_test[i])
            correct = correct + (1 if predicted == test else 0)
        # print('Accuracy: {:f}'.format(correct / total))
        y = np.zeros((len(y_proba), y_proba.shape[1]))
        for i in range(len(y_proba)):
            y[i][np.argmax(y_proba[i])] = 1
        # classification error
        error = np.round(1 - (correct / total), 30)

        y_cappa = np.zeros(len(y_proba))
        for i in range(total):
            y_cappa[i] = np.argmax(y_proba[i])

        # racunamo cohen kappa za statistiku, ovo je indikator
        cohen_kappa = cohen_kappa_score(y_cappa, self.y_test1)

        mcc = matthews_corrcoef(self.y_test1, y_cappa)




        return ((1-mcc),error, y_proba, y)