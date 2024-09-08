import numpy as np
from utilities.fs_utils import transfer_function
from utilities.ht_utils import *

#ovo je dodatno i za feature selection, jer treba transfer funckija u startu
#solutions feature selection hyperparams optimization

class Solution:
    def __init__(self, function,tf, nfeatures, x=None):
        if x is None:


            self.function = function
            self.x = [None] * self.function.D  
            self.objective_function = None
            self.fitness = None
            self.trial = 0
            self.tf = tf
            self.nfeatures = nfeatures
            self.error = None
            self.feature_size = None
            self.model = None

            self.initialize()
            #ranije je bilo
            #self.y_proba = 0
            #self.y = 0
            #ovde dodajemo za no classes i broj instanci u y_test
            #duzina nizova y_proba i y je   broj redova len(y_test), tj. broj instanci u y_test i broj kolona broj klasa:num_classes
            self.y_proba = np.empty([function.y_test_length,function.no_classes])
            self.y = np.empty([function.y_test_length,function.no_classes])

            # ovo je pomocni atribut, cuvamo sve objectives i druge indikatore za population diversity, cuvamo samo u najboljem resenju u svakoj iteraciji
            self.diversity = None

        else:
            self.x = x
            self.function = function
            self.calculate_objective_function()
            self.calculate_fitness()
            self.trial = 0
            self.tf = tf
            self.nfeatures = nfeatures

            # ovo je pomocni atribut, cuvamo sve objectives i druge indikatore za population diversity, cuvamo samo u najboljem resenju u svakoj iteraciji
            #self.diversity = None
      

    def initialize(self):        
        for i in range(0, len(self.x)):
            rnd = np.random.rand()
            self.x[i] = rnd * (self.function.ub[i] - self.function.lb[i]) + self.function.lb[i]

        self.x = np.array(self.x)
        self.x[0:self.nfeatures] = transfer_function(self.x[0:self.nfeatures],self.tf,self.nfeatures)

        #konvertujemo sve elemente x u integer koji treba da budu integer

        self.x = convertToInt(self.x,self.function.intParams)

        #proveravamo da li imamo integer parametre
        #for j in range(len(self.x)):
         #   if j in self.function.intParams:
          #      self.x[j] = np.rint(self.x[j])



        self.x = self.x.tolist()

        self.calculate_objective_function()
        self.calculate_fitness()
        
    #ovde je objective function ili error ili error u kombinaciji sa featrues
    #a error je classification error
    def calculate_objective_function(self):
        #returns objective_function (error), y_proba and y as dummy (1,0,0)
        #(self.objective_function,self.y_proba,self.y) = self.function.function(self.x)
        self.objective_function,self.error,self.y_proba,self.y,self.feature_size, self.model = self.function.function(self.x)
        #print(self.objective_function)
        self.fitness = 1 / (1 + abs(self.objective_function))

    def calculate_fitness(self):
        self.fitness = 1 / (1 + abs(self.objective_function))

    def __str__(self):
        return "Error: " + str(self.error) + "\n"  + "Objective: " + str(self.objective_function)


        

