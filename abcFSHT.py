from utilities.solutionFSHT import Solution
from utilities.fs_utils import transfer_function
from utilities.ht_utils import *
import random
import math
import copy
import numpy as np

class ABCAlgorithm:
    def __init__(self, n, function, tf,nfeatures):
        self.N = n
        self.function = function
        self.size = math.ceil(n / 2)
       # self.limit = self.size * self.function.D
        self.population = []
        self.best_solution = [None] * self.function.D
        self.MR = 0.8
        self.FFE = self.N
        self.tf = tf;  # ovo je transfer function za transformaciju u binary space
        self.nfeatures = nfeatures;  # ovo je max broj features u datasetu, prvih nfeatures parametara su za features, ostali su za hiper-param optimiation
        
    def initial_population(self):
        for i in range(0, self.N):
            local_solution = Solution(self.function,self.tf,self.nfeatures)
            self.population.append(local_solution)
        self.population.sort(key=lambda x: x.objective_function)
        self.best_solution = copy.deepcopy(self.population[0].x)

    def print_global_best(self):
        self.best_solution.sort(key=lambda x: x.objective_function)
        #print('BEST SOLUTION: fitness: {}, obj: {}'.format(self.best_solution[0].fitness, self.best_solution[0].objective_function))

    def update_position(self, t, max_iter):
        lb = self.function.lb
        ub = self.function.ub
        ######################### Employed bee phase #########################
        for i in range(self.size): 
               
            random_partner_index = random.choice(range(len(self.population)))
            # Sve dok su indeksi isti, nadji drugi
            while i == random_partner_index:
                random_partner_index = random.choice(range(len(self.population)))

            fi = np.random.uniform(-1, 1)
            # randomly selected food source
            x_rand = self.population[random_partner_index].x
            # current food source
            x_curr = np.array(copy.deepcopy(self.population[i].x))
            # new food source
            #dodato
            x_new = np.array(copy.deepcopy(x_curr))
            for j in range(len(x_curr)):
                theta = random.uniform(0, 1)
                if(theta<self.MR):
                    fi = np.random.uniform(-1, 1)
                    x_new[j] = x_curr[j] + fi * (x_curr[j] - x_rand[j])
                else:
                    x_new[j] = x_curr[j]

            # must convert to numpy array and check boundaries and apply transfer function
            x_new = np.array(x_new)
            x_new[0:self.nfeatures] = transfer_function(x_new[0:self.nfeatures], self.tf,
                                                        self.nfeatures)  # ovde koristimo transfer funckiju za prvih nfeatures parametara
            # konvertujemo sve elmente koji treba da budu integer u integer
            x_new = convertToInt(x_new, self.function.intParams)
            x_new = self.checkBoundaries(x_new)

            # Ne mora ovde da se racuna fitness i obj jer je to u konstruktoru
            self.FFE = self.FFE + 1
            solution = Solution(self.function, self.tf, self.nfeatures, x_new.tolist())

            if solution.objective_function < self.population[i].objective_function:
                self.population[i] = solution
            else:
                self.population[i].trial += 1

        ######################### Onlooker bee phase #########################
        
        for i in range(self.size, self.N):
            r = random.random()
            prob = 0.9 * (self.population[i].objective_function / max(self.population, key=lambda x: x.objective_function).objective_function) + 0.1

            if r < prob:

                random_partner_index = random.choice(range(len(self.population)))
                # Sve dok su indeksi isti, nadji drugi
                while i == random_partner_index:
                    random_partner_index = random.choice(range(len(self.population)))         
                    
                fi = np.random.uniform(-1, 1)
                # randomly selected food source
                x_rand = self.population[random_partner_index].x
                # current food source
                x_curr = np.array(copy.deepcopy(self.population[i].x))
                # new food source
                x_new = x_curr + fi * (x_curr - x_rand)



                # must convert to numpy array and check boundaries and apply transfer function
                x_new = np.array(x_new)
                x_new[0:self.nfeatures] = transfer_function(x_new[0:self.nfeatures], self.tf,
                                                           self.nfeatures)  # ovde koristimo transfer funckiju za prvih nfeatures parametara
                # konvertujemo sve elmente koji treba da budu integer u integer
                x_new = convertToInt(x_new, self.function.intParams)
                x_new = self.checkBoundaries(x_new)
                        
                # Ne mora ovde da se racuna fitness i obj jer je to u konstruktoru
                self.FFE = self.FFE + 1
                solution = Solution(self.function, self.tf, self.nfeatures, x_new.tolist())

                if solution.objective_function < self.population[i].objective_function:
                    self.population[i] = solution
                else:
                    self.population[i].trial += 1
            
        ######################### Scout phase #########################

        self.population.sort(key=lambda x: x.objective_function)
        self.best_solution = self.population[0].x
        limit = int(max_iter / self.N)
        for i in range(1,self.N):
            if self.population[i].trial > limit:
                self.FFE = self.FFE + 1
                solution = Solution(self.function,self.tf,self.nfeatures)
                self.population[i] = solution
                
                
    def sort_population(self):

        self.population.sort(key=lambda x: x.objective_function)
        self.best_solution = self.population[0].x

    def get_global_best(self):
        return (self.population[0].objective_function, self.population[0].error, self.population[0].y_proba,
                self.population[0].y, self.population[0].feature_size,self.population[0].model)
    
    def get_global_worst(self):
        return self.population[-1].objective_function
    
    def optimum(self):
        print('f(x*) = ', self.function.minimum, 'at x* = ', self.function.solution)
        
    def algorithm(self):
        return 'ABC'
    
    def objective(self):
        
        result = []
        
        for i in range(self.N):
            result.append(self.population[i].objective_function)
            
        return result
    
    def average_result(self):
        return np.mean(np.array(self.objective()))
    
    def std_result(self):        
        return np.std(np.array(self.objective()))
    
    def median_result(self):
        return np.median(np.array(self.objective()))
        
       
    def print_global_parameters(self):
            for i in range(0, len(self.best_solution)):
                 print('X: {}'.format(self.best_solution[i]))
                 
    def get_best_solutions(self):
        return np.array(self.best_solution)

    def get_solutions(self):
        
        sol = np.zeros((self.N, self.function.D))
        for i in range(len(self.population)):
            sol[i] = np.array(self.population[i].x)
        return sol


    def print_all_solutions(self):
        print("******all solutions objectives**********")
        for i in range(0,len(self.population)):
              print('solution {}'.format(i))
              print('objective:{}'.format(self.population[i].objective_function))
              print('solution {}: '.format(self.population[i].x))
              print('--------------------------------------')

    def get_global_best_params(self):
        return self.population[0].x

    def getFFE(self):
        #metoda za prikaz broja FFE
        return self.FFE

    def checkBoundaries(self, Xnew):
        for j in range(self.nfeatures, self.function.D):
            if Xnew[j] < self.function.lb[j]:
                Xnew[j] = self.function.lb[j]

            elif Xnew[j] > self.function.ub[j]:
                Xnew[j] = self.function.ub[j]
        return Xnew

    # funkcija koja vraca najbolji global_best_solution
    def get_global_best_solution(self):
        # ovde pravimo liste sa objective i indicator za celu populaciji

        indicator_list = []  # ovo je indikator, sta god da je u pitanju
        objective_list = []  # ovo je objective, sta god da je u pitanju
        objective_indicator_list = []
        for i in range(len(self.population)):
            indicator_list.append(self.population[
                                      i].error)  # ovo je za error, mada je to bilo koji drugi indikator, samo se tako zoeve
            objective_list.append(self.population[i].objective_function)  # ovo je objective
        objective_indicator_list.append(objective_list)
        objective_indicator_list.append(indicator_list)
        self.population[0].diversity = objective_indicator_list

        return self.population[0]