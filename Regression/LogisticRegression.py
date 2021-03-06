import random
from math import e
import math
import Preprocessing as PRE
import numpy as np

class LogisticRegression:

    def __init__(self, observations):
        self.observations = observations
        self.Y = observations[0]
        self.X = observations[1]
        self.THETAS = list()
        for _ in self.X:
            self.THETAS.append(random.randint(-1,1))
        self.pre_processedX = self.X.copy()


    def prediction(self, X):

        prediction = 0

        for index, theta in enumerate(self.THETAS):

            # ---------- THETA*X ----------
            prediction += theta * X[index]

        return prediction

    def verbose_LogisticFunction(self, X):

        num = 1
        den = 1 + e ** (-(self.prediction(X)))
        print(-self.prediction(X))
        print(e ** (-(self.prediction(X))))
        return num / den

    def LogisticFunction(self, X):

        num = 1
        den = 1 + e ** (-(self.prediction(X)))
        return num / den

    def RowX(self, row):

        X = list()

        for x in self.X:
            X.append(x[row])

        return X


    def Probability_row(self, row):            # preleva i valori di x per la specifica riga

        cost = 0
        X = self.RowX(row)
        y = self.Y[row]

        if y == 1:
            cost = self.LogisticFunction(X)
        if y == 0:
            cost = 1 - self.LogisticFunction(X)

        return cost


    def Probability(self):

        cost = 1

        for row, y in enumerate(self.Y):

            cost *= self.Probability_row(row)

        return cost


    def CostFunction_row(self, row):

        y = self.Y[row]
        cost = 0

        if y == 1:
            cost = math.log(self.Probability_row(row))

        elif y == 0:
            cost = 1 - math.log(self.Probability_row(row))
        return cost

    def verbose_CostFunction_row(self, row):

        y = self.Y[row]
        cost = 0

        if y == 1:
            print('Y = ', y)
            cost = math.log(self.Probability_row(row))

        elif y == 0:
            print('error', row, self.Probability_row(row))
            cost = 1 - math.log(self.Probability_row(row))
        print('cost at line ', row, ' = ', cost)
        return cost

    def CostFunction(self):

        cost = 0

        for row in range(len(self.Y)):

            cost += self.CostFunction_row(row)

        cost = -cost / len(self.Y)

        return cost

    def verbose_prediction_error_row(self, row):
        print('---- prediction error row ----')
        print('ROW: ', row)
        X = self.RowX(row)
        print('X = ', X)
        logistic = self.LogisticFunction(X)
        print('Logistic = ', logistic)
        error = logistic - self.Y[row]

        return error

    def prediction_error_row(self, row):

        X = self.RowX(row)
        logistic = self.LogisticFunction(X)
        error = logistic - self.Y[row]

        return error


    def verbose_j_gradient_row(self, row):  # restituisce una lista dei gradienti di J calcolati per ogni teta

        print('---- J GRADIENT ROW ----')
        print('ROW: ', row)
        gradient = 0
        update = list()

        prediction_error = self.prediction_error_row(row)
        print('prediction_error = ',prediction_error)

        # ---------- update i = gradient * xi ----------
        for index, X in enumerate(self.X):
            gradient = prediction_error * X[row]
            print('X[row]', X[row])
            print('gradient contribute = ', prediction_error * X[row])
            update.append(gradient)

        return update

    def j_gradient_row(self, row):  # restituisce una lista dei gradienti di J calcolati per ogni teta

        update = list()

        prediction_error = self.prediction_error_row(row)

        # ---------- update i = gradient * xi ----------
        for index, X in enumerate(self.X):
            gradient = prediction_error * X[row]
            update.append(gradient)

        return update

    def j_gradient(self, start_row = 0, end_row = None):

        if end_row == None:

            end_row = len(self.Y)

        gradient = list()

        # ---------- inizializzo a zero il valore dell'gradient per ogni teta ----------

        for _ in range(len(self.THETAS)):

            gradient.append(0)

        for row in range(start_row, end_row):            # per tutte le righe del dataset da start row a end row

            update = self.j_gradient_row(row)

            gradient = [x + y for x, y in zip(gradient, update)]  # somma tra elementi corrispondenti di liste di interi

        return gradient


    def new_thetas(self, alfa, start_row = 0, end_row = None):           # rows è il numero di righe del dataset su cui siamo lavorando, di default è tutto il dataset

        if end_row == None:
            end_row = len(self.Y)

        theta_new = list()
        rows = end_row - start_row

        update = self.j_gradient(start_row, end_row)

        for index, theta in enumerate(self.THETAS):

            _theta_new = alfa * update[index] / rows
            _theta_new = theta - _theta_new

            theta_new.append(_theta_new)

        return theta_new

    def batchGD(self, alfa, iterations):

        for _ in range(iterations):
            new_thetas = self.new_thetas(alfa)
            self.THETAS = new_thetas

        #print('Cost Function after batch = ', self.CostFunction())

        return self.THETAS

    def predict_M(self, X):

        return np.dot(X,self.THETAS)


    def solution_zscore(self, solution):

        new_solution = list()
        new_x = 0
        for index, x in enumerate(solution):
            new_x = 0
            new_x = (x - PRE.average(self.pre_processedX[index+1])) / (PRE.standard_deviation(self.pre_processedX[index+1]))
            new_solution.append(new_x)

        return new_solution


input = ([ [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0], [ [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [60,65,70, 75,80,85,90, 95, 100, 67, 55, 50, 19 ,30, 20, 40, 23, 54, 33, 25]  ]  ])

#input = [ [1,1,0,0], [[1,1,1,1], [45,60,20,10] ] ]

classification = LogisticRegression(input)
print('Thetas = ',classification.THETAS)
