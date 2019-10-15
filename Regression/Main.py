import matplotlib.pyplot as plt
import csv
import CSVManager as CSVM
import Input as IN
import UnivariateLinearRegression as ULR


# ---------- PRENDI IN INPUT IL TRAINING SET ----------

file1 = 'monitoraggiotempidiattesa.csv'
file2 = 'mydata.csv'
csv = CSVM.CSVManager(file2)

# ---------- INDICA LA VARIABILE DA PREDIRE ----------

variables = IN.InputTargetFeature(csv)
print('Voglio predire: ', variables[0], 'con le variabili: ', variables[1] )

# ---------- ESTRAPOLA DATI ----------

values = csv.giveme_values(variables)

# ---------- LANCIA REGRESSIONE ----------

if len(values[1]) == 2:
    #print('REGRESSIONE UNIVARIATA')
    regression = ULR.Univariate(values)
    #print('error = ', regression.MeanSquaredError())

    #print(regression.j_gradient())
    print(regression.verbose_batchGD(0.05, 1000))


    '''for index, _ in enumerate(regression.THETAS):
        regression.THETAS[index] = 1

    for cicle in range(10):

        new_theta = regression.batch_GD(0.1)
        regression.THETAS = new_theta
        regression.MeanSquaredError()
        print('THETAS = ', new_theta, ' ', regression.THETAS)
    '''
    #regression.batch_GD()

elif len(values[1]) > 2:
    print('REGRESSIONE MULTIVARIATA')
    regression = ULR.Univariate(values)
    print('error = ', regression.MeanSquaredError())

    while _ in range(10):

        for index, _ in enumerate(regression.THETAS):

            regression.THETAS[index] = 1

        new_theta = regression.batch_GD(1)
        regression.THETAS = new_theta
        print('error = ', regression.MeanSquaredError())

# ---------- STAMPA RISULTATO DELLA REGRESSIONE ----------


