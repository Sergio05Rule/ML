import csv
import CSVManager as CSVM
import Input as IN
import UnivariateLinearRegression as ULR


# ---------- PRENDI IN INPUT IL TRAINING SET ----------

file1 = 'monitoraggiotempidiattesa.csv'
file2 = 'mydata.csv'
csv = CSVM.CSVManager(file1)

# ---------- INDICA LA VARIABILE DA PREDIRE ----------

variables = IN.InputTargetFeature(csv)
print('Voglio predire: ', variables[0], 'con le variabili: ', variables[1] )

# ---------- ESTRAPOLA DATI ----------

values = csv.giveme_values(variables)

# ---------- LANCIA REGRESSIONE ----------

if len(values[1]) == 2:
    #print('REGRESSIONE UNIVARIATA')
    regression = ULR.Univariate(values)
    print(regression.stochasticGD(0.0001, 10000))


elif len(values[1]) > 2:
    print('REGRESSIONE MULTIVARIATA')


# ---------- STAMPA RISULTATO DELLA REGRESSIONE ----------


