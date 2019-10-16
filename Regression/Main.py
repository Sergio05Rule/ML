import csv
import CSVManager as CSVM
import Input as IN
import UnivariateLinearRegression as ULR
import PlotPrint as PP

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

    thetas = regression.miniBatchGD(0.000001, 10000, 100)
    print('Thetas = ', thetas)
    PP.print_graph(regression.Y, regression.X[1])
    PP.print_line(thetas[0], thetas[1])
    PP.show()


elif len(values[1]) > 2:
    print('REGRESSIONE MULTIVARIATA')


# ---------- STAMPA RISULTATO DELLA REGRESSIONE ----------


