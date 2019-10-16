import csv
import CSVManager as CSVM
import Input as IN
import RegressionModule as REG
import PlotPrint as PP
import Preprocessing as PRE

# ---------- PRENDI IN INPUT IL TRAINING SET ----------

file1 = 'monitoraggiotempidiattesa.csv'
file2 = 'mydata.csv'
csv = CSVM.CSVManager(file2)

# ---------- INDICA LA VARIABILE DA PREDIRE ----------


# ---------- INPUT STATICO ----------
variables1 = ['PRENOTAZIONI_DAGARANTIRE',['PRENOTAZIONI'],[['PRENOTAZIONI','PRENOTAZIONI'],['PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI'],['PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI'], ['PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI'],['PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI']]]
variables2 = ['PRENOTAZIONI_DAGARANTIRE',['PRENOTAZIONI'],[]]

#variables = variables2

# ---------- INPUT DINAMICO ----------
variables = IN.InputTargetFeature(csv)

# ---------- ESTRAPOLA DATI ----------

values = csv.giveme_values(variables)

# ---------- LANCIA REGRESSIONE ----------


if len(variables[1]) == 1 and len(variables[2])==0:
    print('REGRESSIONE LINEARE UNIVARIATA')
elif len(variables[2])>0:
    print('REGRESSIONE POLINOMIALE')
elif len(variables[1]) > 1  and len(variables[2])==0:
    print('REGRESSIONE LINEARE MULTIVARIATA')

regression = REG.Univariate(values)

# ---------- NORMALIZZAZIONE ----------
print('NORMALIZZAZIONE Y')
regression.Y = PRE.zscore_norm(regression.Y)

for index, x in enumerate(regression.X):
    if index != 0:
        print('NORMALIZZAZIONE X',index)
        regression.X[index] = PRE.zscore_norm(x)

# ---------- INIZIO REGRESSIONE ----------

thetas = regression.batchGD(0.001, 100000)

# ---------- STAMPA RISULTATO DELLA REGRESSIONE ----------

print('Thetas = ', thetas)

print(regression.predict_Znorm(5))

# ---------- GRAFICO STAMPATO SOLO IN CASO DI UNIVARIATA REGRESSIONE ----------

if len(variables[1]) == 1 and len(variables[2])==0:

    PP.print_graph(regression.Y, regression.X[1])
    PP.Polynomial(thetas, (min(regression.X[1])*0.5), (max(regression.X[1])*1.5))
    PP.show()

