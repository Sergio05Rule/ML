import csv
import CSVManager as CSVM
import Input as IN
import RegressionModule as REG
import PlotPrint as PP
import Preprocessing as PRE

# ---------- PRENDI IN INPUT IL TRAINING SET ----------

file1 = 'monitoraggiotempidiattesa.csv'
file2 = 'mydata.csv'
file3 = 'Dataset/wine_MOD.csv'
csv = CSVM.CSVManager(file3)

# ---------- INDICA LA VARIABILE DA PREDIRE ----------


# ---------- INPUT STATICO ----------
#variables1 = ['PRENOTAZIONI_DAGARANTIRE',['PRENOTAZIONI'],[['PRENOTAZIONI','PRENOTAZIONI'],['PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI'],['PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI'], ['PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI'],['PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI']]]
#variables2 = ['PRENOTAZIONI_DAGARANTIRE',['PRENOTAZIONI'],[]]
#variables3 = ['Y',['A','B','C','D','E','F','G','H','I','J','K',],[]]

#variables = variables3

# ---------- INPUT DINAMICO ----------
variables = IN.InputTargetFeature(csv)

# ---------- ESTRAPOLA DATI ----------

#values = csv.giveme_values(variables3)

# ---------- LANCIA REGRESSIONE ----------


if len(variables[1]) == 1 and len(variables[2])==0:
    print('\n\nREGRESSIONE LINEARE UNIVARIATA\n\n')
elif len(variables[2])>0:
    print('\n\nREGRESSIONE POLINOMIALE\n\n')
elif len(variables[1]) > 1  and len(variables[2])==0:
    print('\n\nREGRESSIONE LINEARE MULTIVARIATA\n\n')

regression = REG.Regression(values)

print(regression.THETAS)
#regression.MeanSquaredError_M()
#regression.MeanSquaredError()

# ---------- NORMALIZZAZIONE ----------

print('NORMALIZZAZIONE Y')
regression.Y = PRE.zscore_norm(regression.Y)

for index, x in enumerate(regression.X):
    if index != 0:
        print('NORMALIZZAZIONE X',index)
        regression.X[index] = PRE.zscore_norm(x)

# ---------- INIZIO REGRESSIONE ----------
print('FIRST ERROR = ', regression.MeanSquaredError())
thetas = regression.batchGD(0.6, 1000)

# ---------- STAMPA RISULTATO DELLA REGRESSIONE ----------

print(thetas)

# ---------- ESEMPIO DI PREDIZIONE ----------

#print(regression.predict_Znorm([100, 1000, 10000, 100000, 1000000, 10000000]))  # regressione polinomiale di grado 6
#print(regression.predict_Znorm([100]))  # regressione univariata


# ---------- GRAFICO STAMPATO SOLO IN CASO DI UNIVARIATA REGRESSIONE ----------
'''
if len(variables[1]) == 1 and len(variables[2])==0:

    PP.print_graph(regression.Y, regression.X[1])
    PP.Polynomial(thetas, (min(regression.X[1])*0.5), (max(regression.X[1])*1.5))
    PP.show()
'''
