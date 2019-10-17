import csv
import CSVManager as CSVM
import Input as IN
import RegressionModule as REG
import LogisticRegression as LG
import PlotPrint as PP
import Preprocessing as PRE
import numpy as np

# ---------- PRENDI IN INPUT IL TRAINING SET ----------

file1 = 'monitoraggiotempidiattesa.csv'
file2 = 'mydata.csv'
file3 = 'Dataset/wine.csv'
csv = CSVM.CSVManager(file3)

# ---------- INDICA LA VARIABILE DA PREDIRE ----------


# ---------- INPUT STATICO ----------
#variables1 = ['PRENOTAZIONI_DAGARANTIRE',['PRENOTAZIONI'],[['PRENOTAZIONI','PRENOTAZIONI'],['PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI'],['PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI'], ['PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI'],['PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI','PRENOTAZIONI']]]
#variables2 = ['PRENOTAZIONI_DAGARANTIRE',['PRENOTAZIONI'],[]]
#variables3 = ['Y',['A','B','C','D','E','F','G','H','I','J','K'],[]]

#variables = variables3

variables_indeces = [11, [0,1,2,3,4,5,6,7,8,9,10],[]]

# ---------- INPUT DINAMICO ----------
#variables = IN.InputTargetFeature(csv)

# ---------- ESTRAPOLA DATI ----------

values = csv.giveme_values_indeces(variables_indeces, '7')

# ---------- LANCIA REGRESSIONE ----------

if len(variables_indeces[1]) == 1 and len(variables_indeces[2])==0:
    print('\n\nREGRESSIONE LINEARE UNIVARIATA\n\n')
elif len(variables_indeces[2])>0:
    print('\n\nREGRESSIONE POLINOMIALE\n\n')
elif len(variables_indeces[1]) > 1  and len(variables_indeces[2])==0:
    print('\n\nREGRESSIONE LINEARE MULTIVARIATA\n\n')

#regression = REG.Regression(values)
classification = LG.LogisticRegression(values)


for x in values[1]:
    print(x)
# ---------- NORMALIZZAZIONE ----------

for index, x in enumerate(classification.X):
    if index != 0:
        print('NORMALIZZAZIONE X',index)
        classification.X[index] = PRE.zscore_norm(x)

# ---------- INIZIO REGRESSIONE ----------
print('FIRST ERROR = ', classification.CostFunction())
thetas = classification.batchGD(0.6, 100)

# ---------- STAMPA RISULTATO DELLA REGRESSIONE ----------
print('SOLUTION THETAS: ', thetas)

# ---------- VALORE DA PREDIRE ----------

#predict = [6.0, 0.31, 0.47, 3.6, 0.067, 18.0, 42.0, 0.99549, 3.39, 0.66, 11.0]
#predict = [6.0, 0.31]
predict = [7.8, 0.88, 0, 2.6, 98, 25.0, 67.0, 0.9968, 3.2, 0.68, 9.8]

# ---------- ZSCORE INPUT (eventuale) ----------
predict = classification.solution_zscore(predict)
predict = np.insert(predict, 0, 1)

# ---------- STAMPA PREDIZIONE ----------

print('PREDICTED VALUE FOR INSERT FEATURES = ', classification.predict_M(predict))
print(classification.LogisticFunction(predict))

values = csv.giveme_values_indeces(variables_indeces, '6')
classification = LG.LogisticRegression(values)

# ---------- NORMALIZZAZIONE ----------

for index, x in enumerate(classification.X):
    if index != 0:
        print('NORMALIZZAZIONE X',index)
        classification.X[index] = PRE.zscore_norm(x)

thetas = classification.batchGD(0.6, 100)

print('PREDICTED VALUE FOR INSERT FEATURES = ', classification.predict_M(predict))
print(classification.LogisticFunction(predict))

values = csv.giveme_values_indeces(variables_indeces, '5')
classification = LG.LogisticRegression(values)

# ---------- NORMALIZZAZIONE ----------

for index, x in enumerate(classification.X):
    if index != 0:
        print('NORMALIZZAZIONE X',index)
        classification.X[index] = PRE.zscore_norm(x)

thetas = classification.batchGD(0.6, 100)

print('PREDICTED VALUE FOR INSERT FEATURES = ', classification.predict_M(predict))
print(classification.LogisticFunction(predict))

# ---------- GRAFICO STAMPATO SOLO IN CASO DI UNIVARIATA REGRESSIONE ----------
'''
if len(variables_indeces[1]) == 1 and len(variables_indeces[2])==0:

    PP.print_graph(regression.Y, regression.X[1])
    PP.Polynomial(thetas, (min(regression.X[1])*0.5), (max(regression.X[1])*1.5))
    PP.show()
'''
