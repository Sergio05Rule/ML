import CSVManager as CSVM
import Input as IN
import Models


# ---------- PRENDI IN INPUT IL TRAINING SET ----------
file1 = 'Dataset/wine.csv'
csv = CSVM.CSVManager(file1)


# ---------- INDICA L'INDICE DEL TARGET ----------
target_index = 11


# ---------- INDICA GLI INDICI DELLE FEATURE ----------
features_indeces = [0,1,2,3,4,5,6,7,8,9,10]
#features_indeces = [1]

# ---------- INDICA LE FEATURE MULTICLASSE----------
'''
NOTA: vanno inserite ciascuna come lista degli indici che le compongono.
Tutte queste liste vanno inseite all'interno della multiclass_feature
Ad esempio: [ [ 1, 3, 4] , [2, 2] , [3 , 3, 1 ] ]
'''
multiclass_feature = []


# ---------- INPUT STATICO NOTI GLI INDICI----------

variables_indeces = [target_index, features_indeces, multiclass_feature]
classification_values = ['5','6','7']             # VALORE DA CLASSIFICARE


# ---------- INPUT DINAMICO ----------

#variables = IN.InputTargetFeature(csv)


# ---------- MODELLO ----------

model = Models.MulticlassLogisticRegression(csv, variables_indeces, classification_values)


# ---------- NORMALIZZAZIONE ----------

model.Normalization(IN.normalizationControl())


# ---------- FITTING ----------
alfa = 0.01
iterations = 100
b = 100

model.fit(alfa, iterations, b)
model.print_solution()

# ---------- VALORE DA PREDIRE ----------

#predict = [6.0, 0.31, 0.47, 3.6, 0.067, 18.0, 42.0, 0.99549, 3.39, 0.66, 11.0]
#predict = [7.8, 0.88, 0, 2.6, 98, 25.0, 67.0, 0.9968, 3.2, 0.68, 9.8]
#predict = [8.1,0.38,0.28,2.1,0.066,13.0,30.0,0.9968,3.23,0.73,9.7]  #7
predict = [10.6,0.34,0.49,3.2,0.078,20.0,78.0,0.9992,3.19,0.7,10.0] #6

#predict = [0.31]
# ---------- PREDIZIONE ----------

model.prediction(predict)


















'''
# ---------- MODELLO ----------

#model = REG.Regression(values)          # REGRESSIONE
model = LG.LogisticRegression(values)  # REGRESSIONE LOGISTICA


# ---------- NORMALIZZAZIONE ----------

norm = IN.normalizationControl()

if norm == 'Y':
    
    for index, x in enumerate(model.X):
        if index != 0:
            print('NORMALIZZAZIONE X',index)
            model.X[index] = PRE.zscore_norm(x)
    
    
# ---------- LANCIA IL MODELLO ----------
print('FIRST ERROR = ', model.CostFunction())
thetas = model.batchGD(0.6, 100)

# ---------- STAMPA RISULTATO DELLA REGRESSIONE ----------
print('SOLUTION THETAS: ', thetas)

if len(classification_value) > 1:
    
    for value in classification_value[1::]:
        
        print('REGRESSIONE LOGISTICA MULTICLASSE PER IL VALORE ', value, '\n')
        

        # ---------- ESTRAPOLA DATI ----------

        values = csv.giveme_values_indeces(variables_indeces, value)


        # ---------- MODELLO ----------

        model = LG.LogisticRegression(values)  # REGRESSIONE LOGISTICA
        # model = LG.LogisticRegression(values)  # REGRESSIONE LOGISTICA


        # ---------- NORMALIZZAZIONE ----------

        if norm == 'Y':

            for index, x in enumerate(model.X):
                if index != 0:
                    print('NORMALIZZAZIONE X', index)
                    model.X[index] = PRE.zscore_norm(x)

        # ---------- LANCIA IL MODELLO ----------
        print('FIRST ERROR = ', model.CostFunction())
        thetas = model.batchGD(0.6, 100)

        # ----------
         STAMPA RISULTATO DELLA REGRESSIONE ----------
        print('SOLUTION THETAS: ', thetas)


# ---------- VALORE DA PREDIRE ----------

predict = [6.0, 0.31, 0.47, 3.6, 0.067, 18.0, 42.0, 0.99549, 3.39, 0.66, 11.0]

# ---------- ZSCORE INPUT (eventuale) ----------
predict = model.solution_zscore(predict)
predict = np.insert(predict, 0, 1)

# ---------- STAMPA PREDIZIONE ----------

print('PREDICTED VALUE FOR INSERT FEATURES = ', model.predict_M(predict))
#print('PROBABILITY = ', model.LogisticFunction(predict))

values = csv.giveme_values_indeces(variables_indeces, '6')
classification = LG.LogisticRegression(values)
'''






'''
if len(variables_indeces[1]) == 1 and len(variables_indeces[2])==0:

    PP.print_graph(regression.Y, regression.X[1])
    PP.Polynomial(thetas, (min(regression.X[1])*0.5), (max(regression.X[1])*1.5))
    PP.show()
'''
