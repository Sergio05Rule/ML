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

model = Models.LogisticRegression(csv, variables_indeces, classification_values)


# ---------- NORMALIZZAZIONE ----------

model.Normalization(IN.normalizationControl())


# ---------- FITTING ----------
alfa = 0.6
iterations = 100
b = 100
_lambda = 0

model.fit(alfa, iterations, _lambda, b)
model.print_solution()

# ---------- VALORE DA PREDIRE ----------

predict = [6.0, 0.31, 0.47, 3.6, 0.067, 18.0, 42.0, 0.99549, 3.39, 0.66, 11.0]
#predict = [7.8, 0.88, 0, 2.6, 98, 25.0, 67.0, 0.9968, 3.2, 0.68, 9.8]
#predict = [8.1,0.38,0.28,2.1,0.066,13.0,30.0,0.9968,3.23,0.73,9.7]  #7
#predict = [10.6,0.34,0.49,3.2,0.078,20.0,78.0,0.9992,3.19,0.7,10.0] #6
#predict = [10.6,0.34,0.49,3.2,0.078,20.0,78.0,0.9992,3.19,0.7,10.0, 0.34**2, 0.34**3]
#predict = [0.31]

# ---------- PREDIZIONE ----------

model.prediction(predict)
