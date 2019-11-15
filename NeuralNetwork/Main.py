from Input import Dataset
import Input
from Regression import Regression
from Regression import LogisticRegression
import Preprocessing
import Validation
from NeuralNetwork import NeuralNetwork
import pandas as pd


if __name__ == '__main__':

    # ----- DATASET -----

    file = "Dataset/candy.csv"
    data = Dataset(file)

    #matrix = [[1,2,3,1,0,0],[2,3,5,1,0,0],[-5,6,-8,0,1,0],[-1,-1,-3,0,0,1]]
    #matrix = [[1,2,3,1,0,0]]

    #matrix = pd.DataFrame(matrix)

    # ----- FEATURES -----

    X_indeces = [0,1,2,3,4,5,6,7,8,9]
    #X_indeces = [0,1,2]

    # ----- TARGET -----

    Y_indeces = [10, 11, 12]
    #Y_indeces = [3,4,5]

    # ----- PREPROCESSING -----

    #Preprocessing.show_statistics(data.dataset, X_indeces)

    #outlier_removal = [5,6]
    #data.dataset = Preprocessing.outlier_removal_quartiles(data.dataset, outlier_removal)


    # ----- Z-SCORE  -----

    normalization = Preprocessing.dataset_z_score(data, X_indeces)
    means = normalization[1]
    std_devs = normalization[2]

    # ----- NORMALIZED DATASET -----

    norm_file = file.replace(".csv", "") + "_ZSCORED.csv"
    data = Dataset(norm_file)


    # ----- SPLIT DATASET AND TEST SET -----

    sets = Preprocessing.split_sets(data.dataset, 0.2)
    dataset = sets[0]
    testset = sets[1]

    newsets = Preprocessing.split_sets(dataset, 0.125)

    trainingset = newsets[0]
    validationset = newsets[1]

    # ----- DATA FROM DATASET  -----

    X = Input.giveme_cols(data.dataset, X_indeces)
    Y = Input.giveme_cols(data.dataset, Y_indeces)


    trainingX = Input.giveme_cols(trainingset, X_indeces)
    trainingY = Input.giveme_cols(trainingset, Y_indeces)

    validationX = Input.giveme_cols(validationset, X_indeces)
    validationY = Input.giveme_cols(validationset, Y_indeces)

    testX = Input.giveme_cols(testset, X_indeces)
    testY = Input.giveme_cols(testset, Y_indeces)

    hidden_layers_neurons = [5,5]
    gianluca_neurons = 2

    alfa = 0.25
    _lambda = 0.0000000001

    model = NeuralNetwork(X,Y,hidden_layers_neurons, alfa, _lambda)
    model.backward_propagation()
    model.print_cost()

    for it in range(1000):

        model.backward_propagation()
        model.print_cost()

    #x = [1,2,3]
    candy = [1, 0, 0, 0, 1, 0, 0, 0.186, 0.26699999, 41.904308] # u dolc
    choco = [0, 0, 0, 1, 0, 0, 1, 0.87199998, 0.84799999, 49.524113] #ciucculat
    sergio1 = [0, 0, 0, 0, 0, 0, 1, 0.31299999, 0.31299999, 44.375519] #dolce
    sergio2 = [1, 0, 0, 0, 1, 0, 0, 0.186, 0.26699999, 41.904308] #dolce
    sergio3 = [0, 0, 0, 1, 0, 0, 1, 0.87199998, 0.84799999, 49.524113] #cioccolato
    prediction = sergio3
    prediction = Preprocessing.zscore_norm_prediction(prediction, means, std_devs)
    solution = model.forward_propagation(prediction)[-1]

    print('----- SOLUTION -----')
    print('- W -\n',solution,'\n')
    for sol in solution:
        if sol > 0.5:
            print('1')
        else:
            print('0')