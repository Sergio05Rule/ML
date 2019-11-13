from Input import Dataset
import Input
from Regression import Regression
import Preprocessing


if __name__ == '__main__':

    # ----- DATASET -----

    file = "Dataset/wine.csv"
    data = Dataset(file)

    # ----- FEATURES -----

    X_indeces = [0,1,2,3,4,5,6,7,8,9,10]

    # ----- TARGET -----

    Y_index = [11]

    # ----- PREPROCESSING -----

    Preprocessing.show_statistics(data.dataset, X_indeces)

    outlier_removal = [5,6]
    data.dataset = Preprocessing.outlier_removal_quartiles(data.dataset, outlier_removal)

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

    #print(trainingset)
    #print(testset)
    #print(validationset)
    #input()

    # ----- DATA FROM DATASET  -----

    X = Input.giveme_cols(trainingset,X_indeces)
    Y = Input.giveme_cols(trainingset, Y_index)

    # ----- MODEL -----

    alfa = 0.1
    _lambda = 0
    iterations = 1000

    model = Regression(X, Y)
    thetas = model.batch_gradient_descent(alfa, _lambda, iterations)

    # ---------- PRINT OF THE SOLUTION ----------

    model.print_solution(thetas)
    

    print('\n\n')

    trainingX = Input.giveme_cols(trainingset, X_indeces)
    trainingY = Input.giveme_cols(trainingset, Y_index)

    validationX = Input.giveme_cols(validationset, X_indeces)
    validationY = Input.giveme_cols(validationset, Y_index)

    lambdas = [0,1,3,5,7,10]
    training_errors = list()
    validation_errors = list()

    for _lambda_ in lambdas:

        alfa = 0.1
        iterations = 1000

        training = Regression(trainingX, trainingY)
        thetas = training.batch_gradient_descent(alfa,_lambda_,iterations)
        training_errors.append(training.cost_function(_lambda_))

        validation = Regression(validationX,validationY)
        validation.thetas = thetas
        validation_errors.append(validation.cost_function(_lambda_))


    print('CHOSEN LAMBDA', lambdas)
    print('TRAINING ERRORS',training_errors)
    print('VALIDATION ERRORS',validation_errors)


    '''
    # ---------- VALUE TO PREDICT ----------

    predict = [6.0, 0.31, 0.47, 3.6, 0.067, 18.0, 42.0, 0.99549, 3.39, 0.66, 11.0]

    #predict = [0,0,0,1,0,0,1,0.87199998, 0.84799999, 49.524113]
    #predict = [0,0,0,0,0,0,1,0.73199999,0.51099998,36.017628]

    # ---------- VALUE NORMALIZATION (eventuale) ----------

    predict = Preprocessing.zscore_norm_prediction(predict, means, std_devs)

    # ---------- PRINT PREDICTION ----------

    solution = model.predict(predict)
    print('SOLUZIONE', solution)
    '''
