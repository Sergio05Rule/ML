from Input import Dataset
import Input
from Regression import Regression
from Regression import LogisticRegression
import Preprocessing
import Validation


if __name__ == '__main__':

    # ----- DATASET -----

    file = "Dataset/candy.csv"
    data = Dataset(file)

    # ----- FEATURES -----

    X_indeces = [0,1,2,3,4,5,6,7,8,9]

    # ----- TARGET -----

    Y_index = [10]

    # ----- PREPROCESSING -----

    Preprocessing.show_statistics(data.dataset, X_indeces)

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

    trainingX = Input.giveme_cols(data.dataset, X_indeces)
    trainingY = Input.giveme_cols(data.dataset, Y_index)

    validationX = Input.giveme_cols(validationset, X_indeces)
    validationY = Input.giveme_cols(validationset, Y_index)

    testX = Input.giveme_cols(testset, X_indeces)
    testY = Input.giveme_cols(testset, Y_index)

    # ----- MODEL -----

    alfa = 0.25
    _lambda = 0
    iterations = 1000

    #Validation.learning_curves_values(dataset, X_indeces, Y_index)

    training = LogisticRegression(trainingX,trainingY)
    validation = LogisticRegression(validationX,validationY)
    test = LogisticRegression(testX,testY)

    #training.value_to_predict(5)
    #validation.value_to_predict(5)
    #test.value_to_predict(5)

    test.thetas = training.batch_gradient_descent(alfa,_lambda,iterations)

    test.confusion_indeces()


    '''
    # ---------- PRINT OF THE SOLUTION ----------

    model.print_solution(thetas)

    # ---------- MODELS VALIDATION ----------

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