import statistics
import pandas as pd
from Input import Dataset

def average(features):
    return sum(features) / len(features)

def variance(features):

    avg = sum(features) / len(features)
    sum_of_diff = 0

    for feature in features:
        temp = (feature - avg) #(x-mean)
        temp = pow(temp,2) #(x-mean)^2
        sum_of_diff += temp

    variance = sum_of_diff / len(features) # [(x-mean)^2] / n
    return variance

def show_statistics(dataset, indeces):

    for index in indeces:

        column = dataset[index]
        column = column.tolist()
        print('\n--- FEATURE WITH INDEX ',index,' ---\n')
        print('avarage = ',average(column))
        print('variance = ',variance(column))
        print('min value = ',min(column))
        print('max value = ',max(column))


# ----- NORMALIZATION -----

def min_max_norm(features, b, a):

    _max = max(features)
    _min = min(features)

    new_features = []

    for feature in features:
        temp = ( (feature - _min) / ( _max - _min) ) * (b-a) + a
        new_features.append(temp)

    return [new_features, _min, _max, b, a]

def standard_deviation(features):

    dev_std = pow(variance(features) , 0.5) #radquad(variance)
    return dev_std

def zscore_norm(features):
    mean = average(features)
    std_dev = standard_deviation(features)

    new_features = []

    for feature in features:
        temp = (feature - mean) / std_dev
        new_features.append(temp)
    return [new_features, mean, std_dev]

def zscore_norm_prediction(features, mean, dev_std):
    new_features = []

    for index, feature in enumerate(features):
        temp = (feature - mean[index]) / dev_std[index]
        new_features.append(temp)
    return new_features

def dataset_z_score(data: Dataset, x):

    dataset = data.dataset
    means = list()
    standard_deviations = list()
    new_dataset = list()

    for column in range(dataset.shape[1]):
        if column in x:
            normalization = zscore_norm(dataset[column])
            new_column = normalization[0]
            means.append(normalization[1])
            standard_deviations.append(normalization[2])
            new_dataset.append(new_column)
        else:
            new_dataset.append(dataset[column])

    # ----- WRITING NORMALIZED DATASET -----
    new_dataset: pd.DataFrame = pd.DataFrame(new_dataset).transpose()
    new_path: str = data.file_path.replace(".csv", "") + "_ZSCORED.csv"
    new_dataset.to_csv(new_path, sep=',', quotechar='"', encoding='utf8', header=None, index=False)
    new_data = Dataset(new_path)
    return [new_data, means, standard_deviations]

def dataset_min_max(data: Dataset, x):

    dataset = data.dataset
    _mins = list()
    _maxs = list()
    b = 0
    a = 0
    new_dataset = list()

    for column in range(dataset.shape[1]):
        if column in x:
            normalization = zscore_norm(dataset[column])
            new_column = normalization[0]
            _mins.append(normalization[1])
            _maxs.append(normalization[2])
            b = normalization[3]
            a = normalization[4]
            new_dataset.append(new_column)
        else:
            new_dataset.append(dataset[column])

    # ----- WRITING NORMALIZED DATASET -----

    new_dataset: pd.DataFrame = pd.DataFrame(new_dataset).transpose()
    new_path: str = data.file_path.replace(".csv", "") + "_MIN_MAX.csv"
    new_dataset.to_csv(new_path, sep=',', quotechar='"', encoding='utf8', header=None, index=False)
    new_data = Dataset(new_path)
    return [new_data, _mins, _maxs, b, a]

#TODO metodo per normalizzare in min max il valore che si vuole predire ( come è stato fatto con zscore_norm_prediction


# ----- OUTLIER REMOVAL -----

def quartiles_boundaries(data):

    start_index = round(len(data)/4)
    end_index = round(len(data)*3/4)

    return (start_index, end_index)

def outlier_removal_quartiles(dataset, indeces):

    boundaries = quartiles_boundaries(dataset)
    new_dataset = dataset.copy()
    to_remove = list()

    for column in indeces:

        col = dataset[column]
        col = col.sort_values(ascending=True)
        indeces = col.index
        a = list()

        for index in indeces[0:boundaries[0]]:

            if index not in to_remove:
                to_remove.append(index)

        for index in indeces[boundaries[1]:]:

            if index not in to_remove:
                to_remove.append(index)

    new_dataset = new_dataset.drop(to_remove)
    return new_dataset


# ----- TRAINING SET / TEST SET -----

def split_sets(dataset, test_set_percentage):

    rows = len(dataset)
    return (dataset.iloc[round(rows*test_set_percentage):],dataset.iloc[:round(rows*test_set_percentage)])
