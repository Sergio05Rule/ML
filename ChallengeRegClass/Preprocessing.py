import statistics

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

def min_max_norm(features):
    print('--- Min/Max Normalization ---')
    b = int(input('Inserisci l intervallo dei valori in cui vuoi che ricada la X normalizzata\nb = '))
    a = int(input('a = '))

    _max = max(features)
    _min = min(features)

    new_features = []

    for feature in features:
        temp = ( (feature - _min) / ( _max - _min) ) * (b-a) + a
        new_features.append(temp)

    return new_features

def standard_deviation(features):

    dev_std = pow(variance(features) , 0.5) #radquad(variance)
    return dev_std

def zscore_norm(features):
    print('--- ZScore Normalization ---')
    mean = average(features)
    dev_std = standard_deviation(features)
    print('mean = ', mean)
    print('dev = ', dev_std)

    new_features = []

    for feature in features:
        temp = (feature - mean) / dev_std
        new_features.append(temp)


    return new_features
