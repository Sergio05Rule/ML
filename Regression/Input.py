import sys


def InputTargetFeature(csv):            # PRENDE IN INPUT UN OGGETTO CSVMANAGER e restituisce una coppia (nome target, [nome feature1, nome feature2, nome feature3, ...]

    attributi = csv.giveme_attributes()
    print('\nDATASET ATTRIBUTES:')
    print(attributi)

    # ---------- INDICA LA VARIABILE DA PREDIRE ----------
    target = ''
    features = list()
    feature = ''

    # ---------- INSERIMENTO VARIABILE TARGET ----------
    while (attributi.count(target) < 1):
        target = input('Inserisci la variabile TARGET (type \'exit\' to quit)\n')
        if attributi.count(target) < 1 and target != 'exit':
            print('Variabile non presente')
        elif target == 'exit':
            sys.exit()
    print('Target ', target, 'inserito\n')

    # ---------- INSERIMENTO VARIABILI FEATURE ----------
    while feature != 'run' and feature != 'exit':
        feature = ''

        while (attributi.count(feature) < 1 and feature != 'run' and feature != 'exit'):
            feature = input('Inserisci la variabile FEATURE (type \'run\' to run the algorithm - \'exit\' to quit)\n')

            if attributi.count(feature) < 1 and feature != 'run' and feature != 'exit':
                print('Variabile non presente')

            if features.count(feature) > 0:
                print(features)
                feature = input('Feature già inserita\n')

        if feature == 'exit':  # Inserisci feature se non è la sequenza di uscita 'exit' e se non è già presente
            sys.exit()

        if feature == target:
            print('ATTENZIONE: la feature specificata è uguale alla variabile target.')
            flag = ''
            while (flag != 'Y' and flag != 'N'):
                flag = input('Confermare? (Y/N)\n')
                if flag == 'Y':
                    features.append(feature)
                    print('Feature ', feature , 'inserita')
        elif feature != 'run' and feature != 'exit':
            print('Feature ', feature, 'inserita')
            features.append(feature)

    print('Variabile target inserita: ', target)
    print('Feature inserite: ', features)
    return [target, features]