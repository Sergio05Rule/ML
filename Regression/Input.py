

def InputTargetFeature(csv):            # PRENDE IN INPUT UN OGGETTO CSVMANAGER e restituisce una coppia (nome target, [nome feature1, nome feature2, nome feature3, ...]

    attributi = csv.giveme_attributes()
    print('\nDATASET ATTRIBUTES:')
    print(attributi)

    # ---------- INDICA LA VARIABILE DA PREDIRE ----------
    target = ''
    features = list()
    feature = ''

    while (attributi.count(target) < 1):
        target = input('Inserisci la variabile TARGET\n')

    while feature != 'exit':
        feature = ''

        while (attributi.count(feature) < 1 and feature != 'exit'):
            feature = input('Inserisci la variabile FEATURE\n')

        if features.count(feature) > 0:
            print(features)
            feature = input('Feature già inserita\n')

        elif feature != 'exit':           # Inserisci feature se non è la sequenza di uscita 'exit' e se non è già presente
            features.append(feature)

    print('Variabile target inserita: ', target)
    print('Feature inserite: ', features)
    return [target, features]