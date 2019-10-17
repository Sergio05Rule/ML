import csv

class CSVManager:


    def __init__(self, file_name):

        self.file_name = file_name


    # ---------- RESTITUISCE LA LISTA DEGLI ATTRIBUTI DEL FILE CSV ----------
    def giveme_attributes(self, file = None):

        attributi = list()

        if file == None:

            file_name = self.file_name

        else:

            file_name = file

        with open(file_name, 'r', encoding='utf-8-sig') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')

            header = next(csv_reader)

            for id in header:

                attributi.append(id)

        return attributi


    # ---------- RESTITUISCE UNA LISTA FORMATA DAL VETTORE Y E DALLA MATRICE X: OUTPUT = [Y, X] ----------
    def giveme_values_dynamic(self, variables_list, classification_target = None):

        # variables_list è la lista dei nomi delle variabili. Il primo è la variabile target, i successivi le variabli feature
        attributi = self.giveme_attributes(self.file_name)

        # ---------- ID DELLA VARIABILE TARGET E LISTA DEGLI ID DELLE FEATURE ----------
        target = variables_list[0]
        features = variables_list[1]
        polinomials = variables_list[2]

        features_index = list()
        target_index = None
        polinomials_indeces = list()

        X = list()         # è la matrice X dei valori delle feature
        Y = list()         # è il vettore Y dei valori della feature
        Z = list()

        output = list()

        # ---------- SETTO LA MATRICE X ----------
        for _ in range(len(features) + 1):          # IL +1 SERVE A CONSIDERARE LA FEATURE FITTIZIA x0

            X.append([])

        # ---------- SETTO LA MATRICE Z ----------
        for _ in range(len(polinomials)):

            Z.append([])


        # ---------- SETTO L'INDICE DELLA VAR TARGET E GLI INDICI DELLE FEATURE ----------
        target_index = attributi.index(target)

        for var in features:

            # ---------- CREA UNA LISTA DI INDICI RIFERITI ALLE FEATURE NEL DATASET ----------
            features_index.append(attributi.index(var))         # lista del tipo features_index = [ indice di x1, indice di x2, indice di x3, ... ]

        for poli in polinomials:

            new_list = list()

            for term in poli:

                new_list.append(attributi.index(term))
            polinomials_indeces.append(new_list)


        print('INDICI: \nTARGET = ',target_index,'\nFEATURE = ', features_index, '\nVARIABILI POLINOMIALI: ', polinomials_indeces)

        # ---------- PRELEVA DAL DATASET PULITO I VALORI DI Y E X ----------
        with open(self.file_name, 'r', encoding='utf-8-sig') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            print('FILE = ', self.file_name)
            next(csv_reader)

            for line in csv_reader:
                value = float(line[target_index])

                if classification_target != None:
                    Y.append(value)
                else:
                    if value == classification_target:
                        Y.append(1)
                    else:
                        Y.append(0)

                for index in range(len(X)):

                    if index == 0:
                        X[0].append(1)
                    else:
                        value = float (line[features_index[index-1]])
                        X[index].append(value)

                for index, poli in enumerate(polinomials_indeces):

                    value = 1

                    for factor in poli:

                        value *= float(line[factor])
                    Z[index].append(value)

        for z in Z:
            X.append(z)
        output.append(Y)
        output.append(X)
        return output

        # ---------- RESTITUISCE UNA LISTA FORMATA DAL VETTORE Y E DALLA MATRICE X: OUTPUT = [Y, X] ----------
    def giveme_values_indeces(self, indeces, classification_target=None):

        # variables_list è la lista dei nomi delle variabili. Il primo è la variabile target, i successivi le variabli feature
        features_index = indeces[1]
        target_index = indeces[0]
        polinomials_indeces = indeces[2]

        X = list()  # è la matrice X dei valori delle feature
        Y = list()  # è il vettore Y dei valori della feature
        Z = list()

        output = list()

        # ---------- SETTO LA MATRICE X ----------
        for _ in range(len(features_index) + 1):  # IL +1 SERVE A CONSIDERARE LA FEATURE FITTIZIA x0

            X.append([])

        # ---------- SETTO LA MATRICE Z ----------
        for _ in range(len(polinomials_indeces)):
            Z.append([])

        print('INDICI: \nTARGET = ', target_index, '\nFEATURE = ', features_index, '\nVARIABILI POLINOMIALI: ',
              polinomials_indeces)

        # ---------- PRELEVA DAL DATASET I VALORI DI Y E X ----------
        with open(self.file_name, 'r', encoding='utf-8-sig') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            print('FILE = ', self.file_name)
            next(csv_reader)

            for line in csv_reader:
                value = line[target_index]

                if classification_target == None:
                    value = float(line[target_index])
                    Y.append(value)
                else:
                    if value == classification_target:
                        Y.append(1)
                    else:
                        Y.append(0)

                for index in range(len(X)):

                    if index == 0:
                        X[0].append(1)
                    else:
                        value = float(line[features_index[index - 1]])
                        X[index].append(value)

                for index, poli in enumerate(polinomials_indeces):

                    value = 1

                    for factor in poli:
                        value *= float(line[factor])
                    Z[index].append(value)

        for z in Z:
            X.append(z)
        output.append(Y)
        output.append(X)
        return output


    def clean_empty_id(self, id_tocheck = None):

        cleaned_csv = list()
        id_index = -1

        with open(self.file_name, 'r', encoding='utf-8-sig') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            attributi = self.giveme_attributes()

            # ---------- PRENDI IL NOME DELL'ATTRIBUTO DA CONTROLLARE SE NON È STATO PASSATO PER ARGOMENTO ----------

            if id_tocheck == None:

                while (attributi.count(id_tocheck) < 1):

                    id_tocheck = input('Inserisci l\'attributo che non può essere nullo\n')

                    if id_tocheck == 'exit':
                        return 'EXIT'

            #---------- TROVO L'INDICE DELL'ATTRIBUTO ----------
            for index, id in enumerate(attributi):

                if id == id_tocheck:
                    id_index = index

            for line in csv_reader:

                try:
                    if line[id_index] != '':
                        cleaned_csv.append(line)
                except:
                    print('NO NO MAMACITA')

        # ---------- SCRIVE IL NUOVO FILE SENZA ELEMENTI NULLI PER L'ATTRIBUTO SCELTO ----------
        new_file_name = 'Dataset' + '.csv'
        with open(new_file_name, 'w') as csv_file:

            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"')
            csv_writer.writerows(cleaned_csv)

        print('Dataset cleaned from lines with empty attribute: ', id_tocheck)
        self.file_name = new_file_name
        return new_file_name


    def clean_variables(self, variables_list):

        indeces = []
        attributi = self.giveme_attributes()

        target = variables_list[0]
        features = variables_list[1]


        # --------- PULIZIA DEL DATASET DAI VALORI ASSENTI ----------

        file = self.clean_empty_id(target)
        for feature in features:
            file = self.clean_empty_id(feature)

        # --------- PRELEVO GLI INDICI DI TARGET E FEATURE ----------

        indeces.append(attributi.index(target))

        for index, var in enumerate(features):

            indeces.append(attributi.index(var))

        with open(file, 'r', encoding='utf-8-sig') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            result = list()

            for line in csv_reader:
                new_line_element = []
                new_line = []
                flag_firstelement = 0

                for index, attribute in enumerate(line):
                    if indeces.count(index) > 0:    # se l'indice è presente nella lista degli indici da selezionare
                        if flag_firstelement == 1:
                            string = ','

                        new_line.append(line[index])

                        flag_firstelement = 1
                new_line_element.append(new_line)

                result.append(new_line_element)


        # ---------- SCRIVE IL NUOVO FILE SENZA ELEMENTI NULLI PER L'ATTRIBUTO SCELTO ----------
        new_file_name = 'Dataset'+ '.csv'
        with open(new_file_name, 'w') as csv_file:

            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"')
            for line in result:
                csv_writer.writerows(line)

        self.file_name = new_file_name
        return new_file_name

    def clean_variables_static(self, variable_indeces):

        indeces = []

        # --------- PRELEVO GLI INDICI DI TARGET E FEATURE ----------

        indeces.append(variable_indeces[0])

        for index, _ in enumerate(variable_indeces[1]):

            indeces.append(_)

        with open(self.file_name, 'r', encoding='utf-8-sig') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            result = list()

            for line in csv_reader:
                new_line_element = []
                new_line = []
                flag_firstelement = 0

                for index, attribute in enumerate(line):
                    if indeces.count(index) > 0:    # se l'indice è presente nella lista degli indici da selezionare
                        if flag_firstelement == 1:
                            string = ','

                        new_line.append(line[index])

                        flag_firstelement = 1
                new_line_element.append(new_line)

                result.append(new_line_element)


        # ---------- SCRIVE IL NUOVO FILE SENZA ELEMENTI NULLI PER L'ATTRIBUTO SCELTO ----------
        new_file_name = 'Dataset' + '.csv'
        with open(new_file_name, 'w') as csv_file:

            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"')
            for line in result:
                csv_writer.writerows(line)

        self.file_name = new_file_name
        return new_file_name