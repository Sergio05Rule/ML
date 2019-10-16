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
    def giveme_values(self, variables_list):

        # variables_list è la lista dei nomi delle variabili. Il primo è la variabile target, i successivi le variabli feature

        new_file_name = self.clean_variables(variables_list)
        attributi = self.giveme_attributes(new_file_name)

        # ---------- ID DELLA VARIABILE TARGET E LISTA DEGLI ID DELLE FEATURE ----------
        target = variables_list[0]
        features = variables_list[1]

        features_index = list()
        target_index = None

        X = list()         # è la matrice X dei valori delle feature
        Y = list()         # è il vettore Y dei valori della feature

        output = list()

        # ---------- SETTO LA MATRICE X ----------
        for _ in range(len(features) + 1):          # IL +1 SERVE A CONSIDERARE LA PRIMA RIGA DI 1

            X.append([])

        # ---------- SETTO L'INDICE DELLA VAR TARGET E GLI INDICI DELLE FEATURE ----------
        target_index = attributi.index(target)

        for var in features:

            # ---------- CREA UNA LISTA DI INDICI RIFERITI ALLE FEATURE NEL DATASET ----------
            features_index.append(attributi.index(var))         # lista del tipo features_index = [ indice di x1, indice di x2, indice di x3, ... ]

        print('Indici: \nTARGET: ',target_index,'\nFEATURE: ', features_index)


        # ---------- PRELEVA DAL DATASET PULITO I VALORI DI Y E X ----------
        with open(new_file_name, 'r', encoding='utf-8-sig') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            print('FILE = ', new_file_name)
            next(csv_reader)

            for line in csv_reader:

                Y.append(line[target_index])

                for index in range(len(X)):

                    if index == 0:
                        X[0].append(1)
                    else:
                        X[index].append(line[features_index[index-1]])

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