class Univariate:

    def __init__(self, observations):
        self.observations = observations
        self.Y = list()
        self.X = list()

        for observation in observations[0]:
            new_Y = int(observation)
            self.Y.append(new_Y)

        for observation in observations[1]:
            new_X = []
            for x in observation:
                new_x = int(x)
                new_X.append(new_x)
            self.X.append(new_X)


        self.THETAS = list()
        for _ in self.X:
            self.THETAS.append(2)



    def VerboseMeanSquaredError(self):

        error = 0

        for row, observation in enumerate(self.Y):
            hypotesis = 0
            print('\n\nOBSERVATION NUMBER = ', row)
            print('OBSERVATION', observation)

            for col, teta in enumerate(self.THETAS):
                print('\nINDEX: ', col)

                feature = self.X[col][row]
                print('teta * feature[index]', '\nteta',col,'=', teta, '\nfeature: ', self.X[col][row], '\nindex: ',row, '\nX[row][col]: ',feature)
                hypotesis += (teta * feature)
                print('ipotesi= ', hypotesis)

            print('\nsingle error = (ipotesi - y)^2', '\nhypotesis: ', hypotesis, '\nobservation: ', observation)
            single_error = (hypotesis - observation) ** 2
            print('single error = ', single_error)
            error += single_error
        print('\n\n ERROR = ', error)
        error = error / (2 * len(self.Y))
        print('\n\n ERROR / 2m = ', error, 'with m = ', len(self.Y))


        return error

    def MeanSquaredError(self):

        error = 0

        for row, observation in enumerate(self.Y):
            hypotesis = 0

            for col, teta in enumerate(self.THETAS):

                feature = self.X[col][row]
                hypotesis += (teta * feature)

            single_error = (hypotesis - observation) ** 2
            error += single_error

        error = error / (2 * len(self.Y))
        print('\n\n ERROR / 2m = ', error, 'with m = ', len(self.Y))


        return error

    def verbose_prediction_error_row(self, row):

        error = 0

        # ---------- h (xi) : predizione sulla data riga ----------
        for index, theta in enumerate(self.THETAS):

            error += theta * self.X[index][row]

            print('theta', index, ' * x', index)
            print('theta', index,': ', theta, ' * x', index,' : ',  self.X[index][row])
            print('estimation increment = ',error)


        # ----------  h * xi - yi : errore di predizione sulla data riga ----------
        error -= self.Y[row]

        return error

    def prediction_error_row(self, row):

        error = 0

        # ---------- h (xi) : predizione sulla data riga ----------
        for index, theta in enumerate(self.THETAS):

            error += theta * self.X[index][row]

        # ----------  h * xi - yi : errore di predizione sulla data riga ----------
        error -= self.Y[row]

        return error


    def verbose_j_gradient_row(self, row):          # restituisce una lista dei gradienti di J calcolati per ogni teta

        gradient = 0
        update = list()

        prediction_error = self.prediction_error_row(row)

        print('PREDICTION ERROR: ', prediction_error)

        # ---------- update i = gradient * xi ----------
        for index, X in enumerate(self.X):

            gradient = prediction_error * X[row]

            print('GRADIENT: ', gradient, 'prediction_error: ', prediction_error, 'X',index,'[row]: ',X[row])

            update.append(gradient)

        return update

    def j_gradient_row(self, row):  # restituisce una lista dei gradienti di J calcolati per ogni teta

        gradient = 0
        update = list()

        prediction_error = self.prediction_error_row(row)

        # ---------- update i = gradient * xi ----------
        for index, X in enumerate(self.X):
            gradient = prediction_error * X[row]

            update.append(gradient)

        return update

    def verbose_j_gradient(self):

        gradient = list()

        # ---------- inizializzo a zero il valore dell'gradient per ogni teta ----------

        for _ in range(len(self.THETAS)):

            gradient.append(0)

        for row, _ in enumerate(self.Y):            # per tutte le righe del dataset

            print('ROW: ', row)
            print('Gradient before computing: ', gradient)

            update = self.j_gradient_row(row)

            print('Gradient increment: ', update)

            gradient = [x + y for x, y in zip(gradient, update)]  # somma tra elementi corrispondenti di liste di interi

            print('Gradient after row',row, ' = ', gradient)

        return gradient


    def j_gradient(self):

        gradient = list()

        # ---------- inizializzo a zero il valore dell'gradient per ogni teta ----------

        for _ in range(len(self.THETAS)):

            gradient.append(0)

        for row, _ in enumerate(self.Y):            # per tutte le righe del dataset

            update = self.j_gradient_row(row)

            gradient = [x + y for x, y in zip(gradient, update)]  # somma tra elementi corrispondenti di liste di interi

        return gradient



    def verbose_new_thetas(self, alfa, rows = None ):           # rows è il numero di righe del dataset su cui siamo lavorando, di default è tutto il dataset

        if rows == None:
            rows = len(self.Y)
        theta_new = list()

        print('verbose_new_thetas\nalfa = ', alfa, 'rows', rows)
        update = self.j_gradient()
        print('UPDATE = ', update)

        for index, theta in enumerate(self.THETAS):

            print('NEW THETAS ROW: ', index)

            print('update[index]', update[index])

            _theta_new = alfa * update[index] / rows

            print('alfa * update[index] / rows', _theta_new)

            _theta_new = theta - _theta_new

            print('_theta_new', _theta_new)

            theta_new.append(_theta_new)

        return theta_new


    def new_thetas(self, alfa, rows = None ):           # rows è il numero di righe del dataset su cui siamo lavorando, di default è tutto il dataset

        if rows == None:
            rows = len(self.Y)
        theta_new = list()
        update = self.j_gradient()

        for index, theta in enumerate(self.THETAS):

            _theta_new = alfa * update[index] / rows

            _theta_new = theta - _theta_new

            theta_new.append(_theta_new)

        return theta_new

    def verbose_batchGD(self, alfa, iterations, rows = None):

        if rows == None:
            rows = len(self.Y)

        print('INITIAL J = ', self.MeanSquaredError())

        for _ in range(iterations):

            print('BATCH GD, ITERATION: ', _)

            new_thetas = self.new_thetas(alfa, rows)
            print('New THETAS = ', new_thetas)
            self.THETAS = new_thetas

            print('\nJ after iteration ', _ ,' = ', self.MeanSquaredError(), '\n')
