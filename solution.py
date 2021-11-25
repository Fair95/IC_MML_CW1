import numpy as np

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10 * X**2) + 0.1 * np.sin(100 * X)


class LinearRegression:
    def __init__(self, basis='polynomial', J=1):
        self.J = J
        self.basis = basis

        self.mle_w = None
        self.mle_variance = None

    def design_matrix(self, X):
        if self.basis == 'polynomial':
            return self._polynomial_design_matrix(X)
        else:
            return self._trigonometric_design_matrix(X)
        
    def _polynomial_design_matrix(self, X):
        """ Return polynomial design matrix of degree J with shape (N, M)

            Args:
                X: input vector of shape (N, 1)

            Output: polynomial design matrix of shape (N, M)
        """

        # --- ENTER SOLUTION HERE ---
        Phi = None
        N = X.shape[0]
        M = self.J + 1
        X = X.reshape(-1)
        Phi = np.zeros((N, M))
        for i in range(M):
            Phi[:, i] = X **i
        return Phi

    def _trigonometric_design_matrix(self, X):
        """ Return trigonometric design matrix of degree J with shape (N, M)

            Args:
                X: input vector of shape (N, 1)

            Output: polynomial design matrix of shape (N, M)
        """

        # --- ENTER SOLUTION HERE ---
        Phi = None
        N = X.shape[0]
        M = 2*self.J + 1
        X = X.reshape(-1)
        Phi = np.zeros((N, M))
        for i in range(M):
            # Cos(0) = 1
            Phi[:, i] = np.cos(np.pi*i*X) if i % 2 == 0 else np.sin(np.pi*(i+1)*X)
        return Phi

    def fit(self, X, Y):
        """ Find maximum likelihood (MLE) solution, given basis Phi and output Y.

        Args:
            Phi: design matrix of shape (M, N)
            Y: vector of shape (N, 1)
            variance: scalar variance

        The function should not return anything, but instead
            1. save maximum likelihood for weights w, a numpy vector of shape (M, N), as variable 'self.mle_w'
            2. save maximum likelihood for variance as float as variable 'self.mle_variance'
        """

        # --- ENTER SOLUTION HERE ---
        N = X.shape[0]
        Phi = self.design_matrix(X)
        M = Phi.shape[1]
        self.mle_w = np.linalg.inv(Phi.T@Phi)@Phi.T@Y
        self.mle_variance = 1/N * np.sum((Y - Phi @ self.mle_w) ** 2)

    def predict(self, X_predict):
        """ Make a prediction using fitted solution.

        Args:
            X_predict: point to make prediction, vector of shape (V, 1)

        Output prediction as numpy vector of shape (V, 1)
        """
        
        # --- ENTER SOLUTION HERE ---
        # hint: remember that you can use functions like 'self.design_matrix(...)'
        #       and the fitted vector 'self.mle_w' here.
        Y_predict = self.design_matrix(X_predict) @ self.mle_w


        return Y_predict

    def predict_range(self, N_points, xmin, xmax):
        """ Make a prediction along a predefined range.

        Args:
            N_points: number of points to evaluate within range
            xmin: start of range to predict
            xmax: end of range to predict

        Returns a tuple containing:
            - numpy vector of shape (N_points, 1) for predicted X locations
            - numpy vector of shape (N_points, 1) for corresponding predicted values Y
        """

        # --- ENTER SOLUTION HERE ---
        X_predict = np.reshape(np.linspace(xmin, xmax, N_points), (N_points,1))
        Y_predict = self.predict(X_predict)

        return X_predict, Y_predict


def leave_one_out_cross_validation(model, X, Y):
    """ Function to perform leave-one-out cross validation.
    
    Args:
        model: Model to perform leave-one-out cross validation.
        X: Full dataset X, of which different folds should be made.
        Y: Labels of dataset X

    Should return two floats:
        - the average test error over different folds
        - the average mle variance over different folds
    """
    N = len(X)

    # --- ENTER SOLUTION HERE ---
    # Hint: use the functions 'model.fit()' to fit on train folds and
    #       the function 'model.predict() to predict on test folds.
    error = 0
    variance = 0
    for i in range(N):
        X_train = np.vstack((X[:i], X[i+1:]))
        print(i, X_train.shape)
        X_test = X[i]

        Y_train = np.vstack((Y[:i], Y[i+1:]))
        Y_test = Y[i]

        model.fit(X_train, Y_train) 
        Y_predict = model.predict(X_test)
        error += np.sum((Y_test - Y_predict) ** 2)
        variance += model.mle_variance

    average_test_error = error / N
    average_mle_variance = variance / N

    return average_test_error, average_mle_variance

if __name__ == '__main__':
                # Test second order J=2 case with small (3, 1) input for X
    dummy_X = np.array([[1.0], [5.0], [3.0], [5.0]])

    model = LinearRegression(basis='polynomial', J=2)
    output = model.design_matrix(dummy_X)
    print(output)
    target_output = np.array([[1.0, 1.0, 1.0],
                              [1.0, 5.0, 25.0],
                              [1.0, 3.0, 9.0],
                              [1.0, 5.0, 25.0]])
    dummy_X = np.array([[1.0], [0.5], [30.0], [0.5]])

    model = LinearRegression(basis='trigonometric', J=1)
    output = model.design_matrix(dummy_X)
    print(output)
    target_output = np.array([[1.00000000e+00, -2.44929360e-16, 1.00000000e+00],
                              [1.00000000e+00, 1.22464680e-16, -1.00000000e+00],
                              [1.00000000e+00, -2.15587355e-14, 1.00000000e+00],
                              [1.00000000e+00, 1.22464680e-16, -1.00000000e+00]])
    correct = np.isclose(output, target_output, atol=1e-3).all()
    print(correct)

    dummy_Phi = np.array([[1.0, 3.0, 4.0], [1.0, 6.0, 7.0], [1.0, 3.0, 5.0], [1.0, 2.0, 3.0]])
    dummy_X = np.array([[2.0], [3.5], [8.0], [9.]])
    dummy_Y = np.array([[4.0], [5.0], [6.0], [7.]])

    # we mock design_matrix function, so we can specifically test the part here that calculates the MLE of the w vector 
    model = LinearRegression(basis='polynomial', J=2)
    model.fit(dummy_X, dummy_Y)

    mle_w = model.mle_w

    target_output = np.array([[6.0], [-0.76923077], [0.46153846]])
    correct = np.isclose(mle_w, target_output, atol=1e-3).all()

