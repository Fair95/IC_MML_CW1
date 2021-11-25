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

        return Phi

    def _trigonometric_design_matrix(self, X):
        """ Return trigonometric design matrix of degree J with shape (N, M)

            Args:
                X: input vector of shape (N, 1)

            Output: polynomial design matrix of shape (N, M)
        """

        # --- ENTER SOLUTION HERE ---
        Phi = None


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

        Phi = self.design_matrix(X)

        # --- ENTER SOLUTION HERE ---
        self.mle_w = None
        self.mle_variance = None


    def predict(self, X_predict):
        """ Make a prediction using fitted solution.

        Args:
            X_predict: point to make prediction, vector of shape (V, 1)

        Output prediction as numpy vector of shape (V, 1)
        """
        
        # --- ENTER SOLUTION HERE ---
        # hint: remember that you can use functions like 'self.design_matrix(...)'
        #       and the fitted vector 'self.mle_w' here.


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
        X_predict = None
        Y_predict = None


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
    average_test_error = None
    average_mle_variance = None


    return average_test_error, average_mle_variance


