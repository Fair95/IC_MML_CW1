import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from solution import LinearRegression
from solution import leave_one_out_cross_validation

class TestCoursework(unittest.TestCase):
    def test_polynomial_design_matrix(self):
        """ Unit test to check whether the 'polynomial_design_matrix' function is correct. """

        # Test second order J=2 case with small (3, 1) input for X
        dummy_X = np.array([[1.0], [5.0], [3.0], [5.0]])

        model = LinearRegression(basis='polynomial', J=2)
        output = model.design_matrix(dummy_X)

        target_output = np.array([[1.0, 1.0, 1.0],
                                  [1.0, 5.0, 25.0],
                                  [1.0, 3.0, 9.0],
                                  [1.0, 5.0, 25.0]])

        correct = np.isclose(output, target_output, atol=1e-3).all()

        self.assertTrue(correct, f"The function 'polynomial_design_matrix' does not seem to be correct yet.")

    def test_trigonometric_design_matrix(self):
        """ Unit test to check whether the 'polynomial_design_matrix' function is correct. """

        # Test second order J=2 case with small (3, 1) input for X
        dummy_X = np.array([[1.0], [0.5], [30.0], [0.5]])

        model = LinearRegression(basis='trigonometric', J=1)
        output = model.design_matrix(dummy_X)

        target_output = np.array([[1.00000000e+00, -2.44929360e-16, 1.00000000e+00],
                                  [1.00000000e+00, 1.22464680e-16, -1.00000000e+00],
                                  [1.00000000e+00, -2.15587355e-14, 1.00000000e+00],
                                  [1.00000000e+00, 1.22464680e-16, -1.00000000e+00]])

        correct = np.isclose(output, target_output, atol=1e-3).all()

        self.assertTrue(correct, f"The function 'polynomial_design_matrix' does not seem to be correct yet.")

    def test_fit_w(self):
        """ Unit test to check whether the 'fit' function produces correct weight vector w. """

        dummy_Phi = np.array([[1.0, 2.0, 4.0],
                              [1.0, 3.5, 12.25],
                              [1.0, 8.0, 64.0],
                              [1.0, 1.0, 1.0]])
        dummy_X = np.array([[2.0], [3.5], [8.0], [1.]])
        dummy_Y = np.array([[4.0], [5.0], [6.0], [7.]])

        # we mock design_matrix function, so we can specifically test the part here that calculates the MLE of the w vector 
        with patch("solution.LinearRegression.design_matrix", MagicMock(return_value=dummy_Phi)) as mock_bar:
            model = LinearRegression(basis='polynomial', J=2)
            model.fit(dummy_X, dummy_Y)

        mle_w = model.mle_w

        target_output = np.array([[7.75137411], [-1.57911355], [0.17097416]])
        correct = np.isclose(mle_w, target_output, atol=1e-3).all()

        self.assertTrue(correct, f"The function 'fit' does not seem to be correct yet.")

    def test_fit_mle_variance(self):
        """ Unit test to check whether the 'fit' function produces correct variance. """

        dummy_Phi = np.array([[1.0, 2.0, 4.0],
                              [1.0, 3.5, 12.25],
                              [1.0, 8.0, 64.0],
                              [1.0, 1.0, 1.0]])
        dummy_X = np.array([[2.0], [3.5], [8.0], [1.]])
        dummy_Y = np.array([[4.0], [5.0], [6.0], [7.]])

        # we mock design_matrix function, so that we can specifically test the part here that calculates the MLE of the variance
        with patch("solution.LinearRegression.design_matrix", MagicMock(return_value=dummy_Phi)) as mock_bar:
            model = LinearRegression(basis='polynomial', J=2)
            model.fit(dummy_X, dummy_Y)

            mle_variance = model.mle_variance

        target_mle_variance = 0.6324406502163489
        correct = np.isclose(mle_variance, target_mle_variance, atol=1e-3).all()

        self.assertTrue(correct, f"The function 'fit' does not seem to be correct yet.")

    def test_leave_one_out_cross_validation_mle_variance(self):
        """ Unit test to check whether the 'leave_one_out_cross_validation' function produces correct variance. """

        dummy_X = np.array([[2.0], [3.5], [8.0], [9.]])
        dummy_Y = np.array([[4.0], [5.0], [6.0], [7.]])

        model = LinearRegression(basis='polynomial', J=0)
        _, average_mle_variance_J0 = leave_one_out_cross_validation(model, dummy_X, dummy_Y)

        model = LinearRegression(basis='polynomial', J=2)
        _, average_mle_variance_J2 = leave_one_out_cross_validation(model, dummy_X, dummy_Y)

        # Check that higher degree basis (J=2 compared to J=0) should result in lower mle variance
        check1 = average_mle_variance_J2 < average_mle_variance_J0

        # With degree J=2, then basis dimensionality M=2+1=3 should match the dimensionality of a leave one-out fold (4-1=3)
        # and we therefore expect a perfect fit and mle variance close to 0.
        check2 = np.isclose(average_mle_variance_J2, 0.0, atol=1e-3).all()

        self.assertTrue(check1, f"The function 'leave_one_out_cross_validation' does not seem to be correct yet.")
        self.assertTrue(check2, f"The function 'leave_one_out_cross_validation' does not seem to be correct yet.")

    def test_leave_one_out_cross_validation_test_error(self):
        """ Unit test to check whether the 'leave_one_out_cross_validation' function produces correct test error. """

        dummy_X = np.array([[2.0], [3.5], [8.0], [9.]])
        dummy_Y = np.array([[4.0], [5.0], [6.0], [7.]])

        model = LinearRegression(basis='polynomial', J=0)
        average_test_error_J0, _ = leave_one_out_cross_validation(model, dummy_X, dummy_Y)

        model = LinearRegression(basis='polynomial', J=1)
        average_test_error_J1, _ = leave_one_out_cross_validation(model, dummy_X, dummy_Y)

        # Check that J=1 results in better test error than J=0.
        check = average_test_error_J1 < average_test_error_J0

        self.assertTrue(check, f"The function 'leave_one_out_cross_validation' does not seem to be correct yet.")


if __name__ == '__main__':
    unittest.main()

