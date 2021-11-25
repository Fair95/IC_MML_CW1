import numpy as np
import argparse
import matplotlib.pyplot as plt

from solution import LinearRegression
from solution import leave_one_out_cross_validation


N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10 * X**2) + 0.1 * np.sin(100 * X)

def main(args):
    if args.plot == 1:
        # Show predictive mean with polynomial basis functions
        plt.title("Predictive mean with polynomial basis functions.")
        plt.scatter(X, Y, marker='x', color='black', label='Data')
        plt.xlim(-0.4, 1.4)
        plt.ylim(-1.5, 2.0)
        plt.xlabel('x')
        plt.ylabel('f(x)')

        for J in [0, 1, 2, 3, 11]:
            model = LinearRegression(basis='polynomial', J=J)
            Phi = model.fit(X, Y)

            X_predict, Y_predict = model.predict_range(N_points=200, xmin=-0.3, xmax=1.3)

            plt.plot(X_predict, Y_predict, label=f"Order J={J}")

        plt.legend()
        plt.show()
    elif args.plot == 2:
        # Show predictive mean with trigonometric basis functions
        plt.title("Predictive mean with trigonometric basis functions.")
        plt.scatter(X, Y, marker='x', color='black', label='Data')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-2.0, 2.0)
        plt.xlabel('x')
        plt.ylabel('f(x)')

        for J in [1, 11]:
            model = LinearRegression(basis='trigonometric', J=J)
            Phi = model.fit(X, Y)

            X_predict, Y_predict = model.predict_range(N_points=200, xmin=-1.0, xmax=1.2)

            plt.plot(X_predict, Y_predict, label=f"Order J={J}")

        plt.legend()
        plt.show()
    elif args.plot == 3:
        # Test error and sigma with trigonometric basis functions
        plt.title("Test error and sigma with trigonometric basis functions")
        plt.xlim(0, 10)
        plt.ylim(0, 1.0)
        plt.xlabel('Order (J)')
        plt.ylabel('')

        Js = list(range(0, 11))
        average_errors = []
        average_variances = []
        for J in Js:
            model = LinearRegression(basis='trigonometric', J=J)

            average_error, average_variance = leave_one_out_cross_validation(model, X, Y)

            average_errors.append(average_error)
            average_variances.append(average_variance)

        plt.plot(Js, average_errors, color='green', label='Average squared test error')
        plt.plot(Js, average_variances, color='blue', label='Sigma maximum likelihood')

        plt.legend()
        plt.show()
    else:
        print(f"Unknown plot. Choose (1) polynomial fits, (2) trigonometric fits, and (3) for test/sigma plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting function for coursework 2 in Mathematics for Machine Learning course.")
    parser.add_argument('--plot', type=int, help="Choose (1) polynomial fits, (2) trigonometric fits, and (3) for test/sigma plot.", default=1)

    args = parser.parse_args()
    main(args)

