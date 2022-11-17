import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv

def find_theta(x, y):
    m = x.shape[0]
    theta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
    return theta

def find_h_theta(x, theta):
    h_theta = 0
    h_theta = np.dot(x, theta)
    return h_theta

def J_theta(x, y, h_theta):
    m=x.shape[0]
    J_theta = 0
    for i in range(m):
        J_theta += (1/2)*((h_theta[i] - y[i])**2)
    return J_theta

def linerReg(data_X, data_Y):
    lr_model = LinearRegression()
    lr_model.fit(data_X, data_Y)
    print (data_X)
    print(data_Y)

    y_pred = lr_model.predict(data_X)
    print("prediction : " ,y_pred)
    plt.scatter(data_X, data_Y, color="black")
    plt.plot(data_X, y_pred, color="blue", linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

    # The coefficients
    print("Coefficients: \n", lr_model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(data_Y, y_pred))
    return lr_model


def getThetaClosedForm(X, y):
    X_transpose = X.T
    theta = inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
    # normal equation
    # theta_best = (X.T * X)^(-1) * X.T * y

    return theta  # returns a list

def loss_function(m, b, data_X, data_Y ):
    total_error = 0
    for i in range(len(data_X)):
        x = data_X[i]
        y = data_Y[i]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(data_X))

def gradient_descent(m_now, b_now, data_X, data_Y, L):
    m_gradient = 0
    b_gradient = 0
    n = float(len(data_X))
    for i in range(len(data_X)):
        x = data_X[i]
        y = data_Y[i]
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    m = m_now - L * m_gradient
    b = b_now - L * b_gradient
    return [m, b]


