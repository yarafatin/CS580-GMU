import numpy as np
import statsmodels.api as sm
from hw4 import *
from sklearn.linear_model import LinearRegression

def load_data(fname,directory='..\data'):
	data = numpy.loadtxt(os.path.join(directory,fname),delimiter=',')
	rows,cols = data.shape
	X_dim = cols-1
	Y_dim = 1
	return data[:,:-1].reshape(-1,X_dim), data[:,-1].reshape(-1,Y_dim)
# generate sample data (single linear)
X = 2 * np.random.rand(200, 1)
y = 1.2 * X + 1 + 0.8 * np.random.randn(200, 1)
X_ = sm.add_constant(X)  # add constant for intercept computation

print('Method 1: matrix formulation')
print(np.dot(np.linalg.inv(np.dot(X_.T, X_)), np.dot(X_.T, y)))

# statsmodels lib
model = sm.OLS(y, X_).fit()
print('Method 2: statsmodels')
print(f'{model.params}')

# LinearRegression
print('Method 3: sklearn.linear_model.LinearRegression')
lr_model = LinearRegression(fit_intercept=True)
lr_model.fit(X, y)
print(f'Intercept: {lr_model.intercept_}, coeff: {lr_model.coef_}')


data_X, data_Y = load_data('1D-no-noise-lin.txt')
plot_helper(data_X, data_Y)

# LinearRegression
print('Method 4: sklearn.linear_model.LinearRegression')
lr_model = LinearRegression(fit_intercept=True)
lr_model.fit(data_Y, data_X)
print(f'Intercept: {lr_model.intercept_}, coeff: {lr_model.coef_}')

print('Method 1: matrix formulation')
X_ = sm.add_constant(data_X)  # add constant for intercept computation
print(np.dot(np.linalg.inv(np.dot(X_.T, X_)), np.dot(X_.T, data_Y)))

# statsmodels lib
model = sm.OLS(data_Y, X_).fit()
print('Method 2: statsmodels')
print(f'{model.params}')



print ("--------------------------------")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.linear_model import LinearRegression

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
from sklearn.metrics import mean_squared_error
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

print('The size of X is:' ,data_X.shape)
print('The size of y is:' ,data_Y.shape)
plot_helper(data_X, data_Y)
lr_model=linerReg(data_X,data_Y)

from numpy.linalg import inv
def getThetaClosedForm(X, y):
    X_transpose = X.T
    best_params = inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
    # normal equation
    # theta_best = (X.T * X)^(-1) * X.T * y

    return best_params  # returns a list

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

theta_calculated= getThetaClosedForm(data_X, data_Y)
print ( " theta is :" ,theta_calculated)

y_preds = np.dot(data_X, theta_calculated)
print ( " preds is :" ,y_preds)

print("Mean squared error: %.2f" % mean_squared_error(data_Y, y_preds))
exit(0)

# Plotting the predictions.
fig = plt.figure(figsize=(8,6))
plt.plot(data_X, data_Y, 'b.')
plt.plot(data_X, y_preds, 'c-')
plt.xlabel('X - Input')
plt.ylabel('y - target / true')



theta = find_theta(data_X, data_Y)
print('Theta is:' ,theta)
h_theta_train = find_h_theta(data_X, theta)
y_pred_train = h_theta_train
J_train = J_theta(data_X, data_Y, h_theta_train)

xlist = []
ylist = []

m = 0
b = 0
L = 0.0001
epochs = 1000

plt.ion()

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax2.set_xlim([0,epochs])
ax2.set_ylim([0,loss_function(m,b,data_X, data_Y)])

ax1.scatter(data_X, data_Y)
line, = ax1.plot(range(20, 80), range(20, 80), color='red')
line2, = ax2.plot(0,0)

for i in range(epochs):
    m, b = gradient_descent(m, b, data_X, data_Y, L)
    line.set_ydata(m * range(20, 80) + b)

    xlist.append(i)
    ylist.append(loss_function(m, b, data_X, data_Y))
    line2.set_xdata(xlist)
    line2.set_ydata(ylist)

    fig.canvas.draw()

plt.ioff()
plt.show()

fig.set_facecolor('#121212')
ax1.set_title('Linear Regression', color='white')
ax2.set_title('Loss Function', color='white')
ax1.grid(True, color='#323232')
ax2.grid(True, color='#323232')
ax1.set_facecolor('black')
ax2.set_facecolor('black')
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')
ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white')
plt.tight_layout()