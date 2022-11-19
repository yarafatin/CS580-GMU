import numpy as np
import statsmodels.api as sm
from hw4 import *
from hw4_module import *
from sklearn.linear_model import LinearRegression



data_X, data_Y = load_data('1D-no-noise-lin.txt')
data_X, data_Y = load_data('2D-noisy-lin.txt')

#plot_helper(data_X, data_Y)

print ("--------------------------------")


#print('The size of X is:' ,data_X.shape)
#print('The size of y is:' ,data_Y.shape)
#plot_helper(data_X, data_Y)


theta_calculated= getThetaClosedForm(data_X, data_Y)
print ( " theta is :" ,theta_calculated)

y_preds = np.dot(data_X, theta_calculated)
#print ( " preds is :" ,y_preds)

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