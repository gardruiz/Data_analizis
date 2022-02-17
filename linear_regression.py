import numpy as np
from sklearn.linear_model import LinearRegression

listx =[19.11, 22.37, 27.28, 31.93]
listy =[1000, 10000, 100000, 1000000]

#listx and y are the list of values used to build the model
#Enter an x value to estimate the y value

x = np.array([listx]).reshape((-1, 1))
y = np.array([listy]).reshape((-1, 1))
#.reshape() is only used for multy-dimensional assays
print(x)
print(y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')
x_new = np.array(19.11).reshape((-1, 1))
y_new = model.predict(x_new)
print("y value", y_new)

