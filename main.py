import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy.linalg import inv

data = pd.read_csv('Real estate.csv')

# check on which data is less valuable or has a negative effect
data.corr()["house_price"]

"""
 if  the data contains alot of x variable look for the correlation hat is not close to 1.0  drop by using
 
 house price = data["house price"]
 data =  data.drop(['x1','x2', 'x3' ,' x4 '], axis = 1)
"""

# create  graph of each individual input(x) with the y output
plt.scatter(data.latitude, data.house_price)
plt.xlabel('Space')
plt.ylabel("House price")
plt.show()

# change to numpty aray
dat_np = data.to_numpy()
ds = dat_np.shape
print(ds)

"""
: => this represent all the columns
"""
x_train, y_train = dat_np[:, :6], dat_np[:, -1]
x_train.shape, y_train.shape
print(x_train.shape)
print(y_train.shape)

# just a guideline  to know u have the correct outcome while building from scratch
sklearn_model = LinearRegression().fit(x_train, y_train)
sklearn_y_predictions = sklearn_model.predict(x_train)
pred = sklearn_y_predictions.shape

print(pred)

# knowing how off our model is from the real output
mean = mean_absolute_error(sklearn_y_predictions, y_train), mean_squared_error(sklearn_y_predictions, y_train)
print(mean)

predictions = pd.DataFrame({"Transaction_date": data["transaction_date"],
                            "House_age": data["house_age"],
                            "Distance_station": data["distance_station"],
                            "Convenience_stores": data["convenience_stores"],
                            "Latitude": data["latitude"],
                            "Longitude": data["longitude"],
                            "House_price": data["house_price"],
                            "price prediction": sklearn_y_predictions
                            })
print(predictions)

"""
from the above prediction we get that our values are off hence we need to set it with the function formula of

f(x) = y
"""


def get_prediction(model, x):
    """
    :param model: is the  parameters (p) such as alpha + alpha .X
    :param x:  are the input variables (n, p-1)
    :return:np.array of floats with shapes(n)

    """
    (n, p_minus_one) = x.shape
    p = p_minus_one + 1

    new_x = np.ones(shape=(n, p))
    new_x[:, 1:] = x

    return np.dot(new_x, model)


# test model  change the array to be in range with the prediction
test_model = np.array([1, .2, 1, .2, .5, .1, 1])
fun = get_prediction(test_model, x_train)
# print(fun)

predictions["Test Predictions"] = fun

print(predictions)

mn = mean_absolute_error(predictions["Test Predictions"], y_train)
print(mn)


def get_bestModel(x, y):
    """"
    :param y: is the  parameters (p) such as alpha + alpha .X
    :param X:  are the input variables (n, p-1)
    :return:np.array of floats with shapes(n)

    """

    (n, p_minus_one) = x.shape
    p = p_minus_one + 1

    new_x = np.ones(shape=(n, p))
    new_x[:, 1:] = x

    # linear regression n statistics
    return np.dot(np.dot(inv(np.dot(new_x.T, new_x)), new_x.T), y)


best_Model = get_bestModel(x_train, y_train)
predictions["Best Predictions"] = get_prediction(best_Model, x_train)
mn2 = mean_absolute_error(predictions["Best Predictions"], y_train)
print(mn2)
print(predictions)
