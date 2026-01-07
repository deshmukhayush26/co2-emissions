import numpy as np # linear algebra
import pandas as pd # data processing

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) # to find path of dataset

df = pd.read_csv("/kaggle/input/co2-emission-by-vehicles/CO2 Emissions_Canada.csv")
df_encoded = pd.get_dummies(
    df,
    columns = ["Make", "Vehicle Class", "Fuel Type"],
    drop_first = True
)


x = df_encoded.drop(columns = ["Model", 
                               "Transmission", 
                               "Fuel Consumption Comb (L/100 km)", 
                               "Fuel Consumption Hwy (L/100 km)", 
                               "Fuel Consumption City (L/100 km)",
                               "CO2 Emissions(g/km)",
                              ])
y = df_encoded["CO2 Emissions(g/km)"]
x.shape, y.shape

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size = 0.3,
    random_state=67
)

from sklearn.linear_model import LinearRegression #linear regression prediction
from sklearn.metrics import r2_score, mean_squared_error

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2, rmse

from sklearn.linear_model import Ridge #ridge regression prediction

ridge = Ridge(alpha=1.0)
ridge.fit(x_train, y_train)
y_pred_ridge = ridge.predict(x_test)

r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

r2_ridge, rmse_ridge

import matplotlib.pyplot as plt
#linear plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()
#ridge plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_ridge, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Ridge Regression: Actual vs Predicted")
plt.show()
