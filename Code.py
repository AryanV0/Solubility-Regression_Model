import pandas as pd

df = pd.read_csv("delaney_solubility_with_descriptors.csv")
print(df.head())

y = df["logS"]
print(y)

x = df.drop("logS", axis=1)
print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=100)

print(x_train)
print(x_test)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

print(y_lr_train_pred)
print(y_lr_test_pred)



from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print("lr mse (Train)): ", lr_train_mse)
print("lr r2 (Train)): ", lr_train_r2)
print("lr mse (Train)): ", lr_train_mse)
print("lr r2 (Train)): ", lr_test_r2)

lr_results = pd.DataFrame([["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]])
lr_results.columns = ["Methods", "Training MSE", "Training R2", "Test MSE", "Test R2"]
print(lr_results)


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)


y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)


from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame([["Random Forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]])
rf_results.columns = ["Methods", "Training MSE", "Training R2", "Test MSE", "Test R2"]
print(rf_results)

df_models = pd.concat([lr_results, rf_results], axis=0)

print(df_models.reset_index(drop=True))


import matplotlib.pyplot as plt
import numpy as np

plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.3)
z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train))
plt.ylabel("Predict LogS")
plt.xlabel("Experimental LogS")

plt.show()



































