from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Dummy data: [load, cycles] → damage
X = []
y = []

for F in np.linspace(100, 500, 10):
    for steps in range(100, 20000, 1000):
        sim_d = simulate_damage(F, L=1.0, I=1e-6, E=200e9, sn_curve_df=sn_df, total_steps=steps)
        X.append([F, steps])
        y.append(sim_d[-1])

X = np.array(X)
y = np.array(y)

model = RandomForestRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, preds))
