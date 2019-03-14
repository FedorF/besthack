import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np

aero = pd.read_csv('./data/fv.csv')
wind = pd.read_csv('./data/wind.csv')

def polynomial_model(X, y, degree):
    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                           ('affine', Ridge())])
    model.fit(X, y)
    return model


Wx_model = polynomial_model(wind['Y'].values.reshape(-1, 1), wind['Wx'], 3)
X = np.linspace(0, 1500, 200).reshape(-1, 1)
pred = Wx_model.predict(X)
plt.ion()
plt.figure()
plt.scatter(wind['Y'], wind['Wx'], c='y', marker='o')
plt.plot(X, pred)
plt.title('Polynomial degree: {}'.format(3))
plt.xlabel('Y')
plt.ylabel('Wx')
plt.ioff()


Wz_model = polynomial_model(wind['Y'].values.reshape(-1, 1), wind['Wz'], 5)
X = np.linspace(0, 1400, 200).reshape(-1, 1)
pred = Wz_model.predict(X)
plt.ion()
plt.figure()
plt.scatter(wind['Y'], wind['Wz'], c='y', marker='o')
plt.plot(X, pred)
plt.title('Polynomial degree: {}'.format(5))
plt.xlabel('Y')
plt.ylabel('Wz')
plt.ioff()



Fv_model = polynomial_model(aero['V'].values.reshape(-1, 1), aero['Fa'], 2)
X = np.linspace(0, 300, 100).reshape(-1, 1)
pred = Fv_model.predict(X)
plt.ion()
plt.figure()
plt.scatter(aero['V'], aero['Fa'], c='y', marker='o')
plt.plot(X, pred, c='g')
plt.title('Polynomial degree: {}'.format(2))
plt.xlabel('V')
plt.ylabel('Fa')
plt.ioff()