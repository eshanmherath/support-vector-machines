import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X = np.linspace(0, 1000)
y = np.array([(10 * np.random.rand(1) + num) for num in (0.1 * X + 2)])

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = SVR(kernel='linear')
model.fit(X_train.reshape(-1, 1), y_train.flatten())

predictions = model.predict(X_test.reshape(-1, 1))

plt.plot(X, y, label='Actual Pattern')
plt.plot(X_test[::2], predictions[::2], 'ro', label='SVR model')
plt.legend(loc='upper left')
plt.show()
