import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_grid(axis_x, axis_y, grid_step_size=.02):
    x_min, x_max = axis_x.min() - 1, axis_x.max() + 1
    y_min, y_max = axis_y.min() - 1, axis_y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step_size),
                         np.arange(y_min, y_max, grid_step_size))

    return xx, yy


def plot_contours(axes, svc_model, xx, yy, **params):
    # Predict each point in the grid
    predictions = svc_model.predict(np.c_[xx.ravel(), yy.ravel()])
    predictions = predictions.reshape(xx.shape)
    axes.contourf(xx, yy, predictions, **params)


# Load iris data
iris = datasets.load_iris()

# Consider two dimensions out of four
X = iris.data[:, :2]
y = iris.target

# Create model
C = 1.0  # SVM regularization parameter
model = svm.SVC(kernel='rbf', gamma=0.7, C=C)

# Train model
model.fit(X, y)

# Make grid. use two columns selected as x axis and y axis
X0, X1 = X[:, 0], X[:, 1]
x_axis, y_axis = make_grid(X0, X1)

# Plot
fig, ax = plt.subplots()
plot_contours(ax, model, x_axis, y_axis, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(x_axis.min(), x_axis.max())
ax.set_ylim(y_axis.min(), y_axis.max())
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('SVC with RBF kernel')

plt.show()
