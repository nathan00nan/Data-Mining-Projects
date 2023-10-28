import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

iris = pd.read_csv(r"C:\Users\pumpk\Desktop\Nathan's File\DOCUMENTS\My Notes\Projects\KNN.csv")
print("Target Labels:", iris["species"].unique())

x = iris.drop("species", axis=1)
y = iris["species"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

knn_best = KNeighborsClassifier(**best_params)
knn_best.fit(x_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn_best.predict(x_new)
print("Prediction:", prediction)

y_pred = knn_best.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))