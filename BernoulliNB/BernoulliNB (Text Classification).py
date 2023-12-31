import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv(r"C:\Users\pumpk\Desktop\Nathan's File\DOCUMENTS\My Notes\Projects\BernoulliNB.csv")
data = data[["CONTENT", "CLASS"]]
data["CLASS"] = data["CLASS"].map({0: "Not Spam", 1: "Spam Comment"})
x = np.array(data["CONTENT"])
y = np.array(data["CLASS"])

cv = CountVectorizer()
x = cv.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid = {'alpha': [0.1, 0.5, 1.0]}

model = BernoulliNB()

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(xtrain, ytrain)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

accuracy = best_model.score(xtest, ytest)
print("Accuracy:", accuracy)

ypred = best_model.predict(xtest)
precision = precision_score(ytest, ypred, average='weighted')
recall = recall_score(ytest, ypred, average='weighted')
f1 = f1_score(ytest, ypred, average='weighted')
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

cm = confusion_matrix(ytest, ypred)
print("Confusion Matrix:")
print(cm)

print("Best Hyperparameters:", best_params)