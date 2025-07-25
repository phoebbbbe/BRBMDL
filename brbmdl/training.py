import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from sklearn.tree import export_text
from sklearn.preprocessing import LabelEncoder


def dataset_loading(features: list, target: str,
                    *paths: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, y = [], []
    for path in paths:
        data = pd.read_csv(f'{path}')
        X.append(data[features])
        y.append(data[[target]])

    X = pd.concat(X, ignore_index=True)
    y = pd.concat(y, ignore_index=True)
    return X, y


def process_dataset(y_set, cons_name) -> pd.DataFrame:
    y_label_set = y_set[cons_name].apply(lambda x: 'up' if x > 0.0 else 'down')
    y_label_set = pd.DataFrame(y_label_set)
    return y_label_set


def process_dl_dataset(X_train, X_test, y_train, y_test, target):
    X_train_dl = np.array(X_train)
    X_test_dl = np.array(X_test)
    X_train_dl = X_train_dl.reshape((X_train_dl.shape[0], X_train_dl.shape[1], 1))
    X_test_dl = X_test_dl.reshape((X_test_dl.shape[0], X_test_dl.shape[1], 1))
    convertor = LabelEncoder()
    y_train_dl = convertor.fit_transform(process_dataset(y_train, target))
    y_test_dl = convertor.fit_transform(process_dataset(y_test, target))
    return X_train_dl, X_test_dl, y_train_dl, y_test_dl

def create_multivariate_dataset(X_data, y_data, window_size=5):
    X, y = [], []
    for i in range(len(X_data) - window_size):
        X.append(X_data[i:i + window_size])
        y.append(y_data[i + window_size])
    return np.array(X), np.array(y)



def gridsearch_decisiontree(X_train, y_train):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [240, 250, 260, 270, 280, 290, 300, 310, 320, 330],
        'max_features': [None, 'sqrt', 'log2']
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search


def search_performance(search, X_test, y_test):
    print("Best parameters found: ", search.best_params_)
    print("Best cross-validation score: {:.2f}".format(search.best_score_))

    best_model = search.best_estimator_
    test_accuracy = best_model.score(X_test, y_test)
    print("Test set accuracy: {:.2f}".format(test_accuracy))

    y_pred = best_model.predict(X_test)
    print("Classify Report:\n", classification_report(y_test, y_pred))
    return


def build_decisiontree(search, X_train, y_train):
    dt = DecisionTreeClassifier(
            criterion=search.best_params_['criterion'],
            max_depth=search.best_params_['max_depth'],
            max_features=search.best_params_['max_features']
            )
    dt.fit(X_train, y_train)
    return dt


def export_treerules(dt_model, features):
    tree_rules = export_text(dt_model, feature_names=features)
    # print("Decision Tree Rules:", tree_rules)
    with open("./data/tree_rules.txt", "w") as fp:
        fp.write(tree_rules)
        # print("Save Decision Tree Rules")
    return tree_rules


def backpropagation(X, y, weights):
    
    def tanh(x):
        return np.tanh(x)
    
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        return x * (1 - x)
    
    output_size = 1
    np.random.seed(42)
    bias = np.random.randn(output_size)

    learning_rate = 9.65E-03
    epochs = 10000

    for epoch in range(epochs):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)

        error = y - y_pred
        loss = np.mean(np.square(error))

        d_loss = error * sigmoid_derivative(y_pred)
        d_weights = np.dot(X.T, d_loss)
        d_bias = np.sum(d_loss)

        weights += learning_rate * d_weights
        bias += learning_rate * d_bias

        weights = np.clip(weights, 0, None)
        weights = weights / np.sum(weights)

        # 每 100 次迭代輸出一次誤差
        # if epoch % 1000 == 0:
        #     print(f'epoch {epoch}, loss: {loss:.6f}')

    # print(f'\n final weights: {weights}')
    # print(f'final bias: {bias}')
    # test_output = tanh(np.dot(X, weights) + bias)
    # print(f'value: {y}, pred_value: {test_output}')
    return weights
