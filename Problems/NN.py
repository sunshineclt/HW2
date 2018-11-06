from Problems.Problem import Problem
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import softmax_2D
from Methods.GeneticAlgorithm import GeneticAlgorithm


class NN(Problem):
    def __init__(self):
        dataset = pd.read_csv('../HW1_data/mobile-price-classification/train.csv')
        self.X = dataset.drop('price_range', axis=1)
        self.y = dataset['price_range']

        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=6357)
        X_train, X_val, self.y_train, self.y_val = train_test_split(X_train, self.y_train, test_size=0.2, random_state=6357)
        X_mean = X_train.mean()
        X_max = X_train.max()
        X_min = X_train.min()
        self.normalize = lambda raw: (raw - X_mean) / (X_max - X_min)
        self.denormalize = lambda raw: raw * (X_max - X_min) + X_mean
        self.X_train = self.normalize(X_train)
        self.X_val = self.normalize(X_val)
        self.X_test = self.normalize(X_test)
        self.describe_data()

    def describe_data(self):
        print("Training amount: ", self.X_train.shape[0])
        print("Validation amount: ", self.X_val.shape[0])
        print("Test amount: ", self.X_test.shape[0])

    def generate_params(self, hidden_dim):
        input_dim = self.X_train.shape[1]
        output_dim = 4
        W1 = np.random.normal(size=(input_dim, hidden_dim))
        b1 = np.random.normal(loc=0.01, size=(1, hidden_dim))
        W2 = np.random.normal(size=(hidden_dim, output_dim))
        b2 = np.random.normal(loc=0.01, size=(1, output_dim))
        return {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2}

    def evaluate(self, param):
        return self.evaluate_on_dataset(param, dataset="train")

    def evaluate_on_dataset(self, params, dataset="train"):
        if dataset == "train":
            X = self.X_train
            y = self.y_train
        elif dataset == "val":
            X = self.X_val
            y = self.y_val
        else:
            X = self.X_test
            y = self.y_test

        y_predict = np.argmax(softmax_2D(np.dot(X, params["W1"]) + params["b1"]).dot(params["W2"]) + params["b2"], axis=1)
        score = np.sum(y_predict == y)
        return score

    def evaluate_on_all_datasets(self, method):
        if isinstance(method, GeneticAlgorithm):
            params = method.param_group[np.argmax(method.param_score)]
        else:
            params = method.params
        train_acc = self.evaluate_on_dataset(params, dataset="train") / self.y_train.shape[0]
        val_acc = self.evaluate_on_dataset(params, dataset="val") / self.y_val.shape[0]
        test_acc = self.evaluate_on_dataset(params, dataset="test") / self.y_test.shape[0]
        print("Training acc: %f" % train_acc)
        print("Validation acc: %f" % val_acc)
        print("Test acc: %f" % test_acc)
        return train_acc, val_acc, test_acc
