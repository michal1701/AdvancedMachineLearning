import numpy as np
import matplotlib.pyplot as plt

from implementation.measures import Measure

class LogRegCCD:
    """ this text is showed in 'Docstring' field when '?LogRegCCD' is used """
    
    __slots__ = ["beta0_", "beta_", "C_", "alpha_", "mean_", "std_"]

    def __init__(self, C=None, alpha=1):
        """ this text is showed in 'Init docstring' field when '?LogRegCCD' is used """

        self.beta0_ = None
        self.beta_ = None
        self.C_ = C
        self.alpha_ = alpha
        self.mean_ = 0
        self.std_ = 0

        return

    def __set_standard(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return

    def __standarize(self, X):
        return (X - self.mean_) / self.std_
    
    def fit(self, X, y, max_iter=100, use_weights=True, fit_intercept=True):
        n, p = X.shape
        
        self.__set_standard(X)
        X = self.__standarize(X)
        
        prior = 0.5
        if fit_intercept:
            prior = y.mean()
            
        self.beta0_ = np.log(prior / (1 - prior))    
        self.beta_ = np.zeros(p)
        
        weights = 1 / n
        wx_squared = 1
        if use_weights:
            weights = prior * (1 - prior)
        
        if self.C_ is None:
            lambdas = LogRegCCD.__generate_lambdas(X, y, self.alpha_, max_iter, use_weights, fit_intercept)
        else:
            lambdas = np.repeat(1 / self.C_, max_iter)
        
        for lmbd in lambdas:
            probs = LogRegCCD.sigmoid(self.beta0_ + X @ self.beta_)
            w = probs * (1 - probs)
            
            for j in range(p):
                if use_weights:
                    weights = w
                    wx_squared = weights @ X[:, j] ** 2
                    
                s = (weights * (y - probs + w * X[:, j] * self.beta_[j])) @ X[:, j]
                self.beta_[j] = LogRegCCD.soft_threshold(s, lmbd * self.alpha_) / (wx_squared + lmbd * (1 - self.alpha_))

        return self

    def validate(self, X, y, measure: "Measure.Type"):
        assert isinstance(measure, Measure.Type)

        match measure:
            case Measure.Type.AUC_ROC | Measure.Type.AUC_PR:
                return Measure.from_type(measure).get(y, self.predict_proba(X))
            case _:
                return Measure.from_type(measure).get(y, self.predict(X))

    def predict_proba(self, X):
        return LogRegCCD.sigmoid(self.beta0_ + self.__standarize(X) @ self.beta_)

    def predict(self, X):        
        return np.where(self.predict_proba(X) > 0.5, 1, 0)

    @staticmethod
    def __generate_lambdas(X, y, alpha, max_iter=100, use_weights=True, fit_intercept=True):
        n, p = X.shape
        
        prior = 0.5
        if fit_intercept:
            prior = y.mean()
        
        weights = 1 / n
        if use_weights:
            weights = prior * (1 - prior)
        
        lambda_max = abs(weights * (y - prior) @ X).max()
        if alpha != 0:
            lambda_max /= alpha
        lambda_min = 0.001 * lambda_max
        
        return np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), max_iter))

    @staticmethod
    def soft_threshold(z, gamma):
        return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def gaussian(x):
        return norm.cdf(x)

    @staticmethod
    def plot(X_train, y_train, X_valid, y_valid, measure_types: ("Measure.Type", list)):
        if not isinstance(measure_types, list):
            measure_types = [measure_types]
        for measure_type in measure_types:
            assert isinstance(measure_type, Measure.Type)

        lambdas = LogRegCCD.__generate_lambdas(X_train, y_train, 1)
        measures_values = [None for _ in range(len(lambdas))]

        for i in range(len(lambdas)):
            model = LogRegCCD(C=1 / lambdas[i]).fit(X_train, y_train)
            tmp = []
            for measure_type in measure_types:
                tmp.append(model.validate(X_valid, y_valid, measure_type))
            
            measures_values[i] = tmp

        plt.plot(lambdas, measures_values)
        plt.xlabel("lambda")
        plt.ylabel("measures values")
        plt.legend([Measure.from_type(measure_type).get_name() for measure_type in measure_types])
        plt.show()

        return
        
    @staticmethod
    def plot_coefficients(X_train, y_train):
        lambdas = LogRegCCD.__generate_lambdas(X_train, y_train, 1)
        
        coefficients = [None for _ in range(len(lambdas))]

        for i in range(len(lambdas)):
            model = LogRegCCD(C=1/lambdas[i]).fit(X_train, y_train)
            coefficients[i] = (model.beta0_, *model.beta_)

        plt.plot(lambdas, coefficients)
        plt.xlabel("lambda")
        plt.ylabel("coefficients values")
        plt.legend([f"beta_{i}" for i in range(len(model.beta_) + 1)])
        plt.show()

        return