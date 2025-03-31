import numpy as np
import matplotlib.pyplot as plt

from implementation.measures import Measure

class LogRegCCD:
    """
    Logistic Regression classifier using cyclic coordinate descend optimization method.
    
    This class implements regularized logistic regression with Elastic-Net penalty.
    
    Parameters
    ----------    
    C : float, default=None
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.
    
    alpha : float, default=1.0
        The Elastic-Net mixing parameter, with ``0 <= alpha <= 1``.
        Setting ``alpha=0`` is equivalent to having ridge regularization (L2),
        while setting ``alpha=1`` is equivalent to having lasso
        regularization (L1). For ``0 < alpha <1``, the penalty is a
        combination of L1 and L2.

    Attributes
    ----------
    beta_ : ndarray of shape (n_features, )
        Coefficient of the features in the decision function.
    
    beta0_ : float
        Intercept (a.k.a. bias) added to the decision function.
    
        If `fit_intercept` is set to False, the intercept is set to zero.
    """

    __slots__ = ["beta0_", "beta_", "lambda_", "alpha_", "mean_", "std_", "classes_"]

    def __init__(self, lambda_=None, alpha=1.0):
        # set intercept and coefficients to None as classifier is not fitted yet
        self.beta0_ = None
        self.beta_ = None
        # set value of penalty parameter
        self.lambda_ = lambda_
        # set value of Elastic-Net L1 ratio
        self.alpha_ = alpha
        self.mean_ = 0
        self.std_ = 0
        self.classes_ = None

        return

    def __set_standard(self, X):
        """ Calculates and stores values of mean and standard deviation of given X. """

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return

    def __standarize(self, X):
        """ Standarizes passed value X based on stored values of mean and standard deviation. """

        return (X - self.mean_) / self.std_

    def fit(self, X, y, max_iter=100, use_weights=True, fit_intercept=True, plot_coefficients=False):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        max_iter : int, default=100
            Maximum number of iterations taken for the solvers to converge.

        use_weights: bool, default=True
            Specifies if method should use weights optimization.
            If `use_weights` is False, then each sample is given unit weight.

        fit_intercept : bool, default=True
            Specifies if a constant (a.k.a. bias or intercept) should be
            added to the decision function.

        plot_coefficients : bool, default=False
            If True plots how coefficients update based on iteration of algorithm.

        Returns
        -------
        self
            Fitted estimator.
        """

        # retrive X shape
        n, p = X.shape

        self.classes_ = np.unique(y)

        # calculate mean and std
        self.__set_standard(X)
        # standarize X
        X = self.__standarize(X)

        # default prior probability
        prior = 0.5
        if fit_intercept:
            # update prior based on y
            prior = y.mean()

        # estimate intercept
        self.beta0_ = np.log(prior / (1 - prior))
        # initialize coefficients of features
        self.beta_ = np.zeros(p)

        # default weights in optimization algorithm
        weights = 1 / n
        # default centering coefficient in optimization algorithm
        wx_squared = 1
        if use_weights:
            # update weights based on prior
            weights = prior * (1 - prior)

        # if reciprocal of penalty parameter was not specified
        if self.lambda_ is None:
            # then generate sequence of lambdas used in `warm-start` strategy
            lambdas = LogRegCCD.__generate_lambdas(X, y, self.alpha_, max_iter, use_weights, fit_intercept)
        else:
            # otherwise use specified value
            lambdas = np.repeat(self.lambda_, max_iter)

        coefficients = []
        loss = []

        # iterate over values of penalty parameters
        for lmbd in lambdas:
            # calculate estimate probabilities
            probs = LogRegCCD.sigmoid(self.beta0_ + X @ self.beta_)
            # calculate weights needed for logisitc quadratic approximation (not weights of optimization algorithm)
            w = probs * (1 - probs)

            # iterate over every coefficient
            for j in range(p):
                if use_weights:
                    # update weigths and centering coefficient if such behavior specified
                    weights = w
                    wx_squared = weights @ X[:, j] ** 2

                # calculate value of first argument to be passed to soft-thresholding function
                s = (weights * (y - probs + w * X[:, j] * self.beta_[j])) @ X[:, j]
                # update beta_j based on weighted optimization algorithm
                self.beta_[j] = LogRegCCD.soft_threshold(s, lmbd * self.alpha_) / (wx_squared + lmbd * (1 - self.alpha_))

            coefficients.append(self.beta_)
            loss.append(LogRegCCD.cross_entropy(y, self.predict_proba(X)))

        if plot_coefficients:
            plt.plot(coefficients)
            plt.xlabel("iteration")
            plt.ylabel("coefficients values")
            plt.legend([f"beta_{i + 1}" for i in range(p)])
            plt.title("coefficients values vs iteration")
            plt.show()

            plt.plot(loss)
            plt.xlabel("iteration")
            plt.ylabel("cross-entropy value")
            plt.title("cross-entropy value vs iteration")
            plt.show()
            
        return self

    def validate(self, X, y, measure: "Measure.Type"):
        """
        Validate performance of classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Validation vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
            `n_features` should be consistent with `n_features` of X passed in `.fit`
            method.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        measure : Measure.Type
            Type of measure to be used in validation. Values are limited,
            since Measure.Type is of Enum type.

        Returns
        -------
        float
            Value returned by measure.
        """

        assert isinstance(measure, Measure.Type)

        match measure:
            case Measure.Type.AUC_ROC | Measure.Type.AUC_PR:
                # if we use area under curve measures, we pass probabilities of positive class, not predictions
                return Measure.from_type(measure).get(y, self.predict_proba(X))
            case _:
                # otherwise, pass predicted values
                return Measure.from_type(measure).get(y, self.predict(X))

    def predict_proba(self, X):
        """ Based on X predict probabilities. """

        return LogRegCCD.sigmoid(self.beta0_ + self.__standarize(X) @ self.beta_)

    def predict(self, X):
        """ Based on X predict target values with threshold=0.5. """

        # np.round() can also be used. This version stays though, since we can set threshold explicitly
        return np.where(self.predict_proba(X) > 0.5, 1, 0)

    def get_params(self, deep: bool = False):
        params = dict()
        params["C"] = self.C_
        params["alpha"] = self.alpha_
        return params

    @staticmethod
    def __generate_lambdas(X, y, alpha, max_iter=100, use_weights=True, fit_intercept=True):
        """
        Generates sequence of penalty parameters based on features X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        alpha : float, default=1.0
            The Elastic-Net mixing parameter, with ``0 <= alpha <= 1``.
            Setting ``alpha=0`` is equivalent to having ridge regularization (L2),
            while setting ``alpha=1`` is equivalent to having lasso
            regularization (L1). For ``0 < alpha <1``, the penalty is a
            combination of L1 and L2.

        max_iter : int, default=100
            Number of penalty parameters to generate in a sequence.

        use_weights: bool, default=True
            Specifies if method should use weights optimization.
            If `use_weights` is False, then each sample is given unit weight.

        fit_intercept : bool, default=True
            Specifies if a constant (a.k.a. bias or intercept) should be
            added to the decision function.

        Returns
        -------
        array-like of shape (max_iter,)
            Sequence of generated penalty parameters.
        """

        # retrive X shape
        n, p = X.shape

        # default prior probability
        prior = 0.5
        if fit_intercept:
            # update prior based on y
            prior = y.mean()

        # default weights in optimization algorithm
        weights = 1 / n
        if use_weights:
            # update weights based on prior
            weights = prior * (1 - prior)

        # estimate the smallest value of penalty parameter for which all coefficients are 0.0
        lambda_max = abs(weights * (y - prior) @ X).max()
        # if not ridge regularization (L2) is used
        if alpha != 0:
            # then we can devide by L1 ratio parameter
            lambda_max /= alpha
        # set the smallest value of penalty parameter to 1/1000 of estimated lambda_max
        lambda_min = 0.001 * lambda_max

        # generate sequence of decreasing values of penalty parameters in logarithmic manner
        return np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), max_iter))

    @staticmethod
    def soft_threshold(z, gamma):
        """ Soft-thresholding function. """

        return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)

    @staticmethod
    def sigmoid(x):
        """ Sigmoid function. """

        x = np.where(x > 50, 50, x)
        x = np.where(x < -50, -50, x)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def gaussian(x):
        """ Cumulative Distribution Function (pl. dystrybuanta) of Normal distribution. """

        return norm.cdf(x)

    @staticmethod
    def cross_entropy(y_true, y_scores):
        return -(y_true * np.log(y_scores) + (1 - y_true) * np.log(1 - y_scores)).mean()

    @staticmethod
    def plot(X_train, y_train, X_valid, y_valid, measure_types: ("Measure.Type", list)):
        """ Based on train data, evaluate model with validation data based on passed measures depending on value of penalty parameter. """

        if not isinstance(measure_types, list):
            # if only one measure was passed, treat it like a one-element array
            measure_types = [measure_types]

        # check if all elements of passed measures are measures indeed
        for measure_type in measure_types:
            assert isinstance(measure_type, Measure.Type)

        # generate sequence of optimal lambda with L1 regularization (alpha=1)
        lambdas = LogRegCCD.__generate_lambdas(X_train, y_train, 1)
        # initiate list to store calculated values
        measures_values = [None for _ in range(len(lambdas))]

        for i in range(len(lambdas)):
            # train model based on specified lambda
            model = LogRegCCD(lambda_=lambdas[i]).fit(X_train, y_train)
            # initiate list to store calculated measures for set penalty parameter
            tmp = []
            for measure_type in measure_types:
                # add value returned by each measure
                tmp.append(model.validate(X_valid, y_valid, measure_type))

            # add values of each measure for set penalty parameter to the list
            measures_values[i] = tmp

        plt.plot(lambdas, measures_values)
        plt.xlabel("lambda")
        plt.ylabel("measures values")
        plt.legend([Measure.from_type(measure_type).get_name() for measure_type in measure_types])
        plt.title("Measure values vs penalty parameter value")
        plt.show()

        return

    @staticmethod
    def plot_coefficients(X_train, y_train):
        """ Based on train data plot values of coefficients depending on value of penalty parameter. """

        # generate sequence of optimal lambda with L1 regularization (alpha=1)
        lambdas = LogRegCCD.__generate_lambdas(X_train, y_train, 1)

        # initiate list to store estimated coefficients
        coefficients = [None for _ in range(len(lambdas))]

        for i in range(len(lambdas)):
            # train model based on specified lambda
            model = LogRegCCD(lambda_=lambdas[i]).fit(X_train, y_train)
            # add values of estimated coefficients to the list
            coefficients[i] = model.beta_

        plt.plot(lambdas, coefficients)
        plt.xlabel("lambda")
        plt.ylabel("coefficients values")
        plt.legend([f"beta_{i + 1}" for i in range(len(model.beta_))])
        plt.title("coefficients values vs penalty parameter value")
        plt.show()

        return