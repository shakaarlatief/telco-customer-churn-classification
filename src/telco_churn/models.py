"""Model factories for the Telco Customer Churn project.

The functions in this module return unfitted estimators. Keeping model
construction in one place makes notebooks cleaner and makes later experiments
more reproducible.

Section 04 introduces simple baseline classifiers:

- dummy classifiers based on target-distribution strategies;
- a deterministic EDA-inspired rule classifier.

Section 05 introduces linear classifiers:

- RidgeClassifier as a regularized least-squares classifier;
- LogisticRegression with L2 regularization;
- LogisticRegression with L1 regularization;
- class-weighted logistic regression as a simple imbalance-aware variant.

Section 07 introduces a custom hybrid Naive Bayes classifier for mixed tabular
data:

- Gaussian likelihoods for numeric features;
- Bernoulli likelihoods for one-hot encoded categorical indicators.

The functions intentionally return scikit-learn estimators rather than fitting
them. Fitting happens inside cross-validation pipelines in the notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from telco_churn.config import CATEGORICAL_FEATURES, RANDOM_STATE
from telco_churn.preprocessing import (
    make_dense_unscaled_preprocessor,
    make_native_categorical_preprocessor,
    make_scaled_preprocessor,
    make_unscaled_preprocessor,
)


@dataclass
class EDAInspiredRuleClassifier(BaseEstimator, ClassifierMixin):
    """Simple transparent rule-based churn classifier.

    The classifier assigns one risk point for each manually selected high-risk
    condition and predicts churn when the total risk score is at least
    ``risk_threshold``.

    This estimator is intentionally simple and deterministic. It is useful as a
    bridge between exploratory analysis and learned models.
    """

    risk_threshold: int = 2

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None):
        self.classes_ = np.array([0, 1])
        return self

    def _risk_score(self, X: pd.DataFrame) -> np.ndarray:
        score = np.zeros(len(X), dtype=float)

        score += (X["Contract"] == "Month-to-month").astype(float)
        score += (X["PaymentMethod"] == "Electronic check").astype(float)
        score += (X["InternetService"] == "Fiber optic").astype(float)
        score += (X["OnlineSecurity"] == "No").astype(float)
        score += (X["TechSupport"] == "No").astype(float)

        return score

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        score = self._risk_score(X)
        return (score >= self.risk_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = self._risk_score(X)
        max_score = 5.0
        probability = np.clip(score / max_score, 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])


class HybridGaussianBernoulliNB(BaseEstimator, ClassifierMixin):
    """Naive Bayes classifier for mixed numeric and binary-indicator features.

    The estimator is designed for a preprocessing pipeline that outputs numeric
    features first, followed by one-hot encoded categorical indicator columns.

    For the first ``n_numeric_features`` columns, the model uses Gaussian
    class-conditional densities:

    ``X_j | Y = y ~ Normal(mu_jy, sigma_jy^2)``.

    For the remaining columns, the model uses Bernoulli class-conditional
    probabilities:

    ``P(Z_k = 1 | Y = y) = theta_ky``.

    The resulting joint log score for class ``y`` is:

    ``log P(Y=y)
      + sum_numeric log p(x_j | Y=y)
      + sum_binary log P(z_k | Y=y)``.

    This estimator is useful for tabular churn data because the raw feature
    space is mixed: a small number of continuous numeric variables and many
    categorical variables represented as binary one-hot indicators.

    Parameters
    ----------
    n_numeric_features:
        Number of leading columns that should be treated as Gaussian numeric
        features. All remaining columns are treated as Bernoulli indicators.

    alpha:
        Additive smoothing strength for Bernoulli indicator probabilities. A
        positive value prevents probabilities from becoming exactly zero or one.

    var_smoothing:
        Non-negative variance smoothing used for Gaussian likelihoods. This
        prevents division by zero for nearly constant numeric features.
    """

    def __init__(
        self,
        *,
        n_numeric_features: int,
        alpha: float = 1.0,
        var_smoothing: float = 1e-9,
    ):
        self.n_numeric_features = n_numeric_features
        self.alpha = alpha
        self.var_smoothing = var_smoothing

    @staticmethod
    def _as_dense_float_array(X) -> np.ndarray:
        """Convert dense or sparse input to a two-dimensional float array."""
        if sparse.issparse(X):
            X = X.toarray()

        array = np.asarray(X, dtype=float)

        if array.ndim != 2:
            raise ValueError("HybridGaussianBernoulliNB expects a 2D feature array.")

        return array

    def _split_features(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Split the preprocessed matrix into numeric and Bernoulli blocks."""
        array = self._as_dense_float_array(X)

        if self.n_numeric_features < 0:
            raise ValueError("n_numeric_features must be non-negative.")

        if self.n_numeric_features > array.shape[1]:
            raise ValueError(
                "n_numeric_features cannot exceed the number of columns in X."
            )

        numeric = array[:, : self.n_numeric_features]
        binary = array[:, self.n_numeric_features :]

        return numeric, binary

    def fit(self, X, y):
        """Estimate class priors, Gaussian parameters, and Bernoulli parameters."""
        if self.alpha <= 0:
            raise ValueError("alpha must be strictly positive.")

        if self.var_smoothing < 0:
            raise ValueError("var_smoothing must be non-negative.")

        X_numeric, X_binary = self._split_features(X)
        y_array = np.asarray(y)

        self.classes_, y_encoded = np.unique(y_array, return_inverse=True)
        self.class_count_ = np.bincount(
            y_encoded,
            minlength=len(self.classes_),
        ).astype(float)

        if len(self.classes_) != 2:
            raise ValueError(
                "HybridGaussianBernoulliNB currently supports binary classification."
            )

        if np.any(self.class_count_ == 0):
            raise ValueError("Each class must have at least one observation.")

        n_samples = y_array.shape[0]
        self.class_log_prior_ = np.log(self.class_count_ / n_samples)

        n_classes = len(self.classes_)
        n_numeric = X_numeric.shape[1]
        n_binary = X_binary.shape[1]

        self.theta_ = np.zeros((n_classes, n_numeric), dtype=float)
        self.var_ = np.zeros((n_classes, n_numeric), dtype=float)

        if n_numeric > 0:
            for class_index in range(n_classes):
                class_mask = y_encoded == class_index
                class_numeric = X_numeric[class_mask]
                self.theta_[class_index] = class_numeric.mean(axis=0)
                self.var_[class_index] = class_numeric.var(axis=0)

            global_variance = np.var(X_numeric, axis=0)
            max_global_variance = (
                float(np.max(global_variance)) if global_variance.size else 0.0
            )
            self.epsilon_ = self.var_smoothing * max(max_global_variance, 1.0)
            self.var_ = self.var_ + self.epsilon_
        else:
            self.epsilon_ = 0.0

        self.feature_log_prob_ = np.zeros((n_classes, n_binary), dtype=float)
        self.feature_log_neg_prob_ = np.zeros((n_classes, n_binary), dtype=float)

        if n_binary > 0:
            for class_index in range(n_classes):
                class_mask = y_encoded == class_index
                class_binary = X_binary[class_mask]

                smoothed_count = class_binary.sum(axis=0) + self.alpha
                smoothed_total = self.class_count_[class_index] + 2.0 * self.alpha
                probability = smoothed_count / smoothed_total
                probability = np.clip(probability, 1e-12, 1.0 - 1e-12)

                self.feature_log_prob_[class_index] = np.log(probability)
                self.feature_log_neg_prob_[class_index] = np.log1p(-probability)

        self.n_features_in_ = X_numeric.shape[1] + X_binary.shape[1]
        self.n_numeric_features_ = n_numeric
        self.n_binary_features_ = n_binary

        return self

    def _joint_log_likelihood(self, X) -> np.ndarray:
        """Compute unnormalized class log posterior scores."""
        check_is_fitted(
            self,
            attributes=[
                "classes_",
                "class_log_prior_",
                "theta_",
                "var_",
                "feature_log_prob_",
                "feature_log_neg_prob_",
            ],
        )

        X_numeric, X_binary = self._split_features(X)

        if X_numeric.shape[1] != self.n_numeric_features_:
            raise ValueError("The numeric feature block has an unexpected width.")

        if X_binary.shape[1] != self.n_binary_features_:
            raise ValueError("The Bernoulli feature block has an unexpected width.")

        joint_log_likelihood = np.tile(
            self.class_log_prior_,
            (X_numeric.shape[0], 1),
        )

        if self.n_numeric_features_ > 0:
            for class_index in range(len(self.classes_)):
                mean = self.theta_[class_index]
                variance = self.var_[class_index]
                gaussian_log_prob = -0.5 * np.sum(
                    np.log(2.0 * np.pi * variance)
                    + ((X_numeric - mean) ** 2 / variance),
                    axis=1,
                )
                joint_log_likelihood[:, class_index] += gaussian_log_prob

        if self.n_binary_features_ > 0:
            joint_log_likelihood += (
                X_binary @ self.feature_log_prob_.T
                + (1.0 - X_binary) @ self.feature_log_neg_prob_.T
            )

        return joint_log_likelihood

    def predict_log_proba(self, X) -> np.ndarray:
        """Predict normalized class log probabilities."""
        joint_log_likelihood = self._joint_log_likelihood(X)
        log_normalizer = logsumexp(joint_log_likelihood, axis=1, keepdims=True)
        return joint_log_likelihood - log_normalizer

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        joint_log_likelihood = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]

    def decision_function(self, X) -> np.ndarray:
        """Return the binary class-1 versus class-0 log-score difference."""
        joint_log_likelihood = self._joint_log_likelihood(X)
        return joint_log_likelihood[:, 1] - joint_log_likelihood[:, 0]


def make_eda_inspired_rule_classifier(
    *,
    risk_threshold: int = 2,
) -> EDAInspiredRuleClassifier:
    """Create the EDA-inspired rule classifier."""
    return EDAInspiredRuleClassifier(risk_threshold=risk_threshold)


def make_most_frequent_dummy_classifier() -> DummyClassifier:
    """Create a majority-class dummy classifier."""
    return DummyClassifier(strategy="most_frequent")


def make_prior_probability_dummy_classifier() -> DummyClassifier:
    """Create a prior-probability dummy classifier.

    The hard predictions are the majority class, but predicted probabilities
    equal the empirical class distribution in each training fold.
    """
    return DummyClassifier(strategy="prior")


def make_stratified_dummy_classifier(
    *,
    random_state: int = RANDOM_STATE,
) -> DummyClassifier:
    """Create a stratified random dummy classifier."""
    return DummyClassifier(strategy="stratified", random_state=random_state)


def make_uniform_dummy_classifier(
    *,
    random_state: int = RANDOM_STATE,
) -> DummyClassifier:
    """Create a uniform random dummy classifier."""
    return DummyClassifier(strategy="uniform", random_state=random_state)


def make_ridge_classifier(
    *,
    alpha: float = 1.0,
    class_weight: str | dict[int, float] | None = None,
    random_state: int = RANDOM_STATE,
) -> RidgeClassifier:
    """Create a regularized least-squares linear classifier.

    ``RidgeClassifier`` fits a linear classifier using squared-error logic with
    an L2 penalty. It is used here as the practical version of least-squares
    classification because unregularized least-squares classification can be
    numerically unstable after one-hot encoding.
    """
    return RidgeClassifier(
        alpha=alpha,
        class_weight=class_weight,
        random_state=random_state,
    )


def make_hybrid_gaussian_bernoulli_nb_classifier(
    *,
    n_numeric_features: int,
    alpha: float = 1.0,
    var_smoothing: float = 1e-9,
) -> HybridGaussianBernoulliNB:
    """Create a hybrid Gaussian and Bernoulli Naive Bayes classifier."""
    return HybridGaussianBernoulliNB(
        n_numeric_features=n_numeric_features,
        alpha=alpha,
        var_smoothing=var_smoothing,
    )


def _sklearn_version_at_least(major: int, minor: int) -> bool:
    """Return whether the installed scikit-learn version is at least major.minor.

    This small helper avoids adding a new dependency only to compare versions.
    It is used because newer scikit-learn versions deprecate the old
    LogisticRegression ``penalty`` argument in favour of ``l1_ratio``.
    """
    import sklearn

    version_parts = sklearn.__version__.split(".")[:2]

    try:
        installed_major = int(version_parts[0])
        installed_minor = int(version_parts[1])
    except (IndexError, ValueError):
        return False

    return (installed_major, installed_minor) >= (major, minor)


def make_logistic_regression_classifier(
    *,
    penalty: str = "l2",
    C: float = 1.0,
    class_weight: str | dict[int, float] | None = None,
    solver: str | None = None,
    l1_ratio: float | None = None,
    max_iter: int = 5000,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Create a logistic regression classifier.

    The function supports both older and newer scikit-learn APIs.

    In older scikit-learn versions, regularization is controlled by the
    ``penalty`` argument. In newer versions, the same idea is represented by
    ``l1_ratio``:

    - ``l1_ratio=0`` corresponds to L2 regularization;
    - ``l1_ratio=1`` corresponds to L1 regularization;
    - intermediate values correspond to elastic-net regularization.

    Scikit-learn's ``C`` is inverse regularization strength. Smaller values
    imply stronger regularization.
    """
    if solver is None:
        if penalty == "l1":
            solver = "liblinear"
        elif penalty == "elasticnet":
            solver = "saga"
        else:
            solver = "lbfgs"

    kwargs = {
        "C": C,
        "class_weight": class_weight,
        "solver": solver,
        "max_iter": max_iter,
        "random_state": random_state,
    }

    uses_l1_ratio_api = _sklearn_version_at_least(1, 8)

    if uses_l1_ratio_api:
        if penalty == "l2":
            kwargs["l1_ratio"] = 0.0
        elif penalty == "l1":
            kwargs["l1_ratio"] = 1.0
        elif penalty == "elasticnet":
            kwargs["l1_ratio"] = 0.5 if l1_ratio is None else l1_ratio
        elif penalty in {None, "none"}:
            kwargs["C"] = np.inf
            kwargs["l1_ratio"] = 0.0
        else:
            raise ValueError(f"Unsupported penalty: {penalty!r}")
    else:
        kwargs["penalty"] = penalty
        if penalty == "elasticnet":
            kwargs["l1_ratio"] = 0.5 if l1_ratio is None else l1_ratio

    return LogisticRegression(**kwargs)


def make_l2_logistic_regression_classifier(
    *,
    C: float = 1.0,
    class_weight: str | dict[int, float] | None = None,
    max_iter: int = 5000,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Create L2-regularized logistic regression."""
    return make_logistic_regression_classifier(
        penalty="l2",
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )


def make_l1_logistic_regression_classifier(
    *,
    C: float = 1.0,
    class_weight: str | dict[int, float] | None = None,
    max_iter: int = 5000,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Create L1-regularized logistic regression."""
    return make_logistic_regression_classifier(
        penalty="l1",
        C=C,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=max_iter,
        random_state=random_state,
    )


def make_classifier_pipeline(
    *,
    preprocessor,
    classifier,
) -> Pipeline:
    """Create a preprocessing-plus-classifier pipeline.

    The preprocessor is fitted inside each cross-validation training fold when
    the pipeline is passed to cross-validation utilities. This prevents leakage
    from validation folds into scaling, encoding, or imputation steps.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def make_l2_logistic_regression_pipeline(
    *,
    C: float = 1.0,
    class_weight: str | dict[int, float] | None = None,
    max_iter: int = 5000,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Create a scaled preprocessing plus L2 logistic regression pipeline."""
    return make_classifier_pipeline(
        preprocessor=make_scaled_preprocessor(),
        classifier=make_l2_logistic_regression_classifier(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        ),
    )


def normalize_optional_positive_int(value: object) -> int | None:
    """Normalize optional positive integer hyperparameters recovered from tables.

    Several notebooks select hyperparameters from pandas result rows. Values
    such as ``None`` may therefore return as ``NaN`` and integer values may
    return as floats, for example ``6.0``. Scikit-learn validates tree parameters
    strictly, so reusable factories normalize these values before constructing
    estimators.
    """
    if value is None or pd.isna(value):
        return None
    return int(value)


def make_decision_tree_classifier(
    *,
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_leaf_nodes: int | None = None,
    ccp_alpha: float = 0.0,
    random_state: int = RANDOM_STATE,
) -> DecisionTreeClassifier:
    """Create a decision-tree classifier with normalized hyperparameters."""
    return DecisionTreeClassifier(
        criterion=criterion,
        max_depth=normalize_optional_positive_int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_leaf_nodes=normalize_optional_positive_int(max_leaf_nodes),
        ccp_alpha=float(ccp_alpha),
        random_state=random_state,
    )


def make_decision_tree_pipeline(
    *,
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_leaf_nodes: int | None = None,
    ccp_alpha: float = 0.0,
) -> Pipeline:
    """Create an unscaled preprocessing plus decision-tree pipeline."""
    return make_classifier_pipeline(
        preprocessor=make_unscaled_preprocessor(),
        classifier=make_decision_tree_classifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            ccp_alpha=ccp_alpha,
        ),
    )


def make_bagging_classifier(
    *,
    n_estimators: int,
    max_samples: float,
    base_max_depth: int | None,
    base_min_samples_leaf: int,
    oob_score: bool = False,
    random_state: int = RANDOM_STATE,
) -> BaggingClassifier:
    """Create a bagged decision-tree classifier.

    The constructor supports both newer scikit-learn versions, where the base
    learner argument is called ``estimator``, and older versions, where it is
    called ``base_estimator``.
    """
    base_tree = make_decision_tree_classifier(
        criterion="gini",
        max_depth=base_max_depth,
        min_samples_leaf=base_min_samples_leaf,
        random_state=random_state,
    )

    common_kwargs = {
        "n_estimators": int(n_estimators),
        "max_samples": float(max_samples),
        "bootstrap": True,
        "oob_score": bool(oob_score),
        "n_jobs": -1,
        "random_state": random_state,
    }

    try:
        return BaggingClassifier(estimator=base_tree, **common_kwargs)
    except TypeError:
        return BaggingClassifier(base_estimator=base_tree, **common_kwargs)


def make_bagging_pipeline(
    *,
    n_estimators: int,
    max_samples: float,
    base_max_depth: int | None,
    base_min_samples_leaf: int,
    oob_score: bool = False,
) -> Pipeline:
    """Create an unscaled preprocessing plus bagged-tree pipeline."""
    return make_classifier_pipeline(
        preprocessor=make_unscaled_preprocessor(),
        classifier=make_bagging_classifier(
            n_estimators=n_estimators,
            max_samples=max_samples,
            base_max_depth=base_max_depth,
            base_min_samples_leaf=base_min_samples_leaf,
            oob_score=oob_score,
        ),
    )


def make_random_forest_classifier(
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: str | float,
    oob_score: bool = False,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """Create a random-forest classifier with normalized hyperparameters."""
    return RandomForestClassifier(
        n_estimators=int(n_estimators),
        criterion="gini",
        max_depth=normalize_optional_positive_int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        bootstrap=True,
        oob_score=bool(oob_score),
        n_jobs=-1,
        random_state=random_state,
    )


def make_random_forest_pipeline(
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: str | float,
    oob_score: bool = False,
) -> Pipeline:
    """Create an unscaled preprocessing plus random-forest pipeline."""
    return make_classifier_pipeline(
        preprocessor=make_unscaled_preprocessor(),
        classifier=make_random_forest_classifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            oob_score=oob_score,
        ),
    )


def make_one_hot_boosting_pipeline(classifier) -> Pipeline:
    """Create dense one-hot preprocessing plus a boosting classifier."""
    return make_classifier_pipeline(
        preprocessor=make_dense_unscaled_preprocessor(),
        classifier=classifier,
    )


def make_native_lightgbm_pipeline(classifier) -> Pipeline:
    """Create native-categorical preprocessing plus a LightGBM classifier."""
    return make_classifier_pipeline(
        preprocessor=make_native_categorical_preprocessor(categorical_dtype=True),
        classifier=classifier,
    )


def make_native_catboost_pipeline(classifier) -> Pipeline:
    """Create native-categorical preprocessing plus a CatBoost classifier."""
    return make_classifier_pipeline(
        preprocessor=make_native_categorical_preprocessor(categorical_dtype=False),
        classifier=classifier,
    )


def make_adaboost_classifier(
    *,
    base_depth: int,
    n_estimators: int,
    learning_rate: float,
    random_state: int = RANDOM_STATE,
) -> AdaBoostClassifier:
    """Create an AdaBoost classifier with a shallow decision-tree base learner."""
    base_tree = make_decision_tree_classifier(
        max_depth=int(base_depth),
        random_state=random_state,
    )
    kwargs = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "random_state": random_state,
    }

    try:
        return AdaBoostClassifier(estimator=base_tree, **kwargs)
    except TypeError:
        return AdaBoostClassifier(base_estimator=base_tree, **kwargs)


def make_adaboost_pipeline(
    *,
    base_depth: int,
    n_estimators: int,
    learning_rate: float,
) -> Pipeline:
    """Create a dense one-hot preprocessing plus AdaBoost pipeline."""
    return make_one_hot_boosting_pipeline(
        make_adaboost_classifier(
            base_depth=base_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
    )


def make_gradient_boosting_classifier(
    *,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
    subsample: float,
    random_state: int = RANDOM_STATE,
) -> GradientBoostingClassifier:
    """Create scikit-learn's classical gradient boosting classifier."""
    return GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        subsample=float(subsample),
        random_state=random_state,
    )


def make_gradient_boosting_pipeline(
    *,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
    subsample: float,
) -> Pipeline:
    """Create a dense one-hot preprocessing plus GradientBoosting pipeline."""
    return make_one_hot_boosting_pipeline(
        make_gradient_boosting_classifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
        )
    )


def make_hist_gradient_boosting_classifier(
    *,
    max_iter: int,
    learning_rate: float,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    random_state: int = RANDOM_STATE,
) -> HistGradientBoostingClassifier:
    """Create scikit-learn's histogram gradient boosting classifier."""
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        max_leaf_nodes=int(max_leaf_nodes),
        min_samples_leaf=int(min_samples_leaf),
        l2_regularization=float(l2_regularization),
        early_stopping=False,
        random_state=random_state,
    )


def make_hist_gradient_boosting_pipeline(
    *,
    max_iter: int,
    learning_rate: float,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
) -> Pipeline:
    """Create a dense one-hot preprocessing plus HistGradientBoosting pipeline."""
    return make_one_hot_boosting_pipeline(
        make_hist_gradient_boosting_classifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
        )
    )


def _raise_optional_dependency_error(package_name: str, install_name: str):
    """Raise an informative error for optional modern boosting packages."""
    raise ImportError(
        f"{package_name} is required for this model factory. Install it with "
        f"`pip install {install_name}` in the project environment."
    )


def make_xgboost_classifier(
    *,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_child_weight: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    random_state: int = RANDOM_STATE,
):
    """Create an XGBoost classifier for dense one-hot encoded input."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        _raise_optional_dependency_error("XGBoost", "xgboost")
        raise exc

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        min_child_weight=float(min_child_weight),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_lambda=float(reg_lambda),
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


def make_xgboost_pipeline(
    *,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_child_weight: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
) -> Pipeline:
    """Create a dense one-hot preprocessing plus XGBoost pipeline."""
    return make_one_hot_boosting_pipeline(
        make_xgboost_classifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
        )
    )


def make_lightgbm_classifier(
    *,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    random_state: int = RANDOM_STATE,
):
    """Create a LightGBM classifier for native-categorical input."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        _raise_optional_dependency_error("LightGBM", "lightgbm")
        raise exc

    return LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        num_leaves=int(num_leaves),
        min_child_samples=int(min_child_samples),
        subsample=float(subsample),
        subsample_freq=1,
        colsample_bytree=float(colsample_bytree),
        reg_lambda=float(reg_lambda),
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def make_lightgbm_pipeline(
    *,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
) -> Pipeline:
    """Create a native-categorical preprocessing plus LightGBM pipeline."""
    return make_native_lightgbm_pipeline(
        make_lightgbm_classifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
        )
    )


class CloneSafeCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """Clone-safe CatBoost wrapper for native categorical columns.

    Passing ``cat_features`` directly to ``CatBoostClassifier`` can fail
    scikit-learn's strict clone validation in some package combinations because
    CatBoost normalizes that constructor parameter internally. This wrapper keeps
    the constructor parameters simple and passes categorical feature names during
    ``fit`` instead.
    """

    def __init__(
        self,
        *,
        iterations: int = 100,
        learning_rate: float = 0.1,
        depth: int = 3,
        l2_leaf_reg: float = 3.0,
        random_state: int = RANDOM_STATE,
        thread_count: int = -1,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.thread_count = thread_count

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        """Fit CatBoost with categorical feature names supplied at fit time."""
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            _raise_optional_dependency_error("CatBoost", "catboost")
            raise exc

        self.model_ = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="PRAUC",
            iterations=int(self.iterations),
            learning_rate=float(self.learning_rate),
            depth=int(self.depth),
            l2_leaf_reg=float(self.l2_leaf_reg),
            random_seed=int(self.random_state),
            verbose=False,
            allow_writing_files=False,
            thread_count=int(self.thread_count),
        )
        self.model_.fit(X, y, cat_features=list(CATEGORICAL_FEATURES))
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels using the fitted CatBoost model."""
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict(X)).reshape(-1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using the fitted CatBoost model."""
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Expose CatBoost feature importances after fitting."""
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.feature_importances_, dtype=float)


def make_catboost_classifier(
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    l2_leaf_reg: float,
    random_state: int = RANDOM_STATE,
    thread_count: int = -1,
) -> CloneSafeCatBoostClassifier:
    """Create the clone-safe CatBoost classifier wrapper."""
    return CloneSafeCatBoostClassifier(
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        l2_leaf_reg=float(l2_leaf_reg),
        random_state=random_state,
        thread_count=thread_count,
    )


def make_catboost_pipeline(
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    l2_leaf_reg: float,
) -> Pipeline:
    """Create a native-categorical preprocessing plus clone-safe CatBoost pipeline."""
    return make_native_catboost_pipeline(
        make_catboost_classifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
        )
    )
