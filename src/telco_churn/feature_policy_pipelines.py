"""Pipeline adapters that combine feature policies with model representations.

The final-comparison registry treats a feature policy as part of the candidate
procedure.  A policy must therefore run inside the fitted scikit-learn pipeline,
before its representation-specific imputation, encoding, scaling, or native
categorical conversion is learned.  This module provides the adapter layer between
``FeaturePolicyTransformer`` and the representations needed by the core candidate
families.

The adapters deliberately leave the historical preprocessing factories unchanged.
Those factories document and reproduce the earlier individual model workflows.  The
functions here are dedicated to the resumable final-comparison system, whose feature
schemas may contain engineered numeric and categorical columns in addition to the
original raw table.
"""

from __future__ import annotations

from typing import Final, Literal, Mapping

from imblearn.pipeline import Pipeline as ImblearnPipeline
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from telco_churn.feature_selection import (
    FEATURE_SELECTION_NONE,
    FeatureSelectionPolicyId,
    make_feature_selector,
    validate_feature_selection_policy_id,
)
from telco_churn.feature_policies import (
    FEATURE_POLICY_RAW,
    FeaturePolicyId,
    FeaturePolicyTransformer,
    feature_policy_categorical_features,
    feature_policy_numeric_features,
    validate_feature_policy_id,
)
from telco_churn.imbalance_policies import (
    BalancedSampleWeightClassifier,
    FeaturePolicySamplerImputer,
    IMBALANCE_CLASS_WEIGHT_BALANCED,
    IMBALANCE_NONE,
    IMBALANCE_RANDOM_OVERSAMPLING,
    IMBALANCE_RANDOM_UNDERSAMPLING,
    IMBALANCE_SMOTENC,
    ImbalancePolicyId,
    make_random_resampler,
    make_smotenc_resampler,
    normalize_imbalance_parameters,
    validate_imbalance_policy_id,
)


FeatureRepresentation = Literal[
    "sparse_scaled_one_hot",
    "sparse_unscaled_one_hot",
    "dense_scaled_one_hot",
    "dense_unscaled_one_hot",
    "native_categorical_dtype",
    "native_categorical_string",
]

REPRESENTATION_SPARSE_SCALED: Final[FeatureRepresentation] = "sparse_scaled_one_hot"
REPRESENTATION_SPARSE_UNSCALED: Final[FeatureRepresentation] = "sparse_unscaled_one_hot"
REPRESENTATION_DENSE_SCALED: Final[FeatureRepresentation] = "dense_scaled_one_hot"
REPRESENTATION_DENSE_UNSCALED: Final[FeatureRepresentation] = "dense_unscaled_one_hot"
REPRESENTATION_NATIVE_CATEGORICAL_DTYPE: Final[FeatureRepresentation] = (
    "native_categorical_dtype"
)
REPRESENTATION_NATIVE_CATEGORICAL_STRING: Final[FeatureRepresentation] = (
    "native_categorical_string"
)


class FeaturePolicyPipelineError(ValueError):
    """Raised when a feature-policy representation is invalid or inconsistent."""


def _make_one_hot_encoder(*, dense: bool) -> OneHotEncoder:
    """Create a version-compatible encoder with a requested sparsity contract."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense)


def make_feature_policy_one_hot_preprocessor(
    policy_id: FeaturePolicyId,
    *,
    scale_numeric: bool,
    dense: bool,
) -> ColumnTransformer:
    """Create representation preprocessing for one fixed post-policy schema.

    The column names are obtained from the declared policy contract, not inferred
    from an arbitrary validation frame.  Consequently, a cloned pipeline has the
    same schema definition in every CV fold while imputation, scaling, and one-hot
    categories are still learned exclusively from the fold's training rows.
    """
    policy_id = validate_feature_policy_id(policy_id)
    numeric_features = feature_policy_numeric_features(policy_id)
    categorical_features = feature_policy_categorical_features(policy_id)

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder(dense=dense)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


class FeaturePolicyNativeCategoricalPreprocessor(BaseEstimator, TransformerMixin):
    """Impute a policy-expanded table while preserving native categorical columns.

    LightGBM and CatBoost receive DataFrames rather than anonymous encoded arrays.
    The preceding ``FeaturePolicyTransformer`` has already created any engineered
    columns.  This transformer then performs fold-local numerical median and
    categorical mode imputation on the exact feature-policy schema and returns the
    categorical block either as pandas ``category`` dtype or as strings.
    """

    def __init__(
        self,
        *,
        numeric_features: tuple[str, ...],
        categorical_features: tuple[str, ...],
        categorical_dtype: bool,
    ) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.categorical_dtype = categorical_dtype

    def fit(self, X: pd.DataFrame, y=None):
        """Fit fold-local native-representation imputers."""
        X_frame = self._validate_input(X)
        self.numeric_imputer_ = SimpleImputer(strategy="median")
        self.categorical_imputer_ = SimpleImputer(strategy="most_frequent")
        self.numeric_imputer_.fit(X_frame.loc[:, list(self.numeric_features)])
        self.categorical_imputer_.fit(X_frame.loc[:, list(self.categorical_features)])
        self.feature_names_in_ = np.asarray(
            [*self.numeric_features, *self.categorical_features], dtype=object
        )
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return an imputed DataFrame that preserves the declared feature schema."""
        check_is_fitted(self, ("numeric_imputer_", "categorical_imputer_"))
        X_frame = self._validate_input(X)

        numeric = pd.DataFrame(
            self.numeric_imputer_.transform(X_frame.loc[:, list(self.numeric_features)]),
            columns=self.numeric_features,
            index=X_frame.index,
        )
        categorical = pd.DataFrame(
            self.categorical_imputer_.transform(
                X_frame.loc[:, list(self.categorical_features)]
            ),
            columns=self.categorical_features,
            index=X_frame.index,
        )
        for column in self.categorical_features:
            if self.categorical_dtype:
                categorical[column] = categorical[column].astype("category")
            else:
                categorical[column] = categorical[column].astype(str)

        return pd.concat([numeric, categorical], axis=1).loc[
            :, [*self.numeric_features, *self.categorical_features]
        ]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the fixed native-column output schema."""
        return np.asarray([*self.numeric_features, *self.categorical_features], dtype=object)

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate that a feature-policy transformer produced the declared columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeaturePolicyNativeCategoricalPreprocessor expects a pandas DataFrame."
            )
        expected = [*self.numeric_features, *self.categorical_features]
        missing = [column for column in expected if column not in X.columns]
        if missing:
            raise FeaturePolicyPipelineError(
                "Feature-policy output is missing declared native categorical columns: "
                f"{missing!r}."
            )
        return X.loc[:, expected].copy()


def make_feature_policy_native_categorical_preprocessor(
    policy_id: FeaturePolicyId,
    *,
    categorical_dtype: bool,
) -> FeaturePolicyNativeCategoricalPreprocessor:
    """Create fold-safe native categorical preprocessing for one feature policy."""
    policy_id = validate_feature_policy_id(policy_id)
    return FeaturePolicyNativeCategoricalPreprocessor(
        numeric_features=tuple(feature_policy_numeric_features(policy_id)),
        categorical_features=tuple(feature_policy_categorical_features(policy_id)),
        categorical_dtype=bool(categorical_dtype),
    )


class CloneSafeFeaturePolicyCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """Clone-safe CatBoost wrapper with a policy-dependent categorical column list.

    The existing historical CatBoost wrapper fixes the original raw categorical
    feature list.  F1 and F2 additionally contain the predeclared
    ``f1_contract_x_payment_method`` categorical interaction, so the final-comparison
    pipeline needs a wrapper whose categorical names are immutable constructor
    parameters.  Supplying them during ``fit`` avoids CatBoost constructor parameter
    normalization problems with scikit-learn cloning.
    """

    def __init__(
        self,
        *,
        iterations: int,
        learning_rate: float,
        depth: int,
        l2_leaf_reg: float,
        categorical_features: tuple[str, ...],
        random_state: int,
        thread_count: int = 1,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.thread_count = thread_count

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight=None,
    ):
        """Fit CatBoost with policy-specific categorical feature names."""
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "CatBoost is required for the final-comparison CatBoost candidate."
            ) from exc

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
        fit_kwargs = {"cat_features": list(self.categorical_features)}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float)
        self.model_.fit(X, y, **fit_kwargs)
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class labels after policy-aware native preprocessing."""
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict(X)).reshape(-1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities after policy-aware native preprocessing."""
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict_proba(X), dtype=float)


class CloneSafeFeaturePolicyTabNetClassifier(ClassifierMixin, BaseEstimator):
    """Clone-safe TabNet wrapper with fold-local categorical integer mappings."""

    def __init__(
        self,
        *,
        n_d: int,
        n_a: int,
        n_steps: int,
        gamma: float,
        cat_emb_dim: int,
        n_independent: int,
        n_shared: int,
        lambda_sparse: float,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        patience: int,
        batch_size: int,
        virtual_batch_size: int,
        mask_type: str,
        numeric_features: tuple[str, ...],
        categorical_features: tuple[str, ...],
        random_state: int,
        device_name: str = "cpu",
        num_workers: int = 0,
        verbose: int = 0,
    ) -> None:
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.cat_emb_dim = cat_emb_dim
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.device_name = device_name
        self.num_workers = num_workers
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight=None,
    ):
        """Fit TabNet with categorical maps learned only from this fitting fold."""
        try:
            import torch
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError as exc:
            raise ImportError(
                "pytorch-tabnet is required for C24_TABNET. "
                "Install the project requirements."
            ) from exc

        X_frame = self._validate_input(X)
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(X_frame):
            raise FeaturePolicyPipelineError(
                "TabNet requires a one-dimensional target aligned with X."
            )

        self.category_mappings_ = self._fit_category_mappings(X_frame)
        self.cat_idxs_ = list(
            range(
                len(self.numeric_features),
                len(self.numeric_features) + len(self.categorical_features),
            )
        )
        self.cat_dims_ = [
            len(self.category_mappings_[column]) + 1
            for column in self.categorical_features
        ]
        self.feature_names_in_ = np.asarray(
            [*self.numeric_features, *self.categorical_features],
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)
        X_encoded = self._transform_with_mappings(X_frame)

        weights = 0
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != (len(target),):
                raise FeaturePolicyPipelineError(
                    "TabNet sample_weight must be a one-dimensional array aligned with X."
                )
            if not np.isfinite(weights).all():
                raise FeaturePolicyPipelineError("TabNet sample_weight must be finite.")

        fit_X, eval_X, fit_y, eval_y, fit_weights = self._make_fit_split(
            X_encoded,
            target,
            weights,
        )
        fit_kwargs = {
            "weights": fit_weights,
            "max_epochs": int(self.max_epochs),
            "patience": int(self.patience),
            "batch_size": int(self.batch_size),
            "virtual_batch_size": int(self.virtual_batch_size),
            "num_workers": int(self.num_workers),
            "drop_last": False,
            "pin_memory": False,
            "compute_importance": False,
        }
        if eval_X is not None:
            fit_kwargs["eval_set"] = [(eval_X, eval_y)]
            fit_kwargs["eval_name"] = ["validation"]
            fit_kwargs["eval_metric"] = ["auc"]

        self.model_ = TabNetClassifier(
            n_d=int(self.n_d),
            n_a=int(self.n_a),
            n_steps=int(self.n_steps),
            gamma=float(self.gamma),
            cat_idxs=list(self.cat_idxs_),
            cat_dims=list(self.cat_dims_),
            cat_emb_dim=int(self.cat_emb_dim),
            n_independent=int(self.n_independent),
            n_shared=int(self.n_shared),
            lambda_sparse=float(self.lambda_sparse),
            seed=int(self.random_state),
            verbose=int(self.verbose),
            optimizer_params={
                "lr": float(self.learning_rate),
                "weight_decay": float(self.weight_decay),
            },
            mask_type=str(self.mask_type),
            device_name=str(self.device_name),
        )

        previous_threads = None
        try:
            previous_threads = int(torch.get_num_threads())
            torch.set_num_threads(1)
        except (AttributeError, RuntimeError, ValueError):
            previous_threads = None
        try:
            try:
                self.model_.fit(fit_X, fit_y, **fit_kwargs)
            except (TypeError, ValueError) as exc:
                if sample_weight is not None:
                    raise FeaturePolicyPipelineError(
                        "TabNet rejected the per-row sample_weight array passed "
                        "through I1_CLASS_WEIGHT_BALANCED."
                    ) from exc
                raise
        finally:
            if previous_threads is not None:
                try:
                    torch.set_num_threads(previous_threads)
                except (RuntimeError, ValueError):
                    pass

        classes = getattr(self.model_, "classes_", None)
        if classes is None:
            raise FeaturePolicyPipelineError("Fitted TabNet model did not expose classes_.")
        self.classes_ = np.asarray(classes)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary labels using fixed fold-local categorical mappings."""
        check_is_fitted(self, ("model_", "category_mappings_"))
        return np.asarray(self.model_.predict(self._transform_with_mappings(X)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using fixed fold-local categorical mappings."""
        check_is_fitted(self, ("model_", "category_mappings_"))
        probabilities = np.asarray(
            self.model_.predict_proba(self._transform_with_mappings(X)),
            dtype=float,
        )
        if probabilities.ndim != 2:
            raise FeaturePolicyPipelineError(
                f"TabNet returned invalid probability shape {probabilities.shape!r}."
            )
        return probabilities

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and order the native-categorical frame expected by TabNet."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TabNet native preprocessing expects a pandas DataFrame.")
        expected = [*self.numeric_features, *self.categorical_features]
        missing = [column for column in expected if column not in X.columns]
        if missing:
            raise FeaturePolicyPipelineError(
                f"TabNet input is missing declared columns: {missing!r}."
            )
        return X.loc[:, expected].copy()

    def _fit_category_mappings(self, X: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Learn deterministic fold-local mappings, reserving zero for unknowns."""
        mappings: dict[str, dict[str, int]] = {}
        for column in self.categorical_features:
            values = X[column].astype("string").dropna().astype(str)
            categories = sorted(values.unique().tolist())
            mappings[column] = {
                category: index
                for index, category in enumerate(categories, start=1)
            }
        return mappings

    def _transform_with_mappings(self, X: pd.DataFrame) -> np.ndarray:
        """Encode new data with fitted mappings without learning new categories."""
        X_frame = self._validate_input(X)
        numeric = X_frame.loc[:, list(self.numeric_features)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        numeric_array = numeric.to_numpy(dtype=np.float32)
        if not np.isfinite(numeric_array).all():
            raise FeaturePolicyPipelineError("TabNet numeric inputs must be finite.")

        encoded_columns = []
        for column in self.categorical_features:
            mapping = self.category_mappings_[column]
            encoded = (
                X_frame[column]
                .astype("string")
                .astype(str)
                .map(mapping)
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
            )
            encoded_columns.append(encoded.reshape(-1, 1))
        if encoded_columns:
            categorical_array = np.hstack(encoded_columns).astype(np.float32)
            return np.hstack([numeric_array, categorical_array]).astype(np.float32)
        return numeric_array.astype(np.float32)

    def _make_fit_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, object]:
        """Create a fold-internal validation split for TabNet early stopping."""
        unique, counts = np.unique(y, return_counts=True)
        can_split = len(y) >= 40 and len(unique) == 2 and np.min(counts) >= 2
        if not can_split:
            return X, None, y, None, weights

        indices = np.arange(len(y))
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=0.15,
            stratify=y,
            random_state=int(self.random_state),
        )
        split_weights = (
            0
            if isinstance(weights, int)
            else np.asarray(weights, dtype=float)[train_indices]
        )
        return (
            X[train_indices],
            X[validation_indices],
            y[train_indices],
            y[validation_indices],
            split_weights,
        )


class CloneSafeFeaturePolicyFTTransformerClassifier(ClassifierMixin, BaseEstimator):
    """Clone-safe FT-Transformer wrapper with fold-local numeric/category state."""

    def __init__(
        self,
        *,
        n_blocks: int,
        d_block: int,
        attention_n_heads: int,
        attention_dropout: float,
        ffn_d_hidden_multiplier: float,
        ffn_dropout: float,
        residual_dropout: float,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        patience: int,
        batch_size: int,
        numeric_features: tuple[str, ...],
        categorical_features: tuple[str, ...],
        random_state: int,
        device_name: str = "cpu",
        num_workers: int = 0,
    ) -> None:
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.device_name = device_name
        self.num_workers = num_workers

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight=None,
    ):
        """Fit FT-Transformer with all preprocessing learned from this fitting fold."""
        try:
            import inspect

            import torch
            from torch.utils.data import DataLoader, TensorDataset
            from rtdl_revisiting_models import FTTransformer, FTTransformerBackbone
        except ImportError as exc:
            raise ImportError(
                "rtdl_revisiting_models and torch are required for C25_FT_TRANSFORMER. "
                "Install the project requirements."
            ) from exc

        if str(self.device_name) != "cpu":
            raise FeaturePolicyPipelineError("C25_FT_TRANSFORMER supports CPU only.")
        if int(self.num_workers) != 0:
            raise FeaturePolicyPipelineError(
                "C25_FT_TRANSFORMER requires num_workers=0."
            )

        X_frame = self._validate_input(X)
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(X_frame):
            raise FeaturePolicyPipelineError(
                "FT-Transformer requires a one-dimensional target aligned with X."
            )
        if not np.isin(target, [0, 1]).all():
            raise FeaturePolicyPipelineError(
                "FT-Transformer requires binary labels encoded as 0 and 1."
            )
        target_float = target.astype(np.float32)

        weights = np.ones(len(target_float), dtype=np.float32)
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=np.float32)
            if weights.shape != (len(target_float),):
                raise FeaturePolicyPipelineError(
                    "FT-Transformer sample_weight must be a one-dimensional array "
                    "aligned with X."
                )
            if not np.isfinite(weights).all():
                raise FeaturePolicyPipelineError(
                    "FT-Transformer sample_weight must be finite."
                )
            if np.any(weights < 0.0):
                raise FeaturePolicyPipelineError(
                    "FT-Transformer sample_weight must be non-negative."
                )

        self._fit_numeric_state(X_frame)
        self.category_mappings_ = self._fit_category_mappings(X_frame)
        self.cat_cardinalities_ = [
            len(self.category_mappings_[column]) + 1
            for column in self.categorical_features
        ]
        self.feature_names_in_ = np.asarray(
            [*self.numeric_features, *self.categorical_features],
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)

        x_cont = self._transform_numeric(X_frame)
        x_cat = self._transform_categorical(X_frame)
        train_indices, validation_indices = self._make_fit_split_indices(target)

        torch.manual_seed(int(self.random_state))
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                pass
        except (AttributeError, RuntimeError):
            pass

        backbone_kwargs = self._make_backbone_kwargs(
            FTTransformer=FTTransformer,
            FTTransformerBackbone=FTTransformerBackbone,
            inspect_module=inspect,
        )
        self.model_ = FTTransformer(
            n_cont_features=len(self.numeric_features),
            cat_cardinalities=list(self.cat_cardinalities_),
            **backbone_kwargs,
        )
        self.model_.to("cpu")
        optimizer = torch.optim.AdamW(
            self.model_.make_parameter_groups(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        train_dataset = TensorDataset(
            torch.as_tensor(x_cont[train_indices], dtype=torch.float32),
            torch.as_tensor(x_cat[train_indices], dtype=torch.int64),
            torch.as_tensor(target_float[train_indices], dtype=torch.float32),
            torch.as_tensor(weights[train_indices], dtype=torch.float32),
        )
        generator = torch.Generator()
        generator.manual_seed(int(self.random_state))
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.batch_size),
            shuffle=True,
            num_workers=0,
            generator=generator,
            pin_memory=False,
        )

        validation_tensors = None
        if validation_indices.size:
            validation_tensors = (
                torch.as_tensor(x_cont[validation_indices], dtype=torch.float32),
                torch.as_tensor(x_cat[validation_indices], dtype=torch.int64),
                torch.as_tensor(target_float[validation_indices], dtype=torch.float32),
            )

        best_loss = np.inf
        best_epoch = -1
        best_state: dict[str, object] | None = None
        epochs_without_improvement = 0

        previous_threads = None
        try:
            previous_threads = int(torch.get_num_threads())
            torch.set_num_threads(1)
        except (AttributeError, RuntimeError, ValueError):
            previous_threads = None
        try:
            for epoch in range(int(self.max_epochs)):
                self.model_.train()
                epoch_losses: list[float] = []
                for batch_cont, batch_cat, batch_y, batch_weight in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = self._forward_logits(self.model_, batch_cont, batch_cat)
                    losses = loss_fn(logits, batch_y)
                    weight_sum = batch_weight.sum().clamp_min(1e-12)
                    loss = (losses * batch_weight).sum() / weight_sum
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(float(loss.detach().cpu().item()))

                if validation_tensors is not None:
                    monitored_loss = self._validation_loss(
                        torch,
                        loss_fn,
                        *validation_tensors,
                    )
                else:
                    monitored_loss = float(np.mean(epoch_losses))

                if monitored_loss < best_loss - 1e-6:
                    best_loss = monitored_loss
                    best_epoch = epoch
                    best_state = {
                        name: parameter.detach().cpu().clone()
                        for name, parameter in self.model_.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= int(self.patience):
                        break
        finally:
            if previous_threads is not None:
                try:
                    torch.set_num_threads(previous_threads)
                except (RuntimeError, ValueError):
                    pass

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.best_epoch_ = int(best_epoch)
        self.best_validation_loss_ = float(best_loss)
        self.device_ = "cpu"
        self.num_workers_ = 0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary labels from class-one probabilities."""
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return two-column class probabilities from FT-Transformer logits."""
        check_is_fitted(self, ("model_", "category_mappings_", "numeric_mean_"))
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for fitted C25_FT_TRANSFORMER predictions."
            ) from exc

        X_frame = self._validate_input(X)
        x_cont = torch.as_tensor(self._transform_numeric(X_frame), dtype=torch.float32)
        x_cat = torch.as_tensor(self._transform_categorical(X_frame), dtype=torch.int64)
        self.model_.eval()
        with torch.no_grad():
            logits = self._forward_logits(self.model_, x_cont, x_cat)
            class_one = torch.sigmoid(logits).detach().cpu().numpy().astype(float)
        if class_one.shape != (len(X_frame),):
            raise FeaturePolicyPipelineError(
                f"FT-Transformer returned invalid probability shape {class_one.shape!r}."
            )
        if not np.isfinite(class_one).all():
            raise FeaturePolicyPipelineError(
                "FT-Transformer returned non-finite probabilities."
            )
        class_one = np.clip(class_one, 0.0, 1.0)
        return np.column_stack([1.0 - class_one, class_one])

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and order the native-categorical frame expected by FT-Transformer."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FT-Transformer native preprocessing expects a pandas DataFrame.")
        expected = [*self.numeric_features, *self.categorical_features]
        missing = [column for column in expected if column not in X.columns]
        if missing:
            raise FeaturePolicyPipelineError(
                f"FT-Transformer input is missing declared columns: {missing!r}."
            )
        return X.loc[:, expected].copy()

    def _fit_numeric_state(self, X: pd.DataFrame) -> None:
        """Learn fold-local numeric centering and scaling state."""
        if not self.numeric_features:
            self.numeric_mean_ = np.asarray([], dtype=np.float32)
            self.numeric_scale_ = np.asarray([], dtype=np.float32)
            return
        numeric = X.loc[:, list(self.numeric_features)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        values = numeric.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise FeaturePolicyPipelineError("FT-Transformer numeric inputs must be finite.")
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
        self.numeric_mean_ = mean
        self.numeric_scale_ = scale

    def _transform_numeric(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted numeric scaling without refitting on new rows."""
        if not self.numeric_features:
            return np.empty((len(X), 0), dtype=np.float32)
        numeric = X.loc[:, list(self.numeric_features)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        values = numeric.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise FeaturePolicyPipelineError("FT-Transformer numeric inputs must be finite.")
        return ((values - self.numeric_mean_) / self.numeric_scale_).astype(np.float32)

    def _fit_category_mappings(self, X: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Learn deterministic fold-local mappings, reserving zero for unknowns."""
        mappings: dict[str, dict[str, int]] = {}
        for column in self.categorical_features:
            values = X[column].astype("string").dropna().astype(str)
            categories = sorted(values.unique().tolist())
            mappings[column] = {
                category: index
                for index, category in enumerate(categories, start=1)
            }
        return mappings

    def _transform_categorical(self, X: pd.DataFrame) -> np.ndarray:
        """Encode new rows with fitted mappings without learning new categories."""
        if not self.categorical_features:
            return np.empty((len(X), 0), dtype=np.int64)
        encoded_columns = []
        for column in self.categorical_features:
            mapping = self.category_mappings_[column]
            encoded = (
                X[column]
                .astype("string")
                .astype(str)
                .map(mapping)
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
            )
            encoded_columns.append(encoded.reshape(-1, 1))
        return np.hstack(encoded_columns).astype(np.int64)

    def _make_fit_split_indices(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a fold-internal validation split for early stopping."""
        indices = np.arange(len(y))
        unique, counts = np.unique(y, return_counts=True)
        can_split = len(y) >= 40 and len(unique) == 2 and np.min(counts) >= 2
        if not can_split:
            return indices, np.asarray([], dtype=int)
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=0.15,
            stratify=y,
            random_state=int(self.random_state),
        )
        return train_indices, validation_indices

    def _make_backbone_kwargs(
        self,
        *,
        FTTransformer,
        FTTransformerBackbone,
        inspect_module,
    ) -> dict[str, object]:
        """Create and validate local-API FT-Transformer backbone kwargs."""
        backbone_kwargs = dict(
            FTTransformer.get_default_kwargs(n_blocks=int(self.n_blocks))
        )
        overrides = {
            "d_out": 1,
            "d_block": int(self.d_block),
            "attention_n_heads": int(self.attention_n_heads),
            "attention_dropout": float(self.attention_dropout),
            "ffn_d_hidden_multiplier": float(self.ffn_d_hidden_multiplier),
            "ffn_dropout": float(self.ffn_dropout),
            "residual_dropout": float(self.residual_dropout),
        }
        accepted_backbone = set(inspect_module.signature(FTTransformerBackbone).parameters)
        unexpected = sorted(set(overrides).difference(accepted_backbone))
        if unexpected:
            raise FeaturePolicyPipelineError(
                "Local FTTransformerBackbone does not accept planned arguments: "
                f"{unexpected!r}."
            )
        backbone_kwargs.update(overrides)
        accepted_constructor = set(inspect_module.signature(FTTransformer).parameters)
        accepted = accepted_backbone.union({"_is_default"})
        unexpected_final = sorted(set(backbone_kwargs).difference(accepted))
        if unexpected_final:
            raise FeaturePolicyPipelineError(
                "Local FTTransformer API does not accept planned constructor arguments: "
                f"{unexpected_final!r}; constructor parameters are "
                f"{sorted(accepted_constructor)!r}."
            )
        if int(self.d_block) % int(self.attention_n_heads) != 0:
            raise FeaturePolicyPipelineError(
                "FT-Transformer d_block must be divisible by attention_n_heads."
            )
        return backbone_kwargs

    def _forward_logits(self, model, x_cont, x_cat):
        """Run the local FTTransformer forward API and return one logit per row."""
        x_cont_arg = None if x_cont.shape[1] == 0 else x_cont
        x_cat_arg = None if x_cat.shape[1] == 0 else x_cat
        output = model(x_cont_arg, x_cat_arg)
        if output.ndim == 2 and output.shape[1] == 1:
            return output[:, 0]
        if output.ndim == 1:
            return output
        raise FeaturePolicyPipelineError(
            f"FT-Transformer returned invalid logit shape {tuple(output.shape)!r}."
        )

    def _validation_loss(self, torch, loss_fn, x_cont, x_cat, y_true) -> float:
        """Evaluate unweighted validation loss for early stopping."""
        self.model_.eval()
        with torch.no_grad():
            logits = self._forward_logits(self.model_, x_cont, x_cat)
            loss = loss_fn(logits, y_true).mean()
        return float(loss.detach().cpu().item())


class CloneSafeFeaturePolicyTabMClassifier(ClassifierMixin, BaseEstimator):
    """Clone-safe TabM wrapper with fold-local numeric/category state."""

    def __init__(
        self,
        *,
        arch_type: str,
        n_blocks: int,
        d_block: int,
        dropout: float,
        k: int,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        patience: int,
        batch_size: int,
        numeric_features: tuple[str, ...],
        categorical_features: tuple[str, ...],
        random_state: int,
        activation: str = "ReLU",
        start_scaling_init: str = "random-signs",
        device_name: str = "cpu",
        num_workers: int = 0,
    ) -> None:
        self.arch_type = arch_type
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.k = k
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.activation = activation
        self.start_scaling_init = start_scaling_init
        self.device_name = device_name
        self.num_workers = num_workers

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight=None,
    ):
        """Fit TabM with preprocessing learned only from this fitting fold."""
        try:
            import inspect

            import torch
            from torch.utils.data import DataLoader, TensorDataset
            from tabm import TabM, make_tabm_backbone
        except ImportError as exc:
            raise ImportError(
                "tabm and torch are required for C26_TABM. Install the project "
                "requirements."
            ) from exc

        if str(self.device_name) != "cpu":
            raise FeaturePolicyPipelineError("C26_TABM supports CPU only.")
        if int(self.num_workers) != 0:
            raise FeaturePolicyPipelineError("C26_TABM requires num_workers=0.")

        X_frame = self._validate_input(X)
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(X_frame):
            raise FeaturePolicyPipelineError(
                "TabM requires a one-dimensional target aligned with X."
            )
        if not np.isin(target, [0, 1]).all():
            raise FeaturePolicyPipelineError(
                "TabM requires binary labels encoded as 0 and 1."
            )
        target_float = target.astype(np.float32)

        weights = np.ones(len(target_float), dtype=np.float32)
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=np.float32)
            if weights.shape != (len(target_float),):
                raise FeaturePolicyPipelineError(
                    "TabM sample_weight must be a one-dimensional array aligned with X."
                )
            if not np.isfinite(weights).all():
                raise FeaturePolicyPipelineError("TabM sample_weight must be finite.")
            if np.any(weights < 0.0):
                raise FeaturePolicyPipelineError(
                    "TabM sample_weight must be non-negative."
                )

        self._fit_numeric_state(X_frame)
        self.category_mappings_ = self._fit_category_mappings(X_frame)
        self.cat_cardinalities_ = [
            len(self.category_mappings_[column]) + 1
            for column in self.categorical_features
        ]
        self.feature_names_in_ = np.asarray(
            [*self.numeric_features, *self.categorical_features],
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)

        x_num = self._transform_numeric(X_frame)
        x_cat = self._transform_categorical(X_frame)
        train_indices, validation_indices = self._make_fit_split_indices(target)

        torch.manual_seed(int(self.random_state))
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                pass
        except (AttributeError, RuntimeError):
            pass

        model_kwargs = self._make_model_kwargs(
            TabM=TabM,
            make_tabm_backbone=make_tabm_backbone,
            inspect_module=inspect,
        )
        self.model_ = TabM.make(**model_kwargs)
        self.model_.to("cpu")
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        train_dataset = TensorDataset(
            torch.as_tensor(x_num[train_indices], dtype=torch.float32),
            torch.as_tensor(x_cat[train_indices], dtype=torch.int64),
            torch.as_tensor(target_float[train_indices], dtype=torch.float32),
            torch.as_tensor(weights[train_indices], dtype=torch.float32),
        )
        generator = torch.Generator()
        generator.manual_seed(int(self.random_state))
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.batch_size),
            shuffle=True,
            num_workers=0,
            generator=generator,
            pin_memory=False,
        )

        validation_tensors = None
        if validation_indices.size:
            validation_tensors = (
                torch.as_tensor(x_num[validation_indices], dtype=torch.float32),
                torch.as_tensor(x_cat[validation_indices], dtype=torch.int64),
                torch.as_tensor(target_float[validation_indices], dtype=torch.float32),
            )

        best_loss = np.inf
        best_epoch = -1
        best_state: dict[str, object] | None = None
        epochs_without_improvement = 0

        previous_threads = None
        try:
            previous_threads = int(torch.get_num_threads())
            torch.set_num_threads(1)
        except (AttributeError, RuntimeError, ValueError):
            previous_threads = None
        try:
            for epoch in range(int(self.max_epochs)):
                self.model_.train()
                epoch_losses: list[float] = []
                for batch_num, batch_cat, batch_y, batch_weight in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = self._forward_logits(self.model_, batch_num, batch_cat)
                    per_row_loss = self._per_row_loss(loss_fn, logits, batch_y)
                    weight_sum = batch_weight.sum().clamp_min(1e-12)
                    loss = (per_row_loss * batch_weight).sum() / weight_sum
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(float(loss.detach().cpu().item()))

                if validation_tensors is not None:
                    monitored_loss = self._validation_loss(
                        torch,
                        loss_fn,
                        *validation_tensors,
                    )
                else:
                    monitored_loss = float(np.mean(epoch_losses))

                if monitored_loss < best_loss - 1e-6:
                    best_loss = monitored_loss
                    best_epoch = epoch
                    best_state = {
                        name: parameter.detach().cpu().clone()
                        for name, parameter in self.model_.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= int(self.patience):
                        break
        finally:
            if previous_threads is not None:
                try:
                    torch.set_num_threads(previous_threads)
                except (RuntimeError, ValueError):
                    pass

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.best_epoch_ = int(best_epoch)
        self.best_validation_loss_ = float(best_loss)
        self.device_ = "cpu"
        self.num_workers_ = 0
        self.ensemble_size_ = int(self.k)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary labels from averaged class-one probabilities."""
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return two-column class probabilities from averaged TabM outputs."""
        check_is_fitted(self, ("model_", "category_mappings_", "numeric_mean_"))
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for fitted C26_TABM predictions.") from exc

        X_frame = self._validate_input(X)
        x_num = torch.as_tensor(self._transform_numeric(X_frame), dtype=torch.float32)
        x_cat = torch.as_tensor(self._transform_categorical(X_frame), dtype=torch.int64)
        self.model_.eval()
        with torch.no_grad():
            logits = self._forward_logits(self.model_, x_num, x_cat)
            member_probabilities = torch.sigmoid(logits)
            if member_probabilities.ndim == 2:
                class_one = member_probabilities.mean(dim=1)
            elif member_probabilities.ndim == 1:
                class_one = member_probabilities
            else:
                raise FeaturePolicyPipelineError(
                    "TabM returned an invalid probability tensor shape."
                )
            class_one_array = class_one.detach().cpu().numpy().astype(float)
        if class_one_array.shape != (len(X_frame),):
            raise FeaturePolicyPipelineError(
                f"TabM returned invalid probability shape {class_one_array.shape!r}."
            )
        if not np.isfinite(class_one_array).all():
            raise FeaturePolicyPipelineError("TabM returned non-finite probabilities.")
        class_one_array = np.clip(class_one_array, 0.0, 1.0)
        return np.column_stack([1.0 - class_one_array, class_one_array])

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and order the native-categorical frame expected by TabM."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TabM native preprocessing expects a pandas DataFrame.")
        expected = [*self.numeric_features, *self.categorical_features]
        missing = [column for column in expected if column not in X.columns]
        if missing:
            raise FeaturePolicyPipelineError(
                f"TabM input is missing declared columns: {missing!r}."
            )
        return X.loc[:, expected].copy()

    def _fit_numeric_state(self, X: pd.DataFrame) -> None:
        """Learn fold-local numeric centering and scaling state."""
        if not self.numeric_features:
            self.numeric_mean_ = np.asarray([], dtype=np.float32)
            self.numeric_scale_ = np.asarray([], dtype=np.float32)
            return
        numeric = X.loc[:, list(self.numeric_features)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        values = numeric.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise FeaturePolicyPipelineError("TabM numeric inputs must be finite.")
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
        self.numeric_mean_ = mean
        self.numeric_scale_ = scale

    def _transform_numeric(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted numeric scaling without refitting on new rows."""
        if not self.numeric_features:
            return np.empty((len(X), 0), dtype=np.float32)
        numeric = X.loc[:, list(self.numeric_features)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        values = numeric.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise FeaturePolicyPipelineError("TabM numeric inputs must be finite.")
        return ((values - self.numeric_mean_) / self.numeric_scale_).astype(np.float32)

    def _fit_category_mappings(self, X: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Learn deterministic fold-local mappings, reserving zero for unknowns."""
        mappings: dict[str, dict[str, int]] = {}
        for column in self.categorical_features:
            values = X[column].astype("string").dropna().astype(str)
            categories = sorted(values.unique().tolist())
            mappings[column] = {
                category: index
                for index, category in enumerate(categories, start=1)
            }
        return mappings

    def _transform_categorical(self, X: pd.DataFrame) -> np.ndarray:
        """Encode new rows with fitted mappings without learning new categories."""
        if not self.categorical_features:
            return np.empty((len(X), 0), dtype=np.int64)
        encoded_columns = []
        for column in self.categorical_features:
            mapping = self.category_mappings_[column]
            encoded = (
                X[column]
                .astype("string")
                .astype(str)
                .map(mapping)
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
            )
            encoded_columns.append(encoded.reshape(-1, 1))
        return np.hstack(encoded_columns).astype(np.int64)

    def _make_fit_split_indices(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a fold-internal validation split for early stopping."""
        indices = np.arange(len(y))
        unique, counts = np.unique(y, return_counts=True)
        can_split = len(y) >= 40 and len(unique) == 2 and np.min(counts) >= 2
        if not can_split:
            return indices, np.asarray([], dtype=int)
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=0.15,
            stratify=y,
            random_state=int(self.random_state),
        )
        return train_indices, validation_indices

    def _make_model_kwargs(
        self,
        *,
        TabM,
        make_tabm_backbone,
        inspect_module,
    ) -> dict[str, object]:
        """Create and validate local-API TabM.make kwargs."""
        tabm_parameters = set(inspect_module.signature(TabM).parameters)
        constructor_args = {
            "n_num_features",
            "cat_cardinalities",
            "d_out",
        }
        missing_constructor = sorted(constructor_args.difference(tabm_parameters))
        if missing_constructor:
            raise FeaturePolicyPipelineError(
                "Local TabM constructor does not accept planned arguments: "
                f"{missing_constructor!r}."
            )

        backbone_parameters = set(inspect_module.signature(make_tabm_backbone).parameters)
        backbone_args = {
            "n_blocks",
            "d_block",
            "dropout",
            "activation",
            "k",
            "arch_type",
            "start_scaling_init",
        }
        missing_backbone = sorted(backbone_args.difference(backbone_parameters))
        if missing_backbone:
            raise FeaturePolicyPipelineError(
                "Local TabM backbone does not accept planned arguments: "
                f"{missing_backbone!r}."
            )
        if int(self.k) < 1:
            raise FeaturePolicyPipelineError("TabM k must be positive.")
        if int(self.n_blocks) < 1:
            raise FeaturePolicyPipelineError("TabM n_blocks must be positive.")
        if int(self.d_block) < 1:
            raise FeaturePolicyPipelineError("TabM d_block must be positive.")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise FeaturePolicyPipelineError("TabM dropout must be in [0, 1).")

        start_scaling_init = (
            None if str(self.arch_type) == "tabm-packed" else str(self.start_scaling_init)
        )
        return {
            "n_num_features": len(self.numeric_features),
            "cat_cardinalities": list(self.cat_cardinalities_),
            "d_out": 1,
            "n_blocks": int(self.n_blocks),
            "d_block": int(self.d_block),
            "dropout": float(self.dropout),
            "activation": str(self.activation),
            "k": int(self.k),
            "arch_type": str(self.arch_type),
            "start_scaling_init": start_scaling_init,
        }

    def _forward_logits(self, model, x_num, x_cat):
        """Run TabM and return handled logit shapes."""
        x_num_arg = None if x_num.shape[1] == 0 else x_num
        x_cat_arg = None if x_cat.shape[1] == 0 else x_cat
        output = model(x_num_arg, x_cat_arg)
        if output.ndim == 3 and output.shape[2] == 1:
            return output[:, :, 0]
        if output.ndim == 2:
            if output.shape[1] == 1:
                return output[:, 0]
            return output
        raise FeaturePolicyPipelineError(
            f"TabM returned invalid logit shape {tuple(output.shape)!r}."
        )

    def _per_row_loss(self, loss_fn, logits, y_true):
        """Reduce optional ensemble-member losses to one loss per row."""
        if logits.ndim == 2:
            losses = loss_fn(logits, y_true.unsqueeze(1).expand_as(logits))
            return losses.mean(dim=1)
        if logits.ndim == 1:
            return loss_fn(logits, y_true)
        raise FeaturePolicyPipelineError(
            f"TabM returned invalid logit shape {tuple(logits.shape)!r}."
        )

    def _validation_loss(self, torch, loss_fn, x_num, x_cat, y_true) -> float:
        """Evaluate unweighted validation loss for early stopping."""
        self.model_.eval()
        with torch.no_grad():
            logits = self._forward_logits(self.model_, x_num, x_cat)
            loss = self._per_row_loss(loss_fn, logits, y_true).mean()
        return float(loss.detach().cpu().item())


def make_feature_policy_classifier_pipeline(
    *,
    policy_id: FeaturePolicyId = FEATURE_POLICY_RAW,
    representation: FeatureRepresentation,
    classifier,
    feature_selection_policy: FeatureSelectionPolicyId = FEATURE_SELECTION_NONE,
    feature_selection_parameters: Mapping[str, object] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Combine feature policy, representation, optional selector, and classifier.

    The steps are deliberately ordered as:

    ``raw rows -> feature policy -> representation preprocessor -> selector -> model``.

    Both selectors operate on the represented matrix, after fold-local imputation,
    scaling, and one-hot encoding have been fitted from the active training partition.
    Native-categorical DataFrame representations are intentionally compatible only with
    ``S0_NONE`` in this phase; applying generic one-hot matrix selectors would break
    their categorical-column contract.
    """
    policy_id = validate_feature_policy_id(policy_id)
    feature_selection_policy = validate_feature_selection_policy_id(
        feature_selection_policy
    )

    if representation == REPRESENTATION_SPARSE_SCALED:
        preprocessor = make_feature_policy_one_hot_preprocessor(
            policy_id, scale_numeric=True, dense=False
        )
    elif representation == REPRESENTATION_SPARSE_UNSCALED:
        preprocessor = make_feature_policy_one_hot_preprocessor(
            policy_id, scale_numeric=False, dense=False
        )
    elif representation == REPRESENTATION_DENSE_SCALED:
        preprocessor = make_feature_policy_one_hot_preprocessor(
            policy_id, scale_numeric=True, dense=True
        )
    elif representation == REPRESENTATION_DENSE_UNSCALED:
        preprocessor = make_feature_policy_one_hot_preprocessor(
            policy_id, scale_numeric=False, dense=True
        )
    elif representation == REPRESENTATION_NATIVE_CATEGORICAL_DTYPE:
        if feature_selection_policy != FEATURE_SELECTION_NONE:
            raise FeaturePolicyPipelineError(
                "Native categorical representations are compatible only with S0_NONE."
            )
        preprocessor = make_feature_policy_native_categorical_preprocessor(
            policy_id, categorical_dtype=True
        )
    elif representation == REPRESENTATION_NATIVE_CATEGORICAL_STRING:
        if feature_selection_policy != FEATURE_SELECTION_NONE:
            raise FeaturePolicyPipelineError(
                "Native categorical representations are compatible only with S0_NONE."
            )
        preprocessor = make_feature_policy_native_categorical_preprocessor(
            policy_id, categorical_dtype=False
        )
    else:
        raise FeaturePolicyPipelineError(
            f"Unknown feature representation {representation!r}."
        )

    selector = make_feature_selector(
        feature_selection_policy,
        n_numeric_features=len(feature_policy_numeric_features(policy_id)),
        parameters=feature_selection_parameters,
        random_state=int(random_state),
    )

    return Pipeline(
        steps=[
            ("feature_policy", FeaturePolicyTransformer(policy_id=policy_id)),
            ("preprocessor", preprocessor),
            ("feature_selection", selector),
            ("classifier", classifier),
        ]
    )


def apply_imbalance_policy_to_pipeline(
    pipeline: Pipeline,
    *,
    imbalance_policy: ImbalancePolicyId = IMBALANCE_NONE,
    imbalance_parameters: Mapping[str, object] | None = None,
    random_state: int = 42,
):
    """Return a fitted-path-safe pipeline with one declared imbalance policy.

    ``I0_NONE`` preserves the original scikit-learn pipeline. ``I1`` wraps only the
    final classifier, so fold-local balanced weights are computed at fitting time. ``I2``
    and ``I3`` use an imbalanced-learn pipeline where row duplication or removal occurs
    after fold-local representation preprocessing and only during ``fit``. ``I4`` uses
    the mixed-data route before one-hot encoding and is intentionally restricted to
    ``F0_RAW``: synthesizing F1/F2 derived columns independently could violate their
    deterministic within-row relationships.
    """
    imbalance_policy = validate_imbalance_policy_id(imbalance_policy)
    parameters = normalize_imbalance_parameters(imbalance_policy, imbalance_parameters)

    required_steps = ("feature_policy", "preprocessor", "feature_selection", "classifier")
    missing_steps = [name for name in required_steps if name not in pipeline.named_steps]
    if missing_steps:
        raise FeaturePolicyPipelineError(
            "Imbalance routing requires the standard feature-policy pipeline steps; "
            f"missing {missing_steps!r}."
        )

    feature_policy_transformer = pipeline.named_steps["feature_policy"]
    policy_id = validate_feature_policy_id(feature_policy_transformer.policy_id)
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_selection = pipeline.named_steps["feature_selection"]
    classifier = pipeline.named_steps["classifier"]

    if imbalance_policy == IMBALANCE_NONE:
        return pipeline

    if imbalance_policy == IMBALANCE_CLASS_WEIGHT_BALANCED:
        return Pipeline(
            steps=[
                ("feature_policy", feature_policy_transformer),
                ("preprocessor", preprocessor),
                ("feature_selection", feature_selection),
                ("classifier", BalancedSampleWeightClassifier(estimator=classifier)),
            ]
        )

    if imbalance_policy in {
        IMBALANCE_RANDOM_OVERSAMPLING,
        IMBALANCE_RANDOM_UNDERSAMPLING,
    }:
        sampler = make_random_resampler(
            imbalance_policy,
            sampling_strategy=parameters["imbalance_sampling_strategy"],
            random_state=int(random_state),
        )
        return ImblearnPipeline(
            steps=[
                ("feature_policy", feature_policy_transformer),
                ("preprocessor", preprocessor),
                ("sampler", sampler),
                ("feature_selection", feature_selection),
                ("classifier", classifier),
            ]
        )

    if imbalance_policy == IMBALANCE_SMOTENC:
        if policy_id != FEATURE_POLICY_RAW:
            raise FeaturePolicyPipelineError(
                "I4_SMOTENC is initially compatible only with F0_RAW so that derived "
                "feature values are never synthesized independently of their inputs."
            )
        sampler = make_smotenc_resampler(
            n_numeric_features=len(feature_policy_numeric_features(policy_id)),
            n_categorical_features=len(feature_policy_categorical_features(policy_id)),
            sampling_strategy=parameters["imbalance_sampling_strategy"],
            k_neighbors=parameters["imbalance_smotenc_k_neighbors"],
            random_state=int(random_state),
        )
        return ImblearnPipeline(
            steps=[
                ("feature_policy", feature_policy_transformer),
                ("sampler_imputer", FeaturePolicySamplerImputer(policy_id=policy_id)),
                ("sampler", sampler),
                ("preprocessor", preprocessor),
                ("feature_selection", feature_selection),
                ("classifier", classifier),
            ]
        )

    raise RuntimeError(f"Unexpected validated imbalance policy {imbalance_policy!r}.")
