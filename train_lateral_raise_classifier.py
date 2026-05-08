import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

DATASET_PATH = "rep_dataset_lateral_raises_features.csv"
MODEL_PATH = "lateral_raise_classifier_bundle.pkl"

LABEL_COLUMN = "label"
GROUP_COLUMN = "set_id"

# Start with the same threshold as bicep curls.
# You can change this after seeing threshold testing results.
NEAR_FAILURE_THRESHOLD = 0.30

THRESHOLDS_TO_TEST = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

COLUMNS_TO_EXCLUDE = [
    LABEL_COLUMN,
    GROUP_COLUMN,
    "rep_id",
    "start_time",
    "end_time",
    "segment_num_samples",
    "segment_sampling_rate",
]


def should_exclude_feature(col):
    if col in COLUMNS_TO_EXCLUDE:
        return True

    # Exclude raw baseline columns because the ratio/diff columns
    # usually contain the useful baseline-relative information.
    if col.startswith("baseline_"):
        return True

    return False


def get_feature_columns(df):
    feature_columns = []

    for col in df.columns:
        if should_exclude_feature(col):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)

    return feature_columns


def clean_labels(df):
    """
    Makes sure labels are numeric:
    normal -> 0
    near_failure -> 1

    If the labels are already 0/1, this leaves them alone.
    """

    df = df.copy()

    if df[LABEL_COLUMN].dtype == "object":
        label_map = {
            "normal": 0,
            "near_failure": 1,
            "near failure": 1,
            "near-failure": 1,
        }

        df[LABEL_COLUMN] = (
            df[LABEL_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(label_map)
        )

    return df


def print_threshold_results(y_test, y_proba, thresholds):
    for threshold in thresholds:
        y_pred_thresholded = (y_proba >= threshold).astype(int)

        near_failure_precision = precision_score(
            y_test,
            y_pred_thresholded,
            pos_label=1,
            zero_division=0
        )

        near_failure_recall = recall_score(
            y_test,
            y_pred_thresholded,
            pos_label=1,
            zero_division=0
        )

        near_failure_f1 = f1_score(
            y_test,
            y_pred_thresholded,
            pos_label=1,
            zero_division=0
        )

        print("\n" + "=" * 60)
        print(f"Threshold: {threshold}")
        print(f"Near-failure precision: {near_failure_precision:.3f}")
        print(f"Near-failure recall:    {near_failure_recall:.3f}")
        print(f"Near-failure F1:        {near_failure_f1:.3f}")

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred_thresholded,
            target_names=["normal", "near_failure"],
            zero_division=0
        ))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_thresholded, labels=[0, 1]))


def print_chosen_threshold_results(y_test, y_proba, threshold):
    y_pred_thresholded = (y_proba >= threshold).astype(int)

    near_failure_precision = precision_score(
        y_test,
        y_pred_thresholded,
        pos_label=1,
        zero_division=0
    )

    near_failure_recall = recall_score(
        y_test,
        y_pred_thresholded,
        pos_label=1,
        zero_division=0
    )

    near_failure_f1 = f1_score(
        y_test,
        y_pred_thresholded,
        pos_label=1,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"Chosen Threshold Results: {threshold}")
    print("=" * 60)
    print(f"Near-failure precision: {near_failure_precision:.3f}")
    print(f"Near-failure recall:    {near_failure_recall:.3f}")
    print(f"Near-failure F1:        {near_failure_f1:.3f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred_thresholded,
        target_names=["normal", "near_failure"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_thresholded, labels=[0, 1]))


def main():
    df = pd.read_csv(DATASET_PATH)

    df = clean_labels(df)

    df = df.dropna(subset=[LABEL_COLUMN, GROUP_COLUMN])

    # Make sure labels are integers after cleaning
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    feature_columns = get_feature_columns(df)

    X = df[feature_columns]
    y = df[LABEL_COLUMN]
    groups = df[GROUP_COLUMN]

    print("Dataset shape:")
    print(df.shape)

    print("\nNumber of features after filtering:")
    print(len(feature_columns))

    print("\nFeature columns:")
    for col in feature_columns:
        print("-", col)

    print("\nClass counts:")
    print(y.value_counts())

    print("\nSet counts:")
    print(groups.value_counts().sort_index())

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    print("\nTrain sets:")
    print(sorted(train_groups.unique()))

    print("\nTest sets:")
    print(sorted(test_groups.unique()))

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__class_weight": [None, "balanced"],
    }

    num_train_groups = train_groups.nunique()
    num_splits = min(5, num_train_groups)

    if num_splits < 2:
        raise ValueError(
            "Not enough training groups for GroupKFold. "
            "Need at least 2 unique set_id values in training data."
        )

    cv = GroupKFold(n_splits=num_splits)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train, groups=train_groups)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest CV F1 Score:")
    print(grid_search.best_score_)

    best_model = grid_search.best_estimator_

    y_pred_default = best_model.predict(X_test)

    print("\n" + "=" * 60)
    print("Default Test Results")
    print("=" * 60)

    print("\nTest Accuracy:")
    print(accuracy_score(y_test, y_pred_default))

    print("\nTest F1 Score for Near Failure:")
    print(f1_score(y_test, y_pred_default, pos_label=1, zero_division=0))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred_default,
        target_names=["normal", "near_failure"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_default, labels=[0, 1]))

    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("Threshold Testing")
    print("=" * 60)

    print_threshold_results(
        y_test=y_test,
        y_proba=y_proba,
        thresholds=THRESHOLDS_TO_TEST
    )

    print_chosen_threshold_results(
        y_test=y_test,
        y_proba=y_proba,
        threshold=NEAR_FAILURE_THRESHOLD
    )

    rf = best_model.named_steps["model"]

    print("\n" + "=" * 60)
    print("Top Feature Importances")
    print("=" * 60)

    importances = sorted(
        zip(feature_columns, rf.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )

    for feature, importance in importances[:30]:
        print(f"{feature}: {importance:.4f}")

    model_bundle = {
        "model": best_model,
        "feature_columns": feature_columns,
        "near_failure_threshold": NEAR_FAILURE_THRESHOLD,
        "columns_excluded": COLUMNS_TO_EXCLUDE,
        "exercise": "lateral_raise",
        "dataset_path": DATASET_PATH,
    }

    joblib.dump(model_bundle, MODEL_PATH)

    print("\n" + "=" * 60)
    print(f"Saved model bundle to {MODEL_PATH}")
    print(f"Saved near-failure threshold: {NEAR_FAILURE_THRESHOLD}")
    print("=" * 60)


if __name__ == "__main__":
    main()