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

DATASET_PATH = "rep_dataset_03.csv"
MODEL_PATH = "rep_classifier_bundle.pkl"

LABEL_COLUMN = "label"
GROUP_COLUMN = "set_id"

BASE_FEATURE_COLUMNS = [
    "duration",
    "max_ay",
    "min_ay",
    "range_ay",
    "mean_ay",
    "std_ay",
    "gyro_var",
]

BASELINE_SOURCE_COLUMNS = [
    "duration",
    "std_ay",
    "gyro_var",
    "range_ay",
    "max_ay",
]

LABEL_NAMES = {
    0: "normal",
    1: "near_failure",
}

THRESHOLDS_TO_TEST = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def add_baseline_features(df, baseline_reps=3):
    """
    Adds baseline-relative features using the first few reps in each set.

    This is okay for real-time later because the first few reps happen before
    the later reps we are trying to classify.
    """
    df = df.copy()

    for feature in BASELINE_SOURCE_COLUMNS:
        baseline_col = f"baseline_{feature}"
        ratio_col = f"{feature}_ratio_to_baseline"
        diff_col = f"{feature}_diff_from_baseline"

        df[baseline_col] = df.groupby(GROUP_COLUMN)[feature].transform(
            lambda x: x.iloc[:baseline_reps].mean()
        )

        df[ratio_col] = df[feature] / df[baseline_col]
        df[diff_col] = df[feature] - df[baseline_col]

    return df


def get_feature_columns():
    baseline_feature_columns = []

    for feature in BASELINE_SOURCE_COLUMNS:
        baseline_feature_columns.append(f"{feature}_ratio_to_baseline")
        baseline_feature_columns.append(f"{feature}_diff_from_baseline")

    return BASE_FEATURE_COLUMNS + baseline_feature_columns


def print_threshold_results(y_test, y_proba, thresholds):
    """
    Tests different probability thresholds for predicting near failure.

    Lower threshold = more near-failure warnings, usually higher recall.
    Higher threshold = fewer near-failure warnings, usually higher precision.
    """
    best_threshold = None
    best_recall = -1
    best_f1 = -1

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

        # Choose threshold by highest near-failure recall,
        # and break ties using near-failure F1.
        if (
            near_failure_recall > best_recall or
            (near_failure_recall == best_recall and near_failure_f1 > best_f1)
        ):
            best_threshold = threshold
            best_recall = near_failure_recall
            best_f1 = near_failure_f1

    print("\n" + "=" * 60)
    print("Suggested threshold:")
    print(best_threshold)
    print(f"Near-failure recall at suggested threshold: {best_recall:.3f}")
    print(f"Near-failure F1 at suggested threshold:     {best_f1:.3f}")

    return best_threshold


def main():
    df = pd.read_csv(DATASET_PATH)

    # Add baseline-relative features
    df = add_baseline_features(df, baseline_reps=3)
    #drop rows that don't have some of the label cols
    df = df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN])

    feature_columns = get_feature_columns()

    needed_columns = feature_columns + [LABEL_COLUMN, GROUP_COLUMN]
    df = df.dropna(subset=needed_columns)

    X = df[feature_columns]
    y = df[LABEL_COLUMN]
    groups = df[GROUP_COLUMN]

    print("Dataset shape:")
    print(df.shape)

    print("\nFeature columns:")
    for col in feature_columns:
        print("-", col)

    print("\nClass counts:")
    print(y.value_counts())

    print("\nSet counts:")
    print(groups.value_counts().sort_index())

    # Split by set_id so reps from the same set stay together
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

    # GroupKFold keeps whole sets together during cross-validation
    cv = GroupKFold(n_splits=5)

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

    # Default prediction with model's built-in threshold of 0.5
    y_pred = best_model.predict(X_test)

    print("\n" + "=" * 60)
    print("Default Test Results")
    print("=" * 60)

    print("\nTest Accuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nTest F1 Score for Near Failure:")
    print(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["normal", "near_failure"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    # Probability of near failure
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("Threshold Testing")
    print("=" * 60)

    best_threshold = print_threshold_results(
        y_test=y_test,
        y_proba=y_proba,
        thresholds=THRESHOLDS_TO_TEST
    )

    # Feature importances
    rf = best_model.named_steps["model"]

    print("\n" + "=" * 60)
    print("Feature Importances")
    print("=" * 60)

    importances = sorted(
        zip(feature_columns, rf.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )

    for feature, importance in importances:
        print(f"{feature}: {importance:.4f}")

    # Save model and metadata for real-time inference
    model_bundle = {
        "model": best_model,
        "feature_columns": feature_columns,
        "base_feature_columns": BASE_FEATURE_COLUMNS,
        "baseline_source_columns": BASELINE_SOURCE_COLUMNS,
        "label_names": LABEL_NAMES,
        "near_failure_threshold": best_threshold,
        "baseline_reps": 3,
    }

    joblib.dump(model_bundle, MODEL_PATH)

    print("\n" + "=" * 60)
    print(f"Saved model bundle to {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()