import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATASET_PATH = "rep_dataset.csv"
MODEL_PATH = "rep_classifier.pkl"

#adjust these to column names used in clean data csv
FEATURE_COLUMNS = [
    "rep_duration",
    "peak_gyro_mag",
    "mean_gyro_mag",
    "gyro_range",
    "accel_mag_std",
    "duration_ratio_to_baseline",
    "peak_ratio_to_baseline",
]

LABEL_COLUMN = "label"

def main():
    df = pd.read_csv(DATASET_PATH)

    #drop rows that don't have some of the label cols
    df = df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN])

    #split into features and labels
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    print("Class counts:")
    print(y.value_counts())
    print()

    #train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    rf_model = RandomForestClassifier(random_state=42)

    #hyperparameter grid for grid search cv
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced"]
    }

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:")
    print(grid_search.best_params_)
    print()
    print("Best CV Score:")
    print(grid_search.best_score_)
    print()

    #choose the best model
    best_model = grid_search.best_estimator_

    #test predictions
    y_pred = best_model.predict(X_test)

    print("Test Accuracy:")
    print(accuracy_score(y_test, y_pred))
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    #save best model
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")