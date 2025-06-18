from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

from feature_engineering import FeatureEngineering
from fillna import FillNan


def load_and_merge_data(data_path: str, fe_data_path: str) -> pd.DataFrame:
    """
    Load and merge data
    """
    # load data :
    data_set = pd.read_csv(data_path)
    fe_set = pd.read_csv(fe_data_path)

    # drop duplicates :
    data_set.drop_duplicates(inplace=True)
    fe_set.drop_duplicates(inplace=True)

    return pd.merge(data_set, fe_set, on="description", how="left")


def apply_feature_engineering(
    data: pd.DataFrame, categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Apply feature engineering process
    """
    fe = FeatureEngineering(data, categorical_cols)
    return fe.transform()


def fill_nan_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill nan values for columns label_* and count_*
    """
    # columns which begin by label_ :
    label_columns = list(data.loc[:, data.columns.str.startswith("label_")].columns)

    # columns which begin by count_ :
    count_columns = list(data.loc[:, data.columns.str.startswith("count_")].columns)

    fillna = FillNan(label_columns, count_columns, data)
    return fillna.fillna()


def train_and_predict(
    model, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
) -> List[float]:
    """
    input:
        - model : model after tuning hyperparameters
        - X_train : training set dataframe
        - y_train : training ground truth
        - X_test : test set dataframe
    output :

    """
    X_tr = X_train.copy()
    X_ts = X_test.copy()

    # Remove duplicate columns :
    X_tr = X_tr.loc[:, ~X_tr.columns.duplicated()]
    X_ts = X_ts.loc[:, ~X_ts.columns.duplicated()]

    def intersection(lst1: List[str], lst2: list):
        return list(set(lst1) & set(lst2))

    # Get columns intersection :
    cols = intersection(list(X_tr.columns), list(X_ts.columns))

    X_tr = X_tr[cols]
    X_ts = X_ts[cols]

    assert list(X_tr.columns) == list(X_ts.columns)

    model.fit(X_tr, y_train)
    return model.predict(X_ts)


def create_submission_csv(y_pred: List[float], target_file: str) -> None:
    """
    Create submission.csv file and save it in data file
    """
    ids = [f"{i}_test" for i in range(len(y_pred))]
    submission = pd.DataFrame({"drug_id": ids, "price": y_pred})

    submission.to_csv(target_file)


def preprocess_data(
    params: Dict[str, str]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load and process data.
    It takes as parameter a dictionary containing
    all data paths and selected columns to deal with the task.
    """
    # Load csv_paths :
    train_set_path = params["training_set"]
    test_set_path = params["test_set"]
    fe_set_path = params["fe_set"]

    # Load data :
    train_set = load_and_merge_data(train_set_path, fe_set_path)
    test_set = load_and_merge_data(test_set_path, fe_set_path)

    # Drop ids columns :
    train_set.drop(columns=["drug_id"], inplace=True)
    test_set.drop(columns=["drug_id"], inplace=True)

    # Fill nan values
    train_set = fill_nan_values(train_set)
    test_set = fill_nan_values(test_set)

    y_train = train_set["price"]
    X_train = train_set.drop(columns=["price"])

    # Apply feature engineering:
    X_train = apply_feature_engineering(
        X_train, params["categorical_variables_apply_get_dummies"]
    )
    test_set = apply_feature_engineering(
        test_set, params["categorical_variables_apply_get_dummies"]
    )

    X_train.drop(columns=params["redundancy_columns"], inplace=True)
    test_set.drop(columns=params["redundancy_columns"], inplace=True)

    return X_train, y_train, test_set


def tuning_parameters(model, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Apply GridSearchCV in order to find suitable parameters based on r2-score
    """
    rg = GridSearchCV(
        estimator=model,
        param_grid={
            "num_leaves": [15, 31, 45],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [50, 100],
        },
        scoring="r2",
        cv=3,
        n_jobs=3,
        verbose=1,
        refit=True,
    )
    rg.fit(X_train, y_train)
    return rg.best_params_


def evaluate_regressor(regressor, X_train: pd.DataFrame, y_train: pd.Series) -> None:

    """
    performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    Inputs :
        - regressor : the regressor to use
        - X_train : the dataset to work on
        - y_train : the y_train used for training and validation
    :return:
    """
    n_splits = 5
    kf = KFold(n_splits=n_splits, random_state=50, shuffle=True)
    r2_scores = []
    mse_list = []

    for training_ids, test_ids in kf.split(X_train):

        if type(X_train) == pd.DataFrame:

            training_set = X_train.loc[training_ids]
            training_y_train = y_train.loc[training_ids]

            test_set = X_train.loc[test_ids]
            test_y_train = y_train.loc[test_ids]

        elif type(X_train) == np.ndarray:

            training_set = X_train[training_ids]
            training_y_train = y_train[training_ids]

            test_set = X_train[test_ids]
            test_y_train = y_train[test_ids]

        regressor.fit(training_set, training_y_train)

        predicted_y_train = regressor.predict(test_set)

        r2_scores.append(r2_score(test_y_train, predicted_y_train))
        mse_list.append(mean_squared_error(test_y_train, predicted_y_train))

    r2_sc = np.mean(r2_scores)
    mse_list = np.mean(mse_list, axis=0)

    print("#" * 50)
    print(f"R2 Score = {r2_sc}")
    print(f"MSE : \n {mse_list}")
    print("#" * 50)


def main():
    config_path = "config.yaml"

    yaml = YAML(typ="safe")
    with open(config_path) as f:
        params = yaml.load(f)

    # Load and preprocess data :
    X_train, y_train, test_set = preprocess_data(params)

    # Tuning model:
    best_model = XGBRegressor(**tuning_parameters(XGBRegressor(), X_train, y_train))

    # Evaluate regressor:
    evaluate_regressor(best_model, X_train, y_train)

    # Train and predict :
    y_pred = train_and_predict(best_model, X_train, test_set, y_train)

    # Create submission file :
    create_submission_csv(y_pred, params["submission_file_path"])


if __name__ == "__main__":
    main()
