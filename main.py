from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
from parser_policy_codes import policy_parser
from virtualization import show_me, show_me2


def process_data(data: pd.DataFrame):
    data = data.fillna(0)
    if data.duplicated().any():
        data.drop_duplicates(inplace=False)
    target = data['cancellation_datetime']
    # extracting h_booking_id
    h_booking_id = data['h_booking_id']

    data.drop('h_booking_id', axis=1)

    data['booking_datetime'] = pd.to_datetime(data['booking_datetime']).dt.dayofyear + (
            pd.to_datetime(data['booking_datetime']).dt.year * 365)

    # new reservation_wait_time column
    data['reservation_wait_time'] = pd.to_datetime(data['checkin_date']).dt.dayofyear + (
            pd.to_datetime(data['checkin_date']).dt.year * 365) - data['booking_datetime']

    # converting to days

    data['checkin_date'] = pd.to_datetime(data['checkin_date']).dt.dayofyear
    data['travel_days_long'] = pd.to_datetime(data['checkout_date']).dt.dayofyear + (
            pd.to_datetime(data['checkout_date']).dt.year * 365) - pd.to_datetime(data['checkin_date']).dt.dayofyear + (
                                       pd.to_datetime(data['checkin_date']).dt.year * 365)

    data['cancellation_datetime'] = data['cancellation_datetime'].fillna(0)
    data['cancellation_datetime'] = pd.to_datetime(data['cancellation_datetime']).dt.dayofyear
    # to make new ?
    data['cancellation_datetime_days'] = pd.to_datetime(data['cancellation_datetime']).dt.dayofyear
    encoded_data_accommadation_type_name = pd.get_dummies(data['accommadation_type_name'], prefix='accommadation_type')
    data = pd.concat([data, encoded_data_accommadation_type_name], axis=1)
    # TODO: not good now
    # data['charge_option'] = data['charge_option'].replace({'Pay Later': 0, 'Pay Now': 1})
    dummies = pd.get_dummies(data['charge_option'], prefix='charge_option')

    # Concatenate the dummies with the original DataFrame
    data = pd.concat([data, dummies], axis=1)

    # additional features
    data = data.drop(["h_booking_id", "hotel_id", "checkout_date", "hotel_area_code", "hotel_brand_code",
                      "origin_country_code", "hotel_country_code",
                      "hotel_chain_code", "hotel_live_date", "h_customer_id", "customer_nationality",
                      "guest_is_not_the_customer", "guest_nationality_country_name", "language",
                      "original_payment_method", "original_payment_type",
                      "original_payment_currency", "is_user_logged_in",
                      "accommadation_type_name", "charge_option"
                      ],
                     axis=1)
    # TODO: remove
    data = data.drop(["cancellation_policy_code"], axis=1)

    return data, target, h_booking_id
    data = policy_parser(data)


path_file = '/Users/alon.frishberg/PycharmProjects/Hackathon/data_src/agoda_cancellation_train.csv'
# dtype={'column_name': str}
data = pd.read_csv(path_file, encoding='utf-8')
data, _, _ = process_data(data)

# show_me2/(data)
# show_me(data)

target = data["cancellation_datetime"]
features = data.drop("cancellation_datetime", axis=1)

classifiers = [
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    LogisticRegression(),
    SVC(),
    KNeighborsClassifier(),
    GaussianNB(),
    GradientBoostingClassifier()
]


def understand_features():
    subset_data = data.sample(n=10000, random_state=42)  # Adjust the number as per your requirement
    # Separate the features and target variable from the subset
    mini_features = subset_data.drop("cancellation_datetime", axis=1)
    mini_target = subset_data["cancellation_datetime"]
    # Train a Random Forest classifier
    clf = RandomForestClassifier()
    clf.fit(mini_features, mini_target)
    # Get feature importances
    feature_importances = clf.feature_importances_
    # Print feature importance scores
    for feature, importance in zip(features.columns, feature_importances):
        print(f"{feature}: {round(importance, 4)}")


def split_and_classify():
    split_sizes = [0.2, 0.3, 0.4]
    for classifier in classifiers:
        print(f"Estimator: {classifier.__class__.__name__}")
        for split_size in split_sizes:
            print(f"Split size: {split_size}")

            # Perform train-test split
            X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=split_size,
                                                                test_size=(1 - split_size), random_state=42)
            # Train a Random Forest classifier
            classifier.fit(X_train, y_train)
            # Make predictions on the test set
            y_pred = classifier.predict(X_test)

            # Evaluate the classifier
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy}\n")


def union_grid_search():
    param_grids = [
        {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]},
        {'max_depth': [None, 5, 10]},
        {'C': [0.1, 1, 10]},
        {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]},
        {'n_neighbors': [3, 5, 7]},
        {},
        {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]},
        {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
    ]
    # Create a results grid DataFrame
    results_grid = pd.DataFrame(columns=['Classifier', 'Parameters', 'Accuracy'])
    # Iterate over classifiers and parameter grids
    for classifier, param_grid in zip(classifiers, param_grids):
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(classifier, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Get the best parameters and accuracy score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Evaluate the classifier on the test set
        y_pred = grid_search.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Append results to the grid DataFrame
        results_grid = results_grid.append({'Classifier': classifier.__class__.__name__,
                                            'Parameters': best_params,
                                            'Accuracy': accuracy}, ignore_index=True)
    # Display the results grid
    print(results_grid)


understand_features()
# version 1
union_grid_search()
# # version 2
split_and_classify()
