from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go

from multi_classifier import MultiClassifier
from parser_policy_codes import policy_parser
from virtualization import show_me, data_plotting


def process_data(data: pd.DataFrame):
    # remove dup
    if data.duplicated().any():
        data.drop_duplicates(inplace=False)
    # target label
    h_booking_id = data['h_booking_id']
    data.drop('h_booking_id', axis=1)

    check_in = pd.to_datetime(data['checkin_date']).dt.dayofyear + (
            pd.to_datetime(data['checkin_date']).dt.year * 365)
    booking_time = pd.to_datetime(data['booking_datetime']).dt.dayofyear + (
            pd.to_datetime(data['booking_datetime']).dt.year * 365)
    # new reservation_wait_time column
    data['time_until_checking_in_days'] = check_in - booking_time
    data['booking_datetime'] = pd.to_datetime(data['booking_datetime']).dt.month
    data['checkin_date_month'] = pd.to_datetime(data['checkin_date']).dt.month
    data['travel_days_long'] = (pd.to_datetime(data['checkout_date']).dt.dayofyear + (
            pd.to_datetime(data['checkout_date']).dt.year * 365)) - (
                                       pd.to_datetime(data['checkin_date']).dt.dayofyear + (
                                       pd.to_datetime(data['checkin_date']).dt.year * 365))

    data['is_cancelled'] = (pd.to_datetime(data['cancellation_datetime']).dt.dayofyear).apply(
        lambda x: 1 if x > 0 else 0)
    encoded_data_accommadation_type_name = pd.get_dummies(data['accommadation_type_name'], prefix='accommadation_type')
    data = pd.concat([data, encoded_data_accommadation_type_name], axis=1)
    dummies = pd.get_dummies(data['charge_option'], prefix='charge_option')
    data = pd.concat([data, dummies], axis=1)

    # clean data
    data = data.fillna(0)
    target = data['is_cancelled']
    # additional features
    data = data.drop(["cancellation_datetime", "is_cancelled"], axis=1)
    data = data.drop(["h_booking_id", "hotel_id", "checkout_date", "hotel_area_code", "hotel_brand_code",
                      "origin_country_code", "hotel_country_code", "hotel_city_code",
                      "hotel_chain_code", "hotel_live_date", "h_customer_id", "customer_nationality",
                      "guest_is_not_the_customer", "guest_nationality_country_name", "language",
                      "original_payment_method", "original_payment_type", "request_nonesmoke",
                      "original_payment_currency",
                      "is_first_booking", "no_of_room", "no_of_extra_bed", "no_of_children",
                      "no_of_adults", "is_user_logged_in", "request_latecheckin",
                      "request_highfloor", "request_highfloor", "request_twinbeds", "request_airport",
                      "request_earlycheckin",
                      "accommadation_type_name", "charge_option", "checkin_date"
                      ], axis=1)
    # TODO: remove
    data = data.drop(["cancellation_policy_code"], axis=1)
    return data, target, h_booking_id
    # data = policy_parser(data)


def corr_func():
    correlations = np.corrcoef(data, target, rowvar=False)[-1, :-1]
    feature_names = data.columns.tolist()[:-1]
    # Print the correlation values
    for feature_name, correlation in zip(feature_names, correlations):
        print(f"Correlation between {feature_name} and y: {correlation}")


path_file = '/Users/alon.frishberg/PycharmProjects/IML_Hackathon/data_src/agoda_cancellation_train.csv'
data = pd.read_csv(path_file, encoding='utf-8')
data, target, h_booking_id = process_data(data)
data.to_csv('output.csv', index=False)
# data_plotting(data)
multi_classifers = MultiClassifier()


# classifiers = [
#     RandomForestClassifier(),
#     DecisionTreeClassifier(),
#     LogisticRegression(),
#     SVC(),
#     KNeighborsClassifier(),
#     GaussianNB(),
#     GradientBoostingClassifier()
# ]


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
    for feature, importance in zip(data.columns, feature_importances):
        print(f"{feature}: {round(importance, 4)}")


def split_and_classify(split_sizes=[0.2, 0.3, 0.4]):
    for classifier in multi_classifers.classifiers:
        print(f"Estimator: {classifier.__class__.__name__}")
        for split_size in split_sizes:
            print(f"Split size: {split_size}")
            # Perform train-test split
            X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=split_size,
                                                                test_size=(1 - split_size), random_state=42)
            # Train a Random Forest classifier
            classifier.fit(X_train, y_train)
            # Make predictions on the test set
            y_pred = classifier.predict(X_test)
            y_pred = np.round(y_pred).astype(int)
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
    for classifier, param_grid in zip(multi_classifers.classifiers, param_grids):
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

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


# def ensemble(data_test:pd.DataFrame):
#     ensemble_pred = []
#     for i in range(len(data_test)):
#         votes = [pred1[i], pred2[i], pred3[i]]
#         # Use majority voting for classification
#         ensemble_pred.append(max(set(votes), key=votes.count))
#     # Step 6: Evaluate the ensemble model's performance
#     ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
#     print("Ensemble Accuracy:", ensemble_accuracy)


# ensemble(data_test)
# understand_features()
# version 1
# union_grid_search()
# # version 2
split_and_classify()
