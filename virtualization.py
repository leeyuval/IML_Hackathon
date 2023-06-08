import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def show_me2(data:pd.DataFrame):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Get the list of categorical features
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    # Get the list of numerical features
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for feature in categorical_features:
        # Bar plot of the target variable against the categorical feature
        plt.figure(figsize=(10, 6))
        sns.barplot(data=data, x=feature, y='cancellation_datetime')
        plt.title(f'{feature} vs Cancellation DateTime')
        plt.xlabel(feature)
        plt.ylabel('Cancellation DateTime')
        plt.show()

    for feature in numerical_features:
        # Scatter plot of the target variable against the numerical feature
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=feature, y='cancellation_datetime')
        plt.title(f'Cancellation DateTime vs {feature}')
        plt.xlabel(feature)
        plt.ylabel('Cancellation DateTime')
        plt.show()


def show_me(data: pd.DataFrame):
    # Examine the distribution of the target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(data['cancellation_datetime'])
    plt.title('Distribution of cancellation_datetime')
    plt.xlabel('cancellation_datetime')
    plt.ylabel('Count')
    plt.show()
    # Explore relationships between the target variable and other features
    # Example: Scatter plot between cancellation_datetime and original_selling_amount
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='cancellation_datetime', y='original_selling_amount')
    plt.title('Relationship between cancellation_datetime and original_selling_amount')
    plt.xlabel('cancellation_datetime')
    plt.ylabel('original_selling_amount')
    plt.show()
    # Example: Box plot of cancellation_datetime by hotel_star_rating
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='hotel_star_rating', y='cancellation_datetime')
    plt.title('Cancellation datetime by hotel_star_rating')
    plt.xlabel('hotel_star_rating')
    plt.ylabel('cancellation_datetime')
    plt.show()
