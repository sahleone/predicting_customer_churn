
""" Rhys Jervis 08/16/21
Predict Customer Churn procedure
"""

# import libraries
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sns.set()


def import_data(pth):
    '''returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    dataframe = pd.read_csv(pth)
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    # Plot hist of Churn
    dataframe['Churn'].plot(title="Histogram of Churn", kind="hist",
                            figsize=(20, 10))
    plt.savefig("images/eda/Histogram_of_Churn.jpg")
    plt.clf()

    # Plot hist of Customer_Age
    dataframe['Customer_Age'].plot(
        title="Histogram of Customer_Age",
        kind="hist",
        figsize=(
            20,
            10))
    plt.savefig("images/eda/Histogram_of_Customer_Age.jpg")
    plt.clf()

    # Plot hist of Marital_Status
    dataframe['Marital_Status'].value_counts('normalize').plot(
        title="Marital_Status", kind="bar", figsize=(20, 10))
    plt.savefig("images/eda/Marital_Status(normalize).jpg")

    # Plot hist of Total_Trans_Ct
    _ = plt.figure(figsize=(20, 10))
    _ = sns.distplot(dataframe['Total_Trans_Ct'])
    plt.savefig("images/eda/Distplot_Total_Trans_Ct.jpg")

    # Plot hist of Total_Trans_Ct
    _ = plt.figure(figsize=(20, 10))
    _ = sns.heatmap(
        dataframe.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig("images/eda/Heatmap_Corr.jpg")
    plt.clf()


def encoder_helper(df, category_lst, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from
    the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
            be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    i = 0
    for cat_name in category_lst:
        cat_name_lst = []
        cat_groups = df.groupby(cat_name).mean()['Churn']

        for val in df[cat_name]:
            cat_name_lst.append(cat_groups.loc[val])

        if response:
            df[response[i]] = cat_name_lst
            i += 1
        else:
            df[cat_name + '_Churn'] = cat_name_lst

    return df


def perform_feature_engineering(df, response=None):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = pd.DataFrame()
    y = df['Churn']
    if response:

        X[response] = df[response]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
    else:
        keep_cols = [
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio',
            'Gender_Churn',
            'Education_Level_Churn',
            'Marital_Status_Churn',
            'Income_Category_Churn',
            'Card_Category_Churn']

        X[keep_cols] = df[keep_cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    _ = plt.rc('figure', figsize=(5, 5))

    _ = plt.text(0.01, 0.9, str('Random Forest Train'),
                 {'fontsize': 10}, fontproperties='monospace')
    _ = plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    _ = plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10},
                 fontproperties='monospace')
    _ = plt.text(
        0.01, 0.65, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    _ = plt.axis('off')
    plt.savefig("images/results/Classification_Report.png")
    plt.clf()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    _ = plt.figure(figsize=(20, 5))

    # Create plot title
    _ = plt.title("Feature Importance")
    _ = plt.ylabel('Importance')

    # Add bars
    _ = plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    _ = plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.clf()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    lrc = LogisticRegression(max_iter=100000)
    rfc = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(
        cv_rfc,
        x_train,
        './images/results/feature_importance_plot.png')
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=ax, alpha=0.8)
    plot_roc_curve(lrc, x_test, y_test, ax=ax, alpha=0.8)

    plt.savefig("./images/results/roc.png")

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    import os
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    filepath = r"./data/bank_data.csv"
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    dataframe = import_data(filepath)
    perform_eda(dataframe)
    dataframe_encoded = encoder_helper(dataframe, category_lst=cat_columns)
    x_train, x_test, y_train, y_test = perform_feature_engineering(dataframe_encoded)
    train_models(x_train, x_test, y_train, y_test)
