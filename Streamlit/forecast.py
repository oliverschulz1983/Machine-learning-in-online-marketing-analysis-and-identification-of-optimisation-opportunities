import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Function to get the filtered data set
def get_data(campaign):
    
    # usage of pathlib
    PROJECT_DIR = Path(__file__).parent
    path = PROJECT_DIR / 'mldata.csv'
    mldata = pd.read_csv(path)

    # Preprocessing LinReg with "post_click_sales_amount" as Target

    # All Campaigns
    mldata_camp_total = mldata

    # Only Camp 1
    mldata_camp_1 = mldata[mldata["campaign_number"] == "camp 1"]

    # Only Camp 2
    mldata_camp_2 = mldata[mldata["campaign_number"] == "camp 2"]

    # Only Camp 3
    mldata_camp_3 = mldata[mldata["campaign_number"] == "camp 3"]


    # Selection of selected features relevant for ML - without "campaign_number" as feature - with "clicks" as label:

    # All Campaigns
    mldata_camp_total = mldata_camp_total[["banner", "placement", "ad_spend_by_company_x", "post_click_sales_amount"]].reset_index(drop=True)

    # Only Camp 1
    mldata_camp_1 = mldata_camp_1[["banner", "placement", "ad_spend_by_company_x", "post_click_sales_amount"]].reset_index(drop=True)

    # Only Camp 2
    mldata_camp_2 = mldata_camp_2[["banner", "placement", "ad_spend_by_company_x", "post_click_sales_amount"]].reset_index(drop=True)

    # Only Camp 3
    mldata_camp_3 = mldata_camp_3[["banner", "placement", "ad_spend_by_company_x", "post_click_sales_amount"]].reset_index(drop=True)

    if campaign == "Campaign 1":
        return mldata_camp_1
    elif campaign == "Campaign 2":
        return mldata_camp_2
    elif campaign == "Campaign 3":
        return mldata_camp_3
    elif campaign == "all Campaigns":
        return mldata_camp_total


# Function to get the model and the scaler
def get_model(campaign):
    data = get_data(campaign)

    # Define the features(X) and the label(y) 
    X = data[["banner", "placement", "ad_spend_by_company_x"]]
    y = data[["post_click_sales_amount"]]

    # Creating dummies for the categorical variables
    X = pd.get_dummies(data=X, columns=["banner"], prefix="banner", dtype=float, drop_first=True)
    X = pd.get_dummies(data=X, columns=["placement"], prefix="placement", dtype=float, drop_first=True)

    # Standardisation: Very many variables in X, with different scales/scaling => therefore standardisation of the data
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Creating the training and test data sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.33, random_state=100)

    # Initialising the model for ridge regression 
    ridge_reg = Ridge(alpha=0.1, solver="cholesky")

    # Training the model
    ridge_reg.fit(X_train, y_train)

    return ridge_reg, scaler, X, y, X_train, X_test, y_train, y_test, scaled_X


# Function to create the forecast
def get_prediction(campaign, adspend, banner_values, placement_values):

    ridge_reg, scaler, X, y, X_train, X_test, y_train, y_test, scaled_X = get_model(campaign)

    prediction = pd.DataFrame([[adspend, banner_values["bv2"]/100, banner_values["bv3"]/100, banner_values["bv4"]/100, banner_values["bv5"]/100, banner_values["bv6"]/100, banner_values["bv7"]/100, banner_values["bv8"]/100, placement_values["pv2"]/100, placement_values["pv3"]/100, placement_values["pv4"]/100, placement_values["pv5"]/100]], columns=X.columns)
    scaled_prediction_data = scaler.transform(prediction)

    # Make and format prediction
    predicted_value = ridge_reg.predict(scaled_prediction_data)

    # inner function, for the purpose of formatting the predicted value
    def format_prediction(predicted_value):
        # Extract the actual predicted value from the array and format it
        return f'{predicted_value[0][0]:,.2f}'

    # Return formatted prediction
    return format_prediction(predicted_value) # gives me the exact prediction


# Function to get the regression results
def get_regression_results(campaign):

    ridge_reg, scaler, X, y, X_train, X_test, y_train, y_test, scaled_X = get_model(campaign)

    # Make a prediction
    y_pred = ridge_reg.predict(X_test)

    # Model performance for test data set:
    r2 = r2_score(y_test, y_pred) # Calculation of the R² coefficient of determination

    # Calculation of the RMSE error measure
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    # OLS regression results: Fit model
    model = sm.OLS(y_train, sm.add_constant(pd.DataFrame(X_train, columns=scaled_X.columns)))
    results = model.fit().summary()

    return rmse, r2, results

# Function to create the scatter plot for the prediction accuracy
def create_seaborn_scatter_plot(y_test, y_pred):
    # Ensure that the arrays are one-dimensional
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)

    # Create a new figure
    plt.figure(figsize=(8, 8))
    
    # Create the scatter plot with Seaborn
    sns.scatterplot(x=y_test, y=y_pred, color='blue', s=30, edgecolor='w', alpha=0.3)
    
    # Set the labels and the title
    plt.xlabel('Correct Labels: y_test')
    plt.ylabel('Predicted Labels: y_pred')
    plt.title('Correct vs Predicted Labels')
    
    # Add grid
    plt.grid(True)
    
    # Return of the figure
    return plt.gcf()

