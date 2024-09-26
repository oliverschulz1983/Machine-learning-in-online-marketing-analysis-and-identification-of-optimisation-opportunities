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

# Funktion, um den gefilterten Datensatz zu bekommen
def get_data(campaign):
    mldata = pd.read_csv("mldata.csv")

    # Preprocessing der LinReg mit "clicks" als Target

    # Alle Kampagnen
    mldata_camp_total = mldata

    # Filter auf Camp 1
    mldata_camp_1 = mldata[mldata["campaign_number"] == "camp 1"]

    # Filter auf Camp 2
    mldata_camp_2 = mldata[mldata["campaign_number"] == "camp 2"]

    # Filter auf Camp 3
    mldata_camp_3 = mldata[mldata["campaign_number"] == "camp 3"]


    # Selektion ausgewählter, für ML relevanter, features - ohne "campaign_number" als feature - mit "clicks" als Label:

    # Mit allen Kampagnen
    mldata_camp_total = mldata_camp_total[["banner", "placement", "ad_spend_by_company_x", "post_click_sales_amount"]].reset_index(drop=True)

    # Gefiltert auf Camp 1
    mldata_camp_1 = mldata_camp_1[["banner", "placement", "ad_spend_by_company_x", "post_click_sales_amount"]].reset_index(drop=True)

    # Gefiltert auf Camp 2
    mldata_camp_2 = mldata_camp_2[["banner", "placement", "ad_spend_by_company_x", "post_click_sales_amount"]].reset_index(drop=True)

    # Gefiltert auf Camp 3
    mldata_camp_3 = mldata_camp_3[["banner", "placement", "ad_spend_by_company_x", "post_click_sales_amount"]].reset_index(drop=True)

    if campaign == "Campaign 1":
        return mldata_camp_1
    elif campaign == "Campaign 2":
        return mldata_camp_2
    elif campaign == "Campaign 3":
        return mldata_camp_3
    elif campaign == "all Campaigns":
        return mldata_camp_total


# Funktion, um das Modell und den Scaler zu bekommen
def get_model(campaign):
    data = get_data(campaign)

    # Definieren der Features(X) und des Labels(y) 
    X = data[["banner", "placement", "ad_spend_by_company_x"]]
    y = data[["post_click_sales_amount"]]

    # Erzeugen der Dummys für die kategoralen Variablen
    X = pd.get_dummies(data=X, columns=["banner"], prefix="banner", dtype=float, drop_first=True)
    X = pd.get_dummies(data=X, columns=["placement"], prefix="placement", dtype=float, drop_first=True)

    # Standardisierung: Sehr viele Variablen in X, mit unterschiedlichen Skalen/Skalierungen => Daher Standardisierung der Daten
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Erzeugen der Trainings- und Test-Datensätze
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.33, random_state=100)

    # Initialisieren des Modells für die Ridge Regression - 
    ridge_reg = Ridge(alpha=0.1, solver="cholesky")

    # Trainieren des Modells
    ridge_reg.fit(X_train, y_train)

    return ridge_reg, scaler, X, y, X_train, X_test, y_train, y_test, scaled_X


# Funktion, um die Vorhersage zu erstellen
def get_prediction(campaign, adspend, banner_values, placement_values):

    ridge_reg, scaler, X, y, X_train, X_test, y_train, y_test, scaled_X = get_model(campaign)

    ### Komplexität hinzufügen: Welche Banner / Placement? abh. von Camp? Selbst Auswahl? ###
    prediction = pd.DataFrame([[adspend, banner_values["bv2"]/100, banner_values["bv3"]/100, banner_values["bv4"]/100, banner_values["bv5"]/100, banner_values["bv6"]/100, banner_values["bv7"]/100, banner_values["bv8"]/100, placement_values["pv2"]/100, placement_values["pv3"]/100, placement_values["pv4"]/100, placement_values["pv5"]/100]], columns=X.columns)
    scaled_prediction_data = scaler.transform(prediction)

    # Vorhersage machen und formatieren
    predicted_value = ridge_reg.predict(scaled_prediction_data)

    def format_prediction(predicted_value):
        # Extrahiere den tatsächlichen Vorhersagewert aus dem Array und formatiere ihn
        return f'{predicted_value[0][0]:,.2f}'

    # Formatierte Vorhersage zurückgeben
    return format_prediction(predicted_value) # gibt mir die exakte Vorhersage


# Funktion, um die Regressions-Ergebnisse zu bekommen
def get_regression_results(campaign):

    ridge_reg, scaler, X, y, X_train, X_test, y_train, y_test, scaled_X = get_model(campaign)

    # Vorhersage treffen
    y_pred = ridge_reg.predict(X_test)

    # Modell Performance für Testdatensatz:
    r2 = r2_score(y_test, y_pred) # Berechnung des R² Bestimmtheitsmaß'

    # Berechnung des RMSE Fehlermaß
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    # OLS Regression Results: Modell anpassen
    model = sm.OLS(y_train, sm.add_constant(pd.DataFrame(X_train, columns=scaled_X.columns)))
    results = model.fit().summary()

    return rmse, r2, results

# Funktion, um den Scatter-Plot für die Vorhersagegenauigkeit zu erstellen
def create_seaborn_scatter_plot(y_test, y_pred):
    # Sicherstellen, dass die Arrays eindimensional sind
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)

    # Erstellen einer neuen Figur
    plt.figure(figsize=(8, 8))
    
    # Erstellen des Scatter Plots mit Seaborn
    sns.scatterplot(x=y_test, y=y_pred, color='blue', s=30, edgecolor='w', alpha=0.3)
    
    # Setzen Sie die Labels und den Titel
    plt.xlabel('Correct Labels: y_test')
    plt.ylabel('Predicted Labels: y_pred')
    plt.title('Correct vs Predicted Labels')
    
    # Optional: Fügen Sie ein Gitter hinzu
    plt.grid(True)
    
    # Rückgabe der Figur
    return plt.gcf()

