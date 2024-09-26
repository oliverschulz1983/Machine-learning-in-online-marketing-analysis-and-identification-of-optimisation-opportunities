# Start with:
# streamlit run main.py

import streamlit as st
import pandas as pd

import forecast

st.set_page_config(layout="wide") # Configure the Streamlit layout

# Main heading of the web app
st.title("Sales forecast for online advertising campaigns by company X")

# Displaying the sidebar
st.sidebar.subheader("What sales amount will you realise? Find out!")

# Selection of the data set
campaign = st.sidebar.radio('Which campaign data should be used to make a prediction:', ['Campaign 1','Campaign 2', 'Campaign 3', 'all Campaigns'])

# Layout of the main page with two columns
col1, col2 = st.columns([4, 5])

# Initialisation of the banner_values and placement_values in session_state
def initialize_session_state():
    if 'banner_values' not in st.session_state:
        st.session_state.banner_values = {f"bv{i}": 12.5 for i in range(1, 9)}
    if 'placement_values' not in st.session_state:
        st.session_state.placement_values = {f"pv{i}": 20.0 for i in range(1, 6)}

# Function to adjust the banner-slider values so that the sum of the sliders remains at 100%
def adjust_other_banner_sliders(changed_key, changed_value):
    total = sum(st.session_state.banner_values.values())
    if total != 100.0:
        difference = 100.0 - total
        other_keys = [key for key in st.session_state.banner_values if key != changed_key]
        for key in other_keys:
            st.session_state.banner_values[key] += difference / (len(other_keys))
            if st.session_state.banner_values[key] < 0.0:
                st.session_state.banner_values[key] = 0.0

# Function to adjust the placement-slider values so that the sum of the sliders remains at 100%
def adjust_other_placement_sliders(changed_key, changed_value):
    total = sum(st.session_state.placement_values.values())
    if total != 100.0:
        difference = 100.0 - total
        other_keys = [key for key in st.session_state.placement_values if key != changed_key]
        for key in other_keys:
            st.session_state.placement_values[key] += difference / (len(other_keys))
            if st.session_state.placement_values[key] < 0.0:
                st.session_state.placement_values[key] = 0.0

# Function that controls what happens when one of the banner-slider values is changed
def update_banner_value(key):
    changed_value = st.session_state[f"slider{key}"]
    st.session_state.banner_values[key] = changed_value
    adjust_other_banner_sliders(key, changed_value)

# Function that controls what happens when one of the placement-slider values is changed
def update_placement_value(key):
    changed_value = st.session_state[f"slider{key}"]
    st.session_state.placement_values[key] = changed_value
    adjust_other_placement_sliders(key, changed_value)

# Initialise banner_values and placement_values in session_state
initialize_session_state()

# Define what appears in the first column
with col1:
    
    # Show by data set selection
    if campaign:
        adspend = st.sidebar.number_input('How much advertising budget (in USD) do you have available?')

    # Show after ad spend entry 
    if adspend:
        
        st.info(":red[Please note: You can use the sliders below to adjust the budget via the ad formats and placements and thus try to optimise sales. You can use the coefficients in the regression results in the right-hand column as a guide.]")

        st.subheader("Determine the proportions of the budget to be distributed via the following ad :red[formats]:")
        
        # initialise banner_names
        banner_names = [
            "Banner 160 x 600", "Banner 240 x 400", "Banner 300 x 250", "Banner 468 x 60",
            "Banner 580 x 400", "Banner 670 x 90", "Banner 728 x 90", "Banner 800 x 250"
        ]

        # Loop that generates all banner sliders
        for i, name in enumerate(banner_names, 1):
            key = f"bv{i}"
            st.slider(
                name,
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.banner_values[key],
                step=0.1,
                format="%.1f%%",
                key=f"slider{key}",
                on_change=update_banner_value,
                args=(key,)
            )


        st.subheader("Determine the proportions of the budget to be distributed via the following ad :red[placements]:")

        # initialise placement_names
        placement_names = [
            "Placement 'abc'", "Placement 'def'", "Placement 'ghi'", "Placement 'jkl'",
            "Placement 'mno'"
        ]

        # Loop that generates all placement sliders
        for i, name in enumerate(placement_names, 1):
            key = f"pv{i}"
            st.slider(
                name,
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.placement_values[key],
                step=0.1,
                format="%.1f%%",
                key=f"slider{key}",
                on_change=update_placement_value,
                args=(key,)
            )

        st.sidebar.subheader("Sales-Amount-Prediction:")
        
        # Generation of the sales forecast
        prediction = forecast.get_prediction(campaign, adspend, st.session_state.banner_values, st.session_state.placement_values)

        # Display the forecast
        st.sidebar.title(f":red[{prediction} USD.]")

        st.sidebar.write(f"With an advertising budget of {adspend:,.2f} USD for {campaign} you will probably generate this amount of sales.")

# Define what appears in the second column
with col2:
       
    # Show when a campaign is selected in the sidebar
    if campaign:

        # Get the regression results
        rmse, r2, results = forecast.get_regression_results(campaign)

        d = {"RMSE": rmse, "R2 Score": r2}
        df = pd.DataFrame(data=d, index=[0])
        df_rounded = df.round(decimals=3).T

        # Output regression results
        st.header(f":blue[Model performance of the Ridge Regression for {campaign}:]")
        st.info(""":red[You can use the quality metrics and coefficients of the respective regression to optimise your campaign results.]""")
        st.subheader(f":blue[Test set:]")
        #st.write(f"RMSE: {rmse:,.3f}\n\nR2 Score: {r2:.3f}")
        st.write(df_rounded)
        st.subheader(f":blue[Training set:]")
        st.info("Caution: Please be aware that the coefficient values for the banner ‘160 x 600’ and for the placement ‘abc’ are missing, as these were omitted during the dummy creation and now act as reference values.")
        st.write(results)

        ridge_reg, scaler, X, y, X_train, X_test, y_train, y_test, scaled_X = forecast.get_model(campaign)

        # Make predictions
        y_pred = ridge_reg.predict(X_test)

        # Create a scatter plot that visualises how the predicted labels relate to the correct labels
        st.write(forecast.create_seaborn_scatter_plot(y_test, y_pred))

