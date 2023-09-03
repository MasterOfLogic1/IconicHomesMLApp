import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sklearn

try:
    # Initialize blank lsoa data
    def initialize_blank_lsoa_data():
        columns = ['IMDScore', 'IncDepriv', 'EmpDepriv', 'EduSklDepriv', 'HealthDepriv',
           'CrimScore', 'HousServDepriv', 'LivEnvDepriv', 'ChildIncDepriv', 'OldPplIncDepriv', 'ChildYPDepriv',
           'AdultSklDepriv', 'GeoBarDepriv', 'WiderBarDepriv', 'IndoorDepriv', 'OutdoorDepriv']
        # Create a DataFrame with one row and all values set to 0.0
        data = [[0.0] * len(columns)]
        df = pd.DataFrame(data, columns=columns)
        return df
    

    def predict():
        s_f = [
        'IMDScore', 'IncDepriv', 'EmpDepriv', 'EduSklDepriv', 'HealthDepriv', 'CrimScore',
        'HousServDepriv', 'LivEnvDepriv', 'ChildIncDepriv', 'OldPplIncDepriv', 'ChildYPDepriv',
        'AdultSklDepriv', 'GeoBarDepriv', 'WiderBarDepriv', 'IndoorDepriv', 'OutdoorDepriv']
        row = [slider_values[feature] for feature in s_f]
        X = pd.DataFrame([row], columns=s_f)
        st.session_state["input_values"] = X
        prediction = model.predict(X)[0]
        st.session_state["Predicted_Price"] = prediction


    if "lsoa_name" not in list(st.session_state.keys()):
        st.session_state["lsoa_name"] = ""
        st.session_state["Actual_Price"] = 0.0000
        st.session_state["data"] = initialize_blank_lsoa_data()
        st.session_state["Predicted_Price"] = 0.0000



    if "model_name" not in list(st.session_state.keys()):
        st.session_state["model_name"] = "RFM"

    selected_features = [
        'IMDScore', 'IncDepriv', 'EmpDepriv', 'EduSklDepriv', 'HealthDepriv', 'CrimScore',
        'HousServDepriv', 'LivEnvDepriv', 'ChildIncDepriv', 'OldPplIncDepriv', 'ChildYPDepriv',
        'AdultSklDepriv', 'GeoBarDepriv', 'WiderBarDepriv', 'IndoorDepriv', 'OutdoorDepriv'
    ]

    if st.session_state["model_name"] == "RFM":
        # Use my random forest model
        model = joblib.load('Models/xRFTpipe.joblib')
    elif st.session_state["model_name"] == "GBM":
        # Use my Gradient boost model
        model = joblib.load('Models/xGBSTpipe.joblib')
    else:
        # Throw an exception if an unexpected option is selected
        raise ValueError("Invalid model selection. Please select RFM or GBM.")

    backButton = st.button('Change Your Area')

    if backButton:
        from streamlit_extras.switch_page_button import switch_page
        st.session_state["lsoa_name"] = ""
        st.session_state["Actual_Price"] = 0.0000
        st.session_state["data"] = initialize_blank_lsoa_data()
        st.session_state["Predicted_Price"] = 0.0000
        switch_page("homepage")  # Navigate to the "app" page

    st.title('Area Deprivation Indices')
    st.text('________________________________________________________________________________')
    st.text('Here, you can use the sliders to adjust any deprivation index of your area,')
    st.text('and click the predict button to estimate your house price.')
    st.text('________________________________________________________________________________')
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text("Option: " + st.session_state["model_name"])
    with col2:
        predictButton = st.button('Predict 🚀', on_click=predict, key="Predict_Button")

    st.text('________________________________________________________________________________')
    if st.session_state["lsoa_name"] != "":
        st.text("Area : " + st.session_state["lsoa_name"])
    
    st.markdown("<span style='color: orange;'>The feature sliders are arranged according to the models feature importance.</span>", unsafe_allow_html=True)
    st.text('________________________________________________________________________________')

    # Automatically get feature importance from the model
    if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
        sorted_feature_indices = np.argsort(feature_importance)[::-1]
        selected_features = [selected_features[i] for i in sorted_feature_indices]

    slider_values = {}


    #st.write(st.session_state["data"].head())
    for feature in selected_features:
        default_value = st.session_state["data"].loc[0, feature]
        slider_values[feature] = st.slider(f'{feature} Index', -100.0000, 100.0000, default_value, step=0.1)
        st.markdown(f'<p style="color: DodgerBlue;">Selected  &nbsp; {feature} Index &nbsp;: &nbsp; {slider_values[feature]:.1f}</p>', unsafe_allow_html=True)
        st.text("")
        st.text("")


    if predictButton:
        from streamlit_extras.switch_page_button import switch_page
        switch_page("terminal")  # Navigate to the "app" page

except Exception as e:
    st.write(f"<span style='font-size: 24px;'>Opps! something went wrong while processing your request please refresh the page.</span>", unsafe_allow_html=True)
    st.write(f"Opps something went wrong while processing your request. Error: {e}")
