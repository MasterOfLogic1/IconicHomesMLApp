import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sklearn

try:
    # feature dictionary
    feature_dict = {
    'IMDScore': 'This deprivation component is weighted with different strengths and compiled into a single deprivation score (ONS, 2021). The weights were compiled from domains 1 to 7',
    'IncDepriv': 'Income deprivation measures the proportion of adults who are income-deprived (ONS, 2021).',
    'EmpDepriv': 'This measures the proportion of the working-age population in an area involuntarily excluded from the labor market (ONS, 2021).',
    'EduSklDepriv': 'This domain index measures the lack of attainment and skills in the local population. The indicators fall into two sub-domains: one relating to children and young people, and one relating to adult skills (ONS, 2021).',
    'CrimScore': 'The domain measures the risk of personal and material victimization at the local level.',
    'HousServDepriv': 'This measures the physical and financial accessibility of housing and local services (ONS, 2021).',
    'HealthDepriv': 'This measures the risk of premature death and the impairment of quality of life through poor physical or mental health (ONS, 2021).',
    'LivEnvDepriv': 'This measures the quality of the local environment (ONS, 2021). The indicators fall into two sub-domains: one for the indoors living environment and one for the outdoors living environment.',
    'ChildIncDepriv': 'This measures the proportion of all children aged 0 to 15 living in income-deprived families (ONS, 2021). Family is used here to indicate a ‘benefit unit’, that is the claimant, any partner and any dependent children for whom Child Benefit is received. This is one of two supplementary indices and is a sub-set of the Income Deprivation Domain.',
    'OldPplIncDepriv': 'This measures the proportion of all those aged 60 or over who experience income deprivation (ONS, 2021). It is a subdomain of the Income Deprivation Domain.',
    'ChildYPDepriv': 'This sub-domain measures the attainment of qualifications and associated measures. It is one of two sub-domains of the Education, Skills, and Training Deprivation domain. The more deprived an area is, the higher the score (ONS, 2021).',
    'AdultSklDepriv': 'This is a subdomain of the Education, Skills, and Training Deprivation domain. It measures the lack of qualifications in the resident working-age adult population. The more deprived an area is, the higher the score (ONS, 2021).',
    'GeoBarDepriv': 'This relates to the physical proximity of local services. The more deprived an area is, the higher the score (ONS, 2021).',
    'WiderBarDepriv': 'This measures issues relating to access to housing such as affordability. One of two sub-domains of the Barriers to Housing and Services domain. The more deprived the area, the higher the score (ONS, 2021).',
    'IndoorDepriv': 'This measures the quality of housing and is a sub-domain of the Living Environment Deprivation domain. The more deprived an area is, the higher the score.',
    'OutdoorDepriv': 'This measures air quality and road traffic accidents, which is one of two sub-domains of the Living Environment Deprivation domain. The more deprived an area is, the higher the score.'
    }

    model_name_dict = {'RFM':'Random Forest Model','GBM':'Gradient Boost Model'}
    model_desc = {'RFM':'an ensemble learning method that combines multiple decision trees to make more accurate predictions.','GBM':'works by building an ensemble of decision trees, where each tree corrects the errors made by the previous ones.'}

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
        st.text("Option: " + model_name_dict[st.session_state["model_name"]],help=model_desc[st.session_state["model_name"]])
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
        slider_values[feature] = st.slider(f'{feature} Index', -100.0000, 100.0000, default_value, step=0.1,help=feature_dict[feature])
        st.markdown(f'<p style="color: DodgerBlue;">Selected  &nbsp; {feature} Index &nbsp;: &nbsp; {slider_values[feature]:.1f}</p>', unsafe_allow_html=True)
        st.text("")
        st.text("")


    if predictButton:
        from streamlit_extras.switch_page_button import switch_page
        switch_page("terminal")  # Navigate to the "app" page

except Exception as e:
    st.write(f"<span style='font-size: 24px;'>Opps! something went wrong while processing your request please refresh the page.</span>", unsafe_allow_html=True)
    st.write(f"Opps something went wrong while processing your request. Error: {e}")
