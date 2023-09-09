#Initialize all important libaries
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sklearn


#Initialize all page settings and environment vairables
st.set_page_config(
    page_title="IconicHomes Estate Master",
    page_icon=":)"
)

hide_streamlit_style = """
<style>
footer {
    visibility: hidden;
}

footer:after {
    content: 'Real Estate Master - The Ultimate Estate Valuation App';  /* Replace 'Your Custom Footer Text' with your own text */
    visibility: visible;
    display: block;
    position: relative;
    padding: 5px;
    top: 2px;
}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)



# Create a Streamlit app
st.title('Welcome to Iconic 🏡 Estate Master')
st.write('Your ultimate Estate valuation app.')
st.sidebar.success("Select a page above")
st.write("")
st.write("__________________________________________________________________")
st.write("This app will demonstrate how certain indices of deprivation play a significant role in influencing house prices in various Lower Super Output Areas across the United Kingdom (UK). Ultimately, this app aims to predict the median house price of any region based on its indices of deprivation.")
st.write("This app was built with steamlit a library of designing web applications with python")
st.write("______________________________________________________________________")


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
    
    #Intiatlize all values when this page is loaded first time
    st.session_state["lsoa_name"] = ""
    st.session_state["lsoa_code"] = ""
    st.session_state["Actual_Price"] = 0.0000
    st.session_state["Reported_Price"] = 0.0000
    st.session_state["data"] = initialize_blank_lsoa_data()  # Initialize a list with 17 zeros



    if "model_name" not in list(st.session_state.keys()):
        st.session_state["model_name"] = "RFM"

    # Initialize all features of the original outlier-free dataset
    all_features = ['HousePrice','Reported_Price','IMDScore', 'IncDepriv', 'EmpDepriv', 'EduSklDepriv', 'HealthDepriv', 'CrimScore',
                    'HousServDepriv', 'LivEnvDepriv', 'ChildIncDepriv', 'OldPplIncDepriv', 'ChildYPDepriv',
                    'AdultSklDepriv', 'GeoBarDepriv', 'WiderBarDepriv', 'IndoorDepriv', 'OutdoorDepriv']

    # Define a function to read CSV into a DataFrame from a given path
    def read_data(filename):
        df = pd.read_csv(filename)
        return df
    


    # Call the function to read your CSV files
    df_main = read_data("Data/data_main.csv")  # Read Main Project data
    list_of_lsoa = df_main['LSOA_CodeName'].unique().tolist()# get list of values for lsoa


    # Display a dropdown with LSOA values from the table
    lsoa_name = st.selectbox("Please select your Lower Super Output Area (LSOA) or proceed by clicking 'Explore' without selecting one.", ['--Select your area--'] +df_main['LSOA_CodeName'].unique().tolist())
    #Select Model
    # Define a list of options for the radio button
    options = ["RFM", "GBM"] #RF is random forest model and GBM is gradient boost model
    # Create a radio button using st.radio
    selected_option = st.radio("Select an option:", options,help='Random Forest Model (RFM) Or Gradient Boost Model (GBM)')
    submit = st.button("Explore")
    

    if submit:
        if "Select" not in lsoa_name:
            st.session_state["lsoa_name"] = lsoa_name
            lsoa_code = lsoa_name.split('-')[0]  # Extract the first item in the split string of the LSOA name as it is intended to be the LSOA code
            df_filtered = df_main[df_main['LSOACode'] == lsoa_code]
            X = df_filtered.iloc[0].to_frame().T
            X = X.reset_index()
            c = ['IMDScore', 'IncDepriv', 'EmpDepriv', 'EduSklDepriv', 'HealthDepriv', 'CrimScore',
                    'HousServDepriv', 'LivEnvDepriv', 'ChildIncDepriv', 'OldPplIncDepriv', 'ChildYPDepriv',
                    'AdultSklDepriv', 'GeoBarDepriv', 'WiderBarDepriv', 'IndoorDepriv', 'OutdoorDepriv']
            X = X[all_features]
            st.session_state["data"] = X[c]
            st.session_state["lsoa_code"] = lsoa_code
            st.session_state["Actual_Price"] = X.loc[0, 'HousePrice']
            st.session_state["Reported_Price"] = X.loc[0, 'Reported_Price']
            st.session_state["model_name"] = selected_option
            st.write("You have entered:", lsoa_name)

        else:
            st.session_state["lsoa_name"] = ""
            st.session_state["lsoa_code"] = ""
            st.session_state["Actual_Price"] = 0.0000
            st.session_state["Reported_Price"] = 0.0000
            st.session_state["data"] = initialize_blank_lsoa_data() # Initialize a list with 17 zeros
            st.session_state["model_name"] = selected_option
            
        from streamlit_extras.switch_page_button import switch_page
        switch_page("app")  # Navigate to the "app" page



except Exception as e:
    st.write(e)
    st.write(f"<span style='font-size: 24px;'>Opps! Sorry we are unable to retrieve data at this time.</span>", unsafe_allow_html=True)

