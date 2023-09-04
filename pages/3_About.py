import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sklearn

st.title("About our app & data")
def get_model_performance_metrics(modelname):
    from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
    from sklearn.metrics import mean_absolute_error
    import pandas as pd
    import os
    import numpy as np
    directory_path = 'Data/PreTrained'
    dfPred = pd.read_csv(os.path.join(directory_path, modelname+"_Predicted.csv"), header=0, delimiter=',')
    dfTest = pd.read_csv(os.path.join(directory_path, modelname+"_Test.csv"), header=0, delimiter=',')
    y_pred = dfPred['Value'].values
    y_test = dfTest['Value'].values
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    st.write("Mean Squared Error:", round(mse, 2))
    st.write("R-squared:", round(r2,2))
    st.write("Mean Absolute Error:", round(mae,2))
    st.write("Root Mean Squared Error:", round(rmse,2))


# Print versions
st.write("")
st.write("----------------------------------")
st.write("<span style='font-size: 1.2em; font-weight: bold;'>Libaries used :</span>", unsafe_allow_html=True)
st.write("NumPy version:", np.__version__)
st.write("Pandas version:", pd.__version__)
st.write("Streamlit version:", st.__version__)
st.write("Joblib version:", joblib.__version__)
st.write("Scikit-learn version:", sklearn.__version__)
st.write("----------------------------------")
st.write("")
st.write("<span style='font-size: 1.2em; font-weight: bold;'>Our data :</span>", unsafe_allow_html=True)
# Read the content of the text file
with open("Resources/About_Dataset.txt", "r") as file:
    about_dataset_text = file.read()

# Display the content using st.write()
st.write(about_dataset_text)
st.markdown("""[Link to download median house price data set](https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/medianpricepaidbylowerlayersuperoutputareahpssadataset46)""")
st.markdown("""[Link to download english indices of deprivation data set](https://opendatacommunities.org/resource?uri=http%3A%2F%2Fopendatacommunities.org%2Fdata%2Fsocietal-wellbeing%2Fimd%2Findicesbyla)""")
st.write("Data Features :")
# Define your feature data
features_data = [
{'FeatureName': 'IMDScore', 'FullName': 'Index of Multiple Deprivation IMDScore', 'Description': 'This deprivation component is weighted with different strengths and compiled into a single deprivation score (ONS, 2021). The weights were compiled from domains 1 to 7'},
{'FeatureName': 'IncDepriv', 'FullName': 'Income Deprivation Domain', 'Description': 'Income deprivation measures the proportion of adults who are income-deprived (ONS, 2021).'},
{'FeatureName': 'EmpDepriv', 'FullName': 'Employment Deprivation Domain', 'Description': 'This measures the proportion of the working-age population in an area involuntarily excluded from the labor market (ONS, 2021).'},
{'FeatureName': 'EduSklDepriv', 'FullName': 'Education, Skills and Training Deprivation', 'Description': 'This domain index measures the lack of attainment and skills in the local population. The indicators fall into two sub-domains: one relating to children and young people, and one relating to adult skills (ONS, 2021).'},
{'FeatureName': 'CrimScore', 'FullName': 'Crime Domain', 'Description': 'The domain measures the risk of personal and material victimization at the local level.'},
{'FeatureName': 'HousServDepriv', 'FullName': 'Barriers to Housing and Services Domain', 'Description': 'This measures the physical and financial accessibility of housing and local services (ONS, 2021).'},
{'FeatureName': 'HealthDepriv', 'FullName': 'Health Deprivation and Disability Domain', 'Description': 'This measures the risk of premature death and the impairment of quality of life through poor physical or mental health (ONS, 2021).'},
{'FeatureName': 'LivEnvDepriv', 'FullName': 'Living Environment Deprivation Domain', 'Description': 'This measures the quality of the local environment (ONS, 2021). The indicators fall into two sub-domains: one for the indoors living environment and one for the outdoors living environment.'},
{'FeatureName': 'ChildIncDepriv', 'FullName': 'Income Deprivation Affecting Children Index', 'Description': 'This measures the proportion of all children aged 0 to 15 living in income-deprived families (ONS, 2021). Family is used here to indicate a ‘benefit unit’, that is the claimant, any partner and any dependent children for whom Child Benefit is received. This is one of two supplementary indices and is a sub-set of the Income Deprivation Domain.'},
{'FeatureName': 'OldPplIncDepriv', 'FullName': 'Income Deprivation Affecting Older People Index', 'Description': 'This measures the proportion of all those aged 60 or over who experience income deprivation (ONS, 2021). It is a subdomain of the Income Deprivation Domain.'},
{'FeatureName': 'ChildYPDepriv', 'FullName': 'Children and Young People Subdomain Score', 'Description': 'This sub-domain measures the attainment of qualifications and associated measures. It is one of two sub-domains of the Education, Skills, and Training Deprivation domain. The more deprived an area is, the higher the score (ONS, 2021).'},
{'FeatureName': 'AdultSklDepriv', 'FullName': 'Adult Skills Deprivation Domain', 'Description': 'This is a subdomain of the Education, Skills, and Training Deprivation domain. It measures the lack of qualifications in the resident working-age adult population. The more deprived an area is, the higher the score (ONS, 2021).'},
{'FeatureName': 'GeoBarDepriv', 'FullName': 'Geographical Barriers Subdomain Score', 'Description': 'This relates to the physical proximity of local services. The more deprived an area is, the higher the score (ONS, 2021).'},
{'FeatureName': 'WiderBarDepriv', 'FullName': 'Wider Barriers Subdomain Score', 'Description': 'This measures issues relating to access to housing such as affordability. One of two sub-domains of the Barriers to Housing and Services domain. The more deprived the area, the higher the score (ONS, 2021).'},
{'FeatureName': 'IndoorDepriv', 'FullName': 'Indoors Deprivation Domain', 'Description': 'This measures the quality of housing and is a sub-domain of the Living Environment Deprivation domain. The more deprived an area is, the higher the score.'},
{'FeatureName': 'OutdoorDepriv', 'FullName': 'Outdoors Deprivation Domain', 'Description': 'This measures air quality and road traffic accidents, which is one of two sub-domains of the Living Environment Deprivation domain. The more deprived an area is, the higher the score.'}
]
displaydf = pd.DataFrame(features_data)
# Display the DataFrame
st.write(displaydf)
st.markdown("""[Read more about these features](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/853811/IoD2019_FAQ_v4.pdf)""")


st.write("")
st.write("----------------------------------")
st.write("")

  # Import the RandomForestClassifier if it's a classifier, or RandomForestRegressor for regression

st.write("<span style='font-size: 1.2em; font-weight: bold;'>Our Models :</span>", unsafe_allow_html=True)
st.write("Ranom Forest Regressor:")
# Load the Random Forest model
model = joblib.load('Models/xRFTpipe.joblib')
# Get all the parameters of the model
model_params = model.get_params()
# Print the model parameters
st.write(" | ".join([f"{param}: {value}" for param, value in model_params.items()]))
get_model_performance_metrics('RFM')

st.write("----------------------------------")

# Load the Gradient boost model
gbst_model = joblib.load('Models/xGBSTpipe.joblib')
# Get all the parameters of the Gradient boost model
gbst_params = gbst_model.get_params()
# Display the gbst model parameters in one line
st.write("Gradient Boost:")
st.write(" | ".join([f"{param}: {value}" for param, value in gbst_params.items()]))
get_model_performance_metrics('GBM')
st.write("----------------------------------")
# Load the KMeans model
kmeans_model = joblib.load('Models/xKMNSpipe.joblib')
# Get all the parameters of the KMeans model
kmeans_params = kmeans_model.get_params()
# Display the KMeans model parameters in one line
st.write("KMeans:")
st.write(" | ".join([f"{param}: {value}" for param, value in kmeans_params.items()]))
st.write("----------------------------------")
st.write("")
st.write("")
st.markdown("""[App built by MolA](https://www.youtube.com/@masteroflogic)""")
st.write("Version 1.001")


