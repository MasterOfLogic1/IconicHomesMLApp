import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sklearn
import re
import matplotlib.pyplot as plt




backButton = st.button('Change Your Area')
st.title('Terminal')
# Initialize blank lsoa data
def initialize_blank_lsoa_data():
    columns = ['IMDScore', 'IncDepriv', 'EmpDepriv', 'EduSklDepriv', 'HealthDepriv',
           'CrimScore', 'HousServDepriv', 'LivEnvDepriv', 'ChildIncDepriv', 'OldPplIncDepriv', 'ChildYPDepriv',
           'AdultSklDepriv', 'GeoBarDepriv', 'WiderBarDepriv', 'IndoorDepriv', 'OutdoorDepriv']
    # Create a DataFrame with one row and all values set to 0.0
    data = [[0.0] * len(columns)]
    df = pd.DataFrame(data, columns=columns)
    return df


# gets rmse
def get_rmse(modelname):
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    import os
    import numpy as np
    directory_path = 'Data\PreTrained'
    dfPred = pd.read_csv(os.path.join(directory_path, modelname+"_Predicted.csv"), header=0, delimiter=',')
    dfTest = pd.read_csv(os.path.join(directory_path, modelname+"_Test.csv"), header=0, delimiter=',')
    y_pred = dfPred['Value'].values
    y_test = dfTest['Value'].values
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse


# Initialize Model Performace metrics
def get_model_performance_metrics(modelname):
    st.write("<span style='font-size: 24px; font-weight: bold;'>Model Overall Performance :</span>", unsafe_allow_html=True)
    from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
    from sklearn.metrics import mean_absolute_error
    import pandas as pd
    import os
    import numpy as np
    directory_path = 'Data\PreTrained'
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

if backButton:
    from streamlit_extras.switch_page_button import switch_page
    st.session_state["lsoa_name"] = ""
    st.session_state["Actual_Price"] = 0.0000
    st.session_state["data"] = initialize_blank_lsoa_data()
    st.session_state["Predicted_Price"] = 0.0000
    switch_page("homepage")  # Navigate to the "app" page


try:
    # Define a function to read CSV into a DataFrame from a given path
    def read_data(filename):
        df = pd.read_csv(filename)
        return df
    

    clusterModel = joblib.load('Models/xKMNSpipe.joblib') #use my kmeans model always


    # Call the function to read your CSV files
    df_trend = read_data("Data/data_trend.csv")  # Read trend data
    df_main = read_data("Data/data_main.csv")  # Read trend data
    trend_features = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010', '2011', '2012','2013', '2014', '2015', '2016', '2017', '2018','2019', '2020', '2021']

    selected_features = ['IMDScore', 'IncDepriv', 'EmpDepriv', 'EduSklDepriv', 'HealthDepriv', 'CrimScore',
                         'HousServDepriv', 'LivEnvDepriv', 'ChildIncDepriv', 'OldPplIncDepriv', 'ChildYPDepriv',
                         'AdultSklDepriv', 'GeoBarDepriv', 'WiderBarDepriv', 'IndoorDepriv', 'OutdoorDepriv']


    def get_min_max_val(fts):
        # Call the function to read your CSV files
        df = read_data("Data/data_main.csv")# Read Main Project data
        min_max_dict = {}
        # Iterate through each column and calculate min-max values
        for column in df[fts].columns:
            min_value = df[column].min()
            max_value = df[column].max()
            min_max_dict[column] = [min_value, max_value]

        #return value
        return min_max_dict



    def trend_graph(row,trend_features):
        st.write("<span style='font-size: 24px; font-weight: bold;'>House Price Trend :</span>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))  # Increase the width (and height if needed)
        ax.set_facecolor("lightgreen")  # Set the background color to green
        # Plot your data with a green line
        row[trend_features].plot.line(ax=ax, color="green")
        # Customize the plot further if needed
        ax.set_xlabel('Year')
        ax.set_ylabel('House price (£)')
        ax.set_title('How house prices have grown over time in '+st.session_state["lsoa_name"])
        # Display the Matplotlib plot in Streamlit
        return st.pyplot(fig)
    
    def color_cluster(df):
        # Calculate the mean house prices for each cluster
        cluster_mean_prices = df.groupby('Cluster')['HousePrice'].mean().reset_index()
        # Sort the clusters by mean house price
        sorted_clusters = cluster_mean_prices.sort_values('HousePrice')
        # Define colors for the clusters
        colors = {}
        colors[sorted_clusters.iloc[0]['Cluster']] = 'red'  # Lowest mean price
        colors[sorted_clusters.iloc[1]['Cluster']] = 'orange'  # Second lowest mean price
        colors[sorted_clusters.iloc[-1]['Cluster']] = 'green'  # Highest mean price
        colors[sorted_clusters.iloc[-2]['Cluster']] = 'green'  # Second highest mean price
        return colors
    

    def cluster_information(df):
        # Calculate the mean house prices for each cluster
        cluster_mean_prices = df.groupby('Cluster')['HousePrice'].mean().reset_index()
        # Sort the clusters by mean house price
        sorted_clusters = cluster_mean_prices.sort_values('HousePrice')
        # Define colors for the clusters
        c_info = {}
        c_info[sorted_clusters.iloc[0]['Cluster']] = "low"  # Lowest mean price
        c_info[sorted_clusters.iloc[1]['Cluster']] = "moderate"  # Second lowest mean price
        c_info[sorted_clusters.iloc[-1]['Cluster']] = "very high"  # Highest mean price
        c_info[sorted_clusters.iloc[-2]['Cluster']] = "high"  # Second highest mean price
        return c_info
    

    
    def cluster_graph(df,df2,selected_cluster):
        # First we need to get all clusters
        clusterIndex = np.sort(df["Cluster"].unique())
        clusterName = []
        results = []
        cluster_colours = color_cluster(df2)

        for cl in clusterIndex:
            dtTemp = df[df["Cluster"] == cl]
            median_value = dtTemp['2021'].median()
            results.append(median_value)
            cluster_label = "K " + str(cl + 1)
            clusterName.append(cluster_label)

        # Find the index of the selected_cluster
        selected_index = np.where(clusterIndex == selected_cluster)[0][0]
        # Set the color for the selected cluster to highlight it
        colors = ['grey' if i != selected_index else cluster_colours[selected_index] for i in range(len(clusterName))]

        # Add "Your Area" to the selected cluster
        clusterName[selected_index] = "Your Area (K " + str(selected_cluster + 1) + ")"

        fig, ax = plt.subplots(figsize=(8, 6))
        bar_width = 0.6
         # Create a legend
        legend_labels = {
        'green': 'Very High House Price',
        'orange': 'Moderate House Price',
        'red': 'Very Low House Price'
        }
        legend_handles = [plt.Rectangle((0,0),1,1, color=color, label=label) for color, label in legend_labels.items()]
        # Customize your plot
        ax.bar(clusterName, results, color=colors, width=bar_width)
        ax.set_title('House price per cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('House price (£)')
        ax.legend(handles=legend_handles)

        # Display the Matplotlib plot in Streamlit
        st.pyplot(fig)


    def clusterPoint(df1, df2, selected_features):
        # Calculate the average price in 2021 for each cluster
        ftr = [feature + "_n" for feature in selected_features]
        cluster_avg_2021 = df2.groupby('Cluster')['2021'].mean()
        # Find the cluster with the highest average price in 2021
        highest_avg_cluster = cluster_avg_2021.idxmax()
        df1 = df1[df1['Cluster'] == highest_avg_cluster]
        df1 = df1[ftr]
        # Calculate the average value for each column in filtered_df
        avg_val = df1.mean()
        # Convert the average values to an array
        clusterCenter = avg_val.values
        X = st.session_state["input_values"][selected_features]
        min_max_vals = get_min_max_val(selected_features)
        # Normalize the row using the min-max values
        for column in selected_features:
            min_val, max_val = min_max_vals[column]
            if X[column][0] > max_val:
                max_val = X[column]

            if X[column][0] < min_val:
                min_val = X[column]

            X[column] = (X[column] - min_val) / (max_val - min_val)

        inputval = X.iloc[0].values
        # Create a dictionary of feature names mapped to percentage difference
        dc = {feature: ((input - center) / center) * 100 for feature, input, center in zip(selected_features, inputval, clusterCenter)}
        st.text("__________________________________________________________________________________________________________________________")
        st.write("<span style='font-size: 24px; font-weight: bold;'>How your area performs when compared to the best areas :</span>", unsafe_allow_html=True)
        st.write('Using areas with high house prices as a base we can making the following comparison : ')
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(clusterCenter)), clusterCenter, label=f'Areas with high house prices', marker='o',color='grey')
        ax.plot(range(len(inputval)), inputval, label=f'Your area', marker='o',color='orange')
        # Configure the plot
        ax.set_xlabel('Features')
        ax.set_ylabel('Index')
        ax.set_title(f'Comparing your area to areas with the highest house prices')
        ax.set_xticks(range(len(selected_features)))
        ax.set_xticklabels(selected_features, rotation=90)
        ax.grid(True)
        ax.legend()
    
        # Display the plot using st.pyplot()
        st.pyplot(fig)
        st.text("__________________________________________________________________________________________________________________________")
        # Create two columns
        st.write('Comparing your area to areas with high house prices')
        col1, col2 = st.columns(2)
        # Iterate through the dictionary and display values in red or green text based on sign
        col1.write('Your area has lesser deprivation indices in : ')
        col2.write('Your area has higher deprivation indices in : ')
        for key, value in dc.items():
            if value < 0:
                col1.write(f'<span style="color: green;">{key}: {round(value, 2)} %</span>', unsafe_allow_html=True)
            else:
                col2.write(f'<span style="color: red;">{key}: +{round(value, 2)} %</span>', unsafe_allow_html=True)

    # Initialize a variable to store the key (default to None if not found)

    def findClusterInClusterInfo(c_info,clustertype):
        # Iterate through the dictionary items and check for the value
        key_found = None
        for key, value in c_info.items():
            if value == clustertype:
                key_found = key
                break  # Exit the loop once a match is found
        return key_found

    def analyze(cls,df):
        # Find the cluster with the highest average price in 2021
        c_info = cluster_information(df)
        cluster_type = cluster_information(df)[cls]
        d = {}
        veryhighHousePriceCluster = findClusterInClusterInfo(c_info,'very high')
        qvh_25 = df[df['Cluster'] == veryhighHousePriceCluster]['HousePrice'].quantile(0.25)
        qvh_75 = df[df['Cluster'] == veryhighHousePriceCluster]['HousePrice'].quantile(0.75)
        d['very high'] = [qvh_25,qvh_75]

        lowHousePriceCluster = findClusterInClusterInfo(c_info,'low')
        ql_25 = df[df['Cluster'] == lowHousePriceCluster]['HousePrice'].quantile(0.25)
        ql_75 = df[df['Cluster'] == lowHousePriceCluster]['HousePrice'].quantile(0.75)
        d['low'] = [ql_25,ql_75]

        highHousePriceCluster = findClusterInClusterInfo(c_info,'high')
        qh_25 = df[df['Cluster'] == highHousePriceCluster]['HousePrice'].quantile(0.25)
        qh_75 = df[df['Cluster'] == highHousePriceCluster]['HousePrice'].quantile(0.75)
        d['high'] = [qh_25,qh_75]

        moderateHousePriceCluster = findClusterInClusterInfo(c_info,'moderate')
        qm_25 = df[df['Cluster'] == moderateHousePriceCluster]['HousePrice'].quantile(0.25)
        qm_75 = df[df['Cluster'] == moderateHousePriceCluster]['HousePrice'].quantile(0.75)
        d['moderate'] = [qm_25,qm_75]
        st.write("Note:")
        for k in d.keys():
            q_25 = d[k][0]
            q_75 = d[k][1]
            pv = st.session_state["Predicted_Price"]
            if cluster_type == 'low':
                if pv > q_75:
                    st.write("* The predicted house price is greater than house prices in the 75th percentile of areas with "+k+" house prices.")
                elif pv > q_25:
                    st.write("* The predicted house price is greater than house prices in the 25th percentile of areas with "+k+" house prices but not more than the 75th percentile.")
                elif pv < q_75 and pv > q_25:
                    st.write("* The predicted house price is between the 25th and 75th percentile of areas with "+k+" house prices.")
            elif (cluster_type == 'high' or cluster_type == 'moderate'):
                if pv < q_25:
                    st.write("* The predicted house price is less than house prices in the 25th percentile of areas with "+k+" house prices.")
                elif pv < q_75:
                    st.write("* The predicted house price is less than house prices in the 75th percentile of areas with "+k+" house prices but not less than the 25th percentile.")
                else:
                    st.write("* The predicted house price is not less than house prices in the 25th and 75th percentile of areas with "+k+" house prices.")
        




    def error_dif(actual_price, predicted_price):
        # Calculate the absolute difference between actual and predicted prices
        absolute_difference = abs(actual_price - predicted_price)
        # Calculate the percentage error as a percentage of the actual price
        percentage_error = (absolute_difference / actual_price) * 100
        # Display the percentage error
        return percentage_error



    def performace():
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        # Assuming you have actual and predicted prices
        actual_price = st.session_state["Actual_Price"]
        predicted_price = st.session_state["Predicted_Price"]
        # Create arrays from the single values
        actual_prices = [actual_price]
        predicted_prices = [predicted_price]
        # Calculate metrics
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        cd = error_dif(actual_price,predicted_price)
       
        # Display the metrics in your Streamlit app
        st.write("<span style='font-size: 24px; font-weight: bold;'>Model Performance Metrics:</span>", unsafe_allow_html=True)
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        if cd < - 15 or cd > 15:
            st.write(f"<span style='color:red'>Error:  ±{cd:.2f} %</span>", unsafe_allow_html=True)
        else:
             st.write(f"<span style='color:green'>Error:  ±{cd:.2f} %</span>", unsafe_allow_html=True)     


    st.text("")
    st.text("__________________________________________________________________________________________________________________________")
    get_model_performance_metrics(st.session_state["model_name"])
    st.text("__________________________________________________________________________________________________________________________")
    st.text("")

    if "Predicted_Price" not in list(st.session_state.keys()):
        st.session_state["lsoa_name"] = ""
        st.session_state["Actual_Price"] = 0.0000
        st.session_state["data"] = [0.0000] * 17
        st.session_state["Predicted_Price"] = 0.0000



    if st.session_state["Predicted_Price"] != 0 and st.session_state["lsoa_name"] != "":
        # for feature in selected_features:
        lsoa_code =st.session_state["lsoa_code"]
        st.text("Option : " + st.session_state["model_name"])
        st.text("Your Area : " + st.session_state["lsoa_name"])
        st.text("Reported house price : £ "+"{:,}".format(st.session_state["Reported_Price"]))
        st.text("Actual house price : £ " + "{:,}".format(st.session_state["Actual_Price"]))
        st.text("Predicted house price : £ " + "{:,}".format(st.session_state["Predicted_Price"]))
        model_name = st.session_state["model_name"]
        rmse = get_rmse(model_name)
        pv = st.session_state["Predicted_Price"]
        st.text("Investment Consideration Range : £ " + "{:,.2f}".format(pv - rmse) + " - £ " + "{:,.2f}".format(pv))
        st.markdown("<span style='color: orange;'>Warning! Reported house price might be suspectible to inflation or human sentimental over-hyped value.</span>", unsafe_allow_html=True)
        st.text("")
        st.text("__________________________________________________________________________________________________________________________")
        st.text("")

        performace()
        st.text("__________________________________________________________________________________________________________________________")
        st.write("")
        try:
            row = df_trend[df_trend['LSOACode'] == lsoa_code].iloc[0][trend_features]
            trend_graph(row,trend_features)
        except Exception as e:
            try:
                pattern = "^(.*?)\s\d+[A-Z]*$"
                cityname = re.match(pattern, st.session_state["lsoa_code"].split("-")[1])(0)
                row = df_trend[df_trend['LSOAname_'] == cityname].iloc[0][trend_features]
                trend_graph(row,trend_features)
            except Exception as e:
                st.write("<font color='red'>Opps! Sorry we are unable to show the 10-year trend of house prices in " + st.session_state["lsoa_name"] + "</font>", unsafe_allow_html=True)



    elif st.session_state["Predicted_Price"] != 0:
        st.text("According to my model the predicted house price in your area is £ " + "{:,}".format(st.session_state["Predicted_Price"]))
    else:
        st.write(f"<span style='font-size: 24px;'>Your predicted value would appear here.</span>", unsafe_allow_html=True)
    





    if st.session_state["Predicted_Price"] != 0 and "Predicted_Price" in list(st.session_state.keys()):
        st.text("")
        st.text("__________________________________________________________________________________________________________________________")
        st.write("<span style='font-size: 24px; font-weight: bold;'>House Price Category or Cluster :</span>", unsafe_allow_html=True)
        st.text("")
        #call the k means function
        min_max_vals = get_min_max_val(selected_features)
        X = st.session_state["input_values"][selected_features]
        #st.write(X.head())
        # Normalize the row using the min-max values
        for column in selected_features:
            min_val, max_val = min_max_vals[column]
            if X[column][0] > max_val:
                max_val = X[column]

            if X[column][0] < min_val:
                min_val = X[column]

            X[column] = (X[column] - min_val) / (max_val - min_val)
        
        # rename columns to match what is expected by kmeans model
        new_column_names = {column: column + "_n" for column in X.columns}
        X = X.rename(columns=new_column_names)
        cluster = clusterModel.predict(X)[0]
        st.write("This area falls under cluster "+str(cluster+1))
        st.write("Houses in areas under this cluster have "+cluster_information(df_main)[cluster]+" prices.")
        st.write("In other to make a good investment, You might want to consider the trend analysis above.")

        #plot the cluster plot
        cluster_graph(df_trend,df_main,cluster)
        clusterPoint(df_main,df_trend,selected_features)
        analyze(cluster,df_main)
        st.text("")
        st.text("__________________________________________________________________________________________________________________________")
        st.text("")
        st.write("<span style='font-size: 24px; font-weight: bold;'>Feature Defintion :</span>", unsafe_allow_html=True)
        st.write("")
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
        input_values = st.session_state["input_values"]
        # Create a DataFrame
        displaydf = pd.DataFrame(features_data)
        # Initialize an empty 'Value' column
        displaydf.insert(1, 'Value', '')

        # Iterate through rows and set 'Value' based on 'FeatureName' from input_values
        for index, row in displaydf.iterrows():
            feature_name = row['FeatureName']
            if feature_name in input_values.columns:
                displaydf.at[index, 'Value'] = input_values.at[0, feature_name]  

        st.write(displaydf)
        st.text("")
        st.text("__________________________________________________________________________________________________________________________")
        st.text("")
        


 

except Exception as e:
    st.write(e)
    st.write(f"<span style='font-size: 24px;'>Opps! something went wrong while displaying the terminal result.</span>", unsafe_allow_html=True)

