# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "python-dotenv",
#     "requests",
#     "scikit-learn",
#     "seaborn",
#     "statsmodels",
# ]
# ///


import logging
import sys
import os
import json
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import base64
import re


# Load environment variables from .env file
def load_env_key():
    try:
        load_dotenv()  # Load environment variables from .env file
        api_key = os.environ["AIPROXY_TOKEN"]
    except KeyError:
        logging.error("Error: AIPROXY_TOKEN is not set in the environment.")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("Error: .env file not found.")
        sys.exit(1)
    logging.info("Environment variable loaded.")
    return api_key

# Returns the fitered filename without . and \\
def name_file(string: str):
    if "\\" in string:
        string = string.split("\\")[-1]  # Get the last part after the backslash
    elif "/" in string:
        string = string.split("/")[-1]  # Handle case with forward slashes (Unix-like paths)
    
    # Remove file extension
    if '.' in string:
        string = string.split(".")[0]  # Remove extension
    
    return string

# Generates a base directory based on the processed name of the input file.
def create_directory():
    # Process the file name
    try:
        processed_name = name_file(sys.argv[1])  
    except IndexError:
        logging.error("No input file path provided. Please provide a file path as a command-line argument.")
        return
    
    # Create the base directory
    try:
        if not os.path.exists(processed_name):
            os.makedirs(processed_name)
            logging.info(f"Directory '{processed_name}' created successfully.")
        else:
            logging.info(f"Directory '{processed_name}' already exists.")
    except Exception as e:
        logging.error(f"An error occurred while creating the directory: {e}")

# Get the dataset file name from the command-line arguments and validate it
def get_dataset():
    if len(sys.argv) != 2:
        logging.error("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    return sys.argv[1]

# Load the dataset
def load_dataset(dataset_filename):
    try:
        df = pd.read_csv(dataset_filename)
        logging.info(f"Dataset loaded successfully from {dataset_filename}.")
        return df
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_filename, encoding='ISO-8859-1')
        logging.info(f"Dataset loaded successfully from {dataset_filename}.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset from {dataset_filename}: {e}")
        sys.exit(1)

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Data cleaning
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input DataFrame by dropping columns with all NaN values
    and removing duplicate rows.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """   
    # Drop columns that are all NaN
    df_cleaned = df.dropna(axis=1, how='all')
    logging.info(f"Dropped columns with all NaN values. Remaining columns: {df_cleaned.columns.tolist()}")
    
    # Remove duplicate rows
    duplicate_rows = df_cleaned.duplicated().sum()
    df_cleaned = df_cleaned.drop_duplicates()
    logging.info(f"Removed {duplicate_rows} duplicate rows.")
    
    return df_cleaned


# Sends a prompt to the GPT model and returns the model's response.   
def chat(prompt, api_key, model='gpt-4o-mini'):
    """
    Sends a user prompt to the OpenAI API and returns the model's response.

    Args:
        prompt (str): The message to be sent to the model.
        api_key (str): The API key for authentication.
        model (str, optional): The model to be used (default is 'gpt-4o-mini').

    Returns:
        str: The model's response or None if an error occurs.
        """
    
    url = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': 'You are a data science expert. Provide clear, concise, and actionable answers.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        'temperature': 0.7,
        'max_tokens': 100
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        output = response.json()

        if output.get('error', None):
            logging.error(f"LLM Error during chat with prompt '{prompt[:30]}...'. Error: {output}")
            return None

        monthly_cost = output.get('monthlyCost', 'N/A')
        logging.info(f"Chat completion successful for prompt '{prompt[:30]}...'. Monthly Cost: {monthly_cost}")
        return output['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for chat. Error: {e}")
        return None

# Analysis

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Generic analysis 
def generic_analysis(df):
    """
    Perform basic analysis on a dataset, including summary statistics, missing values,
    column data types, and basic data shape.
    
    Args:
    - df (pd.DataFrame): The dataset to analyze.
    
    Returns:
    - dict: A dictionary containing the analysis results.
    """
    logging.info("Starting generic analysis...")

    analysis = {}
    try:
        # Basic checks: Dataframe not empty
        if df.empty:
            raise ValueError("The DataFrame is empty")
        
        logging.info("DataFrame received for analysis")

        # Cleaning data
        df_cleaned = clean_data(df)
        if df_cleaned.empty:
            raise ValueError("The DataFrame is empty") 

        # Summary statistics for numeric columns
        numeric_columns = df_cleaned.select_dtypes(include='number')
        if numeric_columns.empty:
            logging.warning("No numeric columns found in the DataFrame.")
            analysis['Summary Stats'] = "No numeric columns"
        else:
            analysis['Summary Stats'] = numeric_columns.describe().loc[['count', 'mean', 'std', 'min', '50%', 'max']].transpose().round(3).to_dict()
            logging.info("Summary statistics computed for numeric columns")

        # Missing values
        analysis['Missing Values'] = df_cleaned.isnull().sum().to_dict()
        logging.info("Missing values computed")

        # Column data types
        analysis['Column Data Types'] = df_cleaned.dtypes.apply(str).to_dict()
        logging.info("Column data types retrieved")

        # Additional info (e.g., data size)
        analysis['DataFrame Shape'] = {
            'Rows': df_cleaned.shape[0],
            'Columns': df_cleaned.shape[1]
        }
        logging.info(f"DataFrame shape: {df_cleaned.shape}")

    except Exception as e:
        logging.error(f"An error occurred during analysis: {e}\n{traceback.format_exc()}")
        analysis['Error'] = str(e)

    return analysis

# Generate outlier plot
def outlier_plot(name,df):
    df_cleaned = clean_data(df)
    if df_cleaned.empty:
        logging.error("The Dataframe is empty after cleaning.")
        return None
    
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        plt.figure(figsize=(10, 8),dpi=100)
        sns.boxplot(data=df_cleaned[numeric_cols])
        chart_file_name = f"{name}/outlier_plot.png"
        plt.savefig(chart_file_name, dpi=100)
        plt.close()
        logging.info(f"Outlier plot saved as {chart_file_name}")
        return chart_file_name
    else:
        logging.error("No numeric columns available for outlier plot.")
        return None

# Generate a correlation matrix 
def correlation_matrix(name, df):
    df_cleaned = clean_data(df)
    if df_cleaned.empty:
        logging.error("The Dataframe is empty after cleaning.")
        return None
    numeric_data = df_cleaned.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        plt.figure(figsize=(10, 8),dpi=100)
        sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
        plt.title('Correlation Matrix')
        chart_file_name = f"{name}/correlation_matrix.png"
        plt.savefig(chart_file_name, dpi = 100)
        plt.close()
        logging.info(f"Correlation matrix saved as {chart_file_name}")
        return chart_file_name
    else:
        logging.error("No numeric columns available for correlation matrix.")
        return None


# Non-generic analysis
def perform_ml_analysis(name, df, api_key):
    """
    Perform in-depth analysis by consulting a language model (LLM) to suggest 
    appropriate machine learning techniques based on the provided dataset.
    """

    # Extract column names and a sample of the dataset to pass to LLM   
    column_info = df.columns.tolist()
    example_data = df.head(1).to_dict(orient="records")
    
    # Define a list of machine learning techniques
    ml_techniques = ['regression' , 'clustering', 'classification', 'time series']
    
    # Generate a detailed prompt to ask the LLM for appropriate techniques based on the dataset
    prompt = f"""
    Dataset columns: {column_info}. Sample: {example_data}.
    Suggest most suitable machine learning technique (only 1) strictly from this list: {ml_techniques} 
    Output format: {{"techniques": ["technique"]}}
    If none, return an empty list.
    """
    
    # Get technique suggestions from LLM
    try:
        analysis_suggestion = chat(prompt, api_key)
        if not analysis_suggestion:
            logging.error("LLM returned no analysis suggestion.")
            return None
    except Exception as e:
        logging.error(f"Error while querying LLM: {e}")
        return None
    
    # Parse the LLM response
    try:
        analysis_suggestion = analysis_suggestion.strip()       
        if analysis_suggestion.startswith('{"techniques":'):
            response_data = json.loads(analysis_suggestion)
            
            # Extract the list of techniques
            techniques = response_data.get("techniques", [])
            
            # Ensure the techniques are in the proper format (a list)
            if isinstance(techniques, list):
                if techniques:  # If the list is not empty
                    logging.info(f"LLM suggested techniques: {techniques}")
                else:
                    logging.info("LLM suggested no techniques.")
                    return None
            else:
                logging.error("Expected 'techniques' to be a list.")
                return None
        else:
            logging.error(f"Unexpected format in LLM response: {analysis_suggestion}")
            return None    
        
        # Execute the suggested techniques.
        ml_results = {}

        for technique in techniques:
            technique = technique.strip().lower()  
            if technique == 'regression':
                logging.info("Performing Regression.")
                ml_results['Regression'] = regression(df, api_key)
            elif technique == 'clustering':
                logging.info("Performing Clustering.")
                ml_results['Clustering'] = kmeans_clustering(df, name, n_clusters=3)
            elif technique == 'classification':
                logging.info("Performing Classification.")
                ml_results['Classification'] = classification(name, df, api_key)
            elif technique == 'time series':     
                logging.info("Performing Time Series.")          
                ml_results['Time Series'] = time_series(name, df, api_key)
        
        return ml_results
    except Exception as e:
        logging.error(f"Error parsing the LLM response: {e}")
        return None

# Perform Regression based on LLM suggestion
def regression(df, api_key):
    """
    Performs linear regression on the given dataset based on column suggestions from a language model.
    The function cleans the dataset, obtains feature and target column suggestions from the LLM,
    validates the columns, performs regression, and returns evaluation metrics (MSE, MAE, R-squared).
    
    Args:
        df (pd.DataFrame): The input dataset for regression.
        api_key (str): The API key to interact with the LLM.
    
    Returns:
        dict: A dictionary with regression metrics ("mse", "mae", "r2") or None if an error occurs.
    """
    # Data cleaning
    df_cleaned = clean_data(df)
    if df_cleaned.empty or df_cleaned.shape[1] == 0:
        logging.error("The dataset is empty or has no usable columns.")
        return None
    
    # Extract column names and a sample of the dataset to pass to LLM   
    column_info = df_cleaned.columns.tolist()
    example_data = df_cleaned.head(1).to_dict(orient="records")

    # Send a prompt to the LLM to suggest appropriate columns for regression
    prompt = f"""\
        Dataset columns: {column_info}. Sample row: {example_data}.
        Task: Identify the relevant feature columns (only 2) and the target column for a regression task. Exclude the target column from the feature set. Return the names of the selected columns.
        Output format:
        {{"features": ["feature1", "feature2"],"target": "target_column"}}
        """
    
    # Get column selection suggestion from LLM
    column_suggestion = chat(prompt, api_key)
    
    if not column_suggestion:
        logging.error("Failed to get column suggestion for regression from LLM.")
        return None

    logging.info(f"LLM suggested columns for regression: {column_suggestion}")

    try:
        # Parse the suggested columns from the LLM response 
        parsed_response = json.loads(column_suggestion.strip())
        print(parsed_response)
        # Extract features and target columns
        selected_features = parsed_response.get("features", [])
        target_column = parsed_response.get("target", None) 

        if not selected_features or not target_column:
            logging.error("LLM didn't provide valid columns for regression.")
            return None
        
        # Ensure the target column and feature columns exist in the dataframe
        missing_columns = [col for col in [target_column] + selected_features if col not in df_cleaned.columns]
        if missing_columns:
            logging.error(f"Missing columns in dataset: {missing_columns}")
            return None
        
        logging.info(f"Selected features: {selected_features}")
        logging.info(f"Target column: {target_column}")

        df_cleaned = df_cleaned.dropna(subset=selected_features + [target_column])
        # Perform the regression using the selected columns
        X = df_cleaned[selected_features].values
        y = np.ravel(df_cleaned[target_column].values)  # Ensure y is 1D
        
        # Ensure we have enough rows for regression
        if X.shape[0] < 5 or y.shape[0] < 5:
            logging.warning("Not enough data to perform regression.")
            return None

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Perform linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"Mean Squared Error: {mse}")
        logging.info(f"Mean Absolute Error: {mae}")
        logging.info(f"R-squared: {r2}")

        return {
        "mse": mse,
        "mae": mae,
        "r2": r2
    }

    except Exception as e:
        logging.error(f"Error performing regression: {e}")
        return None

# Perform classification on LLM suggestion
def classification(name, df, api_key):
    """
    Performs classification on the provided dataframe using GPT model to suggest features and target columns. 
    Trains a RandomForestClassifier, evaluates its performance, and visualizes the confusion matrix.

    Args:
        name (str): The name or identifier used to save the output chart.
        df (pd.DataFrame): The input dataset containing numerical and categorical data.
        api_key (str): The API key for accessing the GPT model to suggest relevant columns.

    Returns:
        dict: A dictionary containing the accuracy, classification report, and confusion matrix chart filename.
              Returns None if an error occurs.
    """
    if df.empty:
        logging.error("The provided dataframe is empty.")
        return None

    # Clean data 
    df_cleaned = clean_data(df)  
    if df_cleaned.empty:
        logging.error("The Dataframe is empty after cleaning.")
        return None
    
     # Extract column names and a sample of the dataset to pass to LLM   
    column_info = df_cleaned.columns.tolist()
    example_data = df_cleaned.head(1).to_dict(orient="records")

    # Send a prompt to the LLM to suggest appropriate columns for classification
    prompt = f"""\
        Dataset columns: {column_info}. Sample row: {example_data}.
        Task: Identify the relevant feature columns (limit 4) and the target column for a classification task (confusion matrix).Exclude the target column from the feature set. Return the names of the selected columns.
        Output format:
        {{"features": ["feature1", "feature2",...],"target": "target_column"}}
        """
    
    # Get column selection suggestion from LLM
    column_suggestion = chat(prompt, api_key)
    
    if not column_suggestion:
        logging.error("Failed to get column suggestion for classification from LLM.")
        return None

     # Parse the LLM output 
    try:
        # Parse the suggested columns from the LLM response 
        column_suggestion = json.loads(column_suggestion.strip())

        # Extract features and target columns
        features = column_suggestion.get('features', [])
        target = column_suggestion.get('target', None)

        # Validate the suggestion
        if not features or not target:
            logging.error("LLM did not provide valid feature or target columns.")
            return None

        logging.info(f"LLM suggested columns for classification: Features: {features}, Target: {target}")

        # Ensure the target column and feature columns exist in the dataframe
        missing_columns = [col for col in [target] + features if col not in df_cleaned.columns]
        if missing_columns:
            logging.error(f"Missing columns in dataset: {missing_columns}")
            return None

    except Exception as e:
        logging.error(f"Error while parsing LLM suggestion: {e}")
        return None
    try:
        df_cleaned = df_cleaned.dropna(subset=features + [target])
        # Select features and target from the dataframe
        X = df_cleaned[features]
        y = df_cleaned[target]

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

        # Ensure that there are categorical and numerical columns identified
        if not categorical_cols or not numerical_cols:
            logging.error("No categorical or numerical columns were identified.")
            return None

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocessing for numerical and categorical features
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine both transformations into a single column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Create a pipeline with preprocessor and RandomForestClassifier
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Get unique class labels for multi-class
        labels = sorted(y_test.unique())
        # Visualize the confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        # Save the plot
        chart_filename = f"{name}/confusion_matrix.png"
        plt.savefig(chart_filename, dpi =100)
        plt.close()

        # Log the accuracy and classification report
        logging.info(f"Accuracy: {accuracy:.4f}")
        #logging.info("Confusion Matrix:\n")
        #logging.info(f"{cm}")
        logging.info("Classification Report:\n")
        logging.info(report)

        return {
        "accuracy": accuracy,
        "classification_report": report
        }
    except Exception as e:
        logging.error(f"Error performing classification: {e}")
        return None

# Perform Clustering  with KMeans
def kmeans_clustering(df, name, n_clusters=3):
    """
    Perform KMeans clustering, generate a clustering graph, and return useful values.
    
    Args:
        df (pd.DataFrame): The dataset containing numeric features for clustering.
        name (str): The name to use for saving the clustering graph.
        n_clusters (int): The number of clusters for KMeans (default is 3).
    
    Returns:
        dict: Contains 'Cluster centers', 'Labels', 'Inertia', and saves the clustering graph.
    """
    try:
        if df.empty:
            logging.error("The provided dataframe is empty.")
            return None

        # Clean data 
        df_cleaned = clean_data(df)  
        if df_cleaned.empty:
            logging.error("The DataFrame is empty after cleaning.")
            return None
        
        # Select numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            logging.warning("Not enough numeric columns for clustering.")
            return None

        # Handle missing values using SimpleImputer (mean imputation for simplicity)
        imputer = SimpleImputer(strategy='mean')
        df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])

        # Extract features for clustering
        X = df_cleaned[numeric_cols].values

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply KMeans clustering and predict cluster labels in one step
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_cleaned['Cluster'] = kmeans.fit_predict(X_scaled)

        # Perform PCA if more than 2 dimensions (for visualization purposes)
        if X_scaled.shape[1] > 2:
            pca = PCA(n_components=2)
            X_scaled = pca.fit_transform(X_scaled)

        # Plot the clusters
        plt.figure(figsize=(8, 6), dpi =100)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df_cleaned['Cluster'], cmap='viridis', marker='o', s=50)

        # Plot cluster centers
        if X_scaled.shape[1] == 2:
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                        c='red', s=200, marker='x', label='Cluster Centers')
        else:
            # Project cluster centers to 2D if PCA was applied
            cluster_centers_2d = pca.transform(kmeans.cluster_centers_)
            plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1],
                        c='red', s=200, marker='x', label='Cluster Centers')

        # Add labels and title
        plt.title(f"KMeans Clustering with {n_clusters} clusters")
        plt.xlabel("Principal Component 1" if X_scaled.shape[1] == 2 else "Feature 1")
        plt.ylabel("Principal Component 2" if X_scaled.shape[1] == 2 else "Feature 2")
        plt.legend(loc='best')

        # Generate a file name and save the plot
        chart_file_name = f"{name}/clustering_graph.png"
        plt.savefig(chart_file_name, dpi =100)
        plt.close()  

        logging.info(f"Clustering chart saved as {chart_file_name}")

        # Return useful values: cluster centers, labels, and inertia
        return {
            "Cluster centres": kmeans.cluster_centers_,
            "Labels": df_cleaned['Cluster'].values,
            "Inertia": kmeans.inertia_
        }
    
    except Exception as e:
        logging.error(f"Error during clustering or generating the clustering graph: {e}")
        return None

# Perform Time Series Analysis
def time_series(name, df, api_key):
    """
    Consults LLM, Analyzes the time series data, performing ADF test and decomposition.

    Parameters:
    - name (str): Identifier for the time series.
    - df (pandas.DataFrame): DataFrame with time series data.
    - api_key (str): API key for external services.

    Returns:
    - dict: Contains ADF test results (statistic, p-value, critical values) and 
            components (trend, seasonal, residual).
    """ 
    
    if df.empty:
        logging.error("The provided dataframe is empty.")
        return None
    # Clean data 
    df_cleaned = clean_data(df)  
    if df_cleaned.empty:
        logging.error("The DataFrame is empty after cleaning.")
        return None
    
     # Extract column names and a sample of the dataset to pass to LLM   
    column_info = df_cleaned.columns.tolist()
    example_data = df_cleaned.head(1).to_dict(orient="records")

    # Send a prompt to the LLM to suggest appropriate columns for classification
    prompt = f"""\
        Dataset columns: {column_info}. Sample row: {example_data}.
        Task: Identify the column representing dates/times (Time Column) and the column with numeric values (Value Column) for a time series task.Ensure that the Value Column is not included in the Time Column.Return the names of the selected columns.
        Output format:
        {{"time": "time_column","value": "value_column"}}
        """
    
    # Get column selection suggestion from LLM
    column_suggestion = chat(prompt, api_key)
    
    if not column_suggestion:
        logging.error("Failed to get column suggestion for classification from LLM.")
        return None

    # Parse the LLM output 
    try:
        # Parse the suggested columns from the LLM response 
        column_suggestion = json.loads(column_suggestion.strip())

        # Extract time and value columns
        time_column = column_suggestion.get("time")
        value_column = column_suggestion.get("value")

        # Validate the suggestion
        if not time_column or not value_column:
            logging.error("Invalid suggestion from LLM: Missing 'time' or 'value' columns.")
            return None
        
        # Ensure the time column and value column exist in the dataframe
        if time_column not in df_cleaned.columns or value_column not in df_cleaned.columns:
            logging.error(f"The suggested columns '{time_column}' or '{value_column}' do not exist in the dataframe.")
            return None
        
        # Ensure the time column is in datetime format
        df_cleaned[time_column] = pd.to_datetime(df_cleaned[time_column], errors='coerce')

        # Drop rows where the time column is invalid or value column is NaN
        df_cleaned = df_cleaned.dropna(subset=[time_column, value_column])

    except Exception as e:
        logging.error(f"Error while parsing LLM suggestion: {e}")
        return None
    try:
        
        # Perform the Augmented Dickey-Fuller (ADF) test
        adf_result = adfuller(df_cleaned[value_column])
        adf_statistic, p_value, _, _, critical_values, _ = adf_result

        # Decompose the time series
        decomposition = seasonal_decompose(df_cleaned[value_column], model='multiplicative', period=12)
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        residual = decomposition.resid.dropna()

        # Generate the time series plot
        plt.figure(figsize=(10, 6), dpi =100)
        plt.plot(df_cleaned[time_column], df_cleaned[value_column], marker='o', linestyle='-', color='b')

        # Customize plot labels and title
        plt.xlabel("Time")
        plt.ylabel(value_column)
        plt.title(f"Time Series Plot: {name}")
        
        # Save the plot
        chart_filename = f"{name}/time_series.png"
        plt.savefig(chart_filename, dpi =100)
        plt.close()

        logging.info(f"Time series plot successfully saved as {chart_filename}")
        return {
            "adf_statistic": adf_statistic,
            "adf_p_value": p_value,
            "adf_critical_values": critical_values,
            "trend_component": trend.tolist(),  
            "seasonal_component": seasonal.tolist(),
            "residual_component": residual.tolist()
        }

    except Exception as e:
        logging.error(f"Error generating Time series graph: {e}")
        return None

# Narrate a story
def create_story_and_markdown(name, description, api_key, model='gpt-4o-mini'):
    """
    Generates a compelling data analysis story in markdown format and saves it as README.md.
    """
    proxy_url = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        'model': model,
        "messages": [
            {
                'role': 'system', 
                'content': "You are a data analyst and storyteller. Present your findings as a creative and compelling narrative."
            },
            {
                'role': 'user', 
                'content': f"Data: {description}. Craft an engaging, story-driven analysis in markdown (README.md format), with clear sections. Highlight key insights, trends, and actionable recommendations. Make it engaging with plot, characters, and dialogues to bring the data to life."
            }
        ],
        'temperature': 0.7,
        'max_tokens': 1500
    }
    try:
        response = requests.post(url=proxy_url, headers=headers, json=payload)
        if response.ok:
            ai_response = response.json()
            result = ai_response["choices"][0]["message"]["content"].strip()

            # Ensure directory exists
            if not os.path.exists(name):
                logging.info(f"Directory {name} does not creating it.")
                os.makedirs(name)
            
            # Write the result to a README.md file
            try:
                with open(f"{name}/README.md", "w", encoding="utf-8") as f:
                    f.write(result)
                logging.info(f"Successfully written to {name}/README.md")
            except Exception as e:
                logging.error(f"Error writing README.md: {e}")
        else:
            logging.error(f"Error fetching the summary. Status code: {response.status_code}")
            logging.error(f"Response content: {response.content}")
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")


# Main function
def main():
    # Set up logging
    logging.basicConfig(
        filename="autolysis.log",   # Log file name
        level=logging.INFO,         # Capture INFO level and above
        format="%(asctime)s - %(levelname)s - %(message)s"  # Log message format
        )
    
    api_key = load_env_key()

    dataset_filename = get_dataset()
    df = load_dataset(dataset_filename)
    name = name_file(dataset_filename)
    create_directory()

    analysis = generic_analysis(df)

    outlier_plot(name,df)
    correlation_matrix(name,df)
    ml_results = perform_ml_analysis(name, df, api_key)

    list_chart = []
    for filename in os.listdir(name):
        if filename.endswith('.png'):
            list_chart.append(os.path.join(name, filename))
    image_data = [encode_image(image) for image in list_chart]
    description = {"general analysis": analysis,"in-depth analysis": ml_results, "charts": image_data}
    
    # print(description)
    create_story_and_markdown(name, description, api_key, model='gpt-4o-mini')
    
    readme_path = os.path.join(name, 'README.md')
    with open(readme_path, 'a') as readme_file:
        for chart in list_chart:
            # Extract the base name of the chart (e.g., 'confusion_matrix.png')
            chart_name = os.path.basename(chart).split('.')[0]
            
            # Add heading (e.g., Confusion Matrix)
            readme_file.write(f"\n\n## {chart_name.replace('_', ' ').title()}\n")
            
            # Add image (markdown format: ![alt text](image_path))
            relative_image_path = os.path.relpath(chart, name)
            readme_file.write(f"![{chart_name}]({relative_image_path})\n")

    logging.info("Autolysis completed.")


if __name__ == "__main__":
    main()
