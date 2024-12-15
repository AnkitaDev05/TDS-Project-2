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
import re

from dateutil import parser
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

# Writing to file
def write_file(name, text_content,title=None):
    """Write text content to a file, optionally adding a title."""

    logging.info(f'Writing README.md to directory: {name}')

    # Ensure directory exists
    if not os.path.exists(name):
        logging.info(f"Directory {name} does not exist, creating it.")
        os.makedirs(name, exist_ok=True)

    try:
        with open(os.path.join(name, "README.md"), "w", encoding="utf-8") as f:
            # If a title is provided, add it to the top of the file
            if title:
                f.write("# " + title + "\n\n")

            # If content starts with a markdown block, strip it out properly    
            if text_content.startswith("```markdown"):
                text_content = text_content[11:].strip()  # Remove the ```markdown part
                if text_content.endswith("```"):
                    text_content = text_content[:-3].strip()  # Remove the closing ```
            
            # Write the content to the file
            f.write(text_content + "\n\n")
        logging.info(f"Successfully written to {name}/README.md")

    except Exception as e:
        logging.error(f"Error writing README.md: {e}")


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


# AI Proxy 'Function Call' Functions
def chat_function_call(prompt, api_key, function_descriptions, model='gpt-4o-mini'):
    """Call an AI API for a function completion based on user input."""
    
    url = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        'model': model,
        'messages': [
            {'role': 'system','content': "You are a data science expert with expertise in machine learning, statistics, and data analysis. Provide clear, concise, and actionable answers."},
            {'role': 'user','content': prompt}
        ],
        'functions': function_descriptions,
        'function_call': 'auto',
        'max_tokens': 200
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        output = response.json()

        if output.get('error'):
            logging.error(f"LLM Error: {output}")
            return None
        
        logging.info(f"Monthly Cost: {output.get('monthlyCost', None)}")
        
        return {
            'arguments': output['choices'][0]['message']['function_call']['arguments'],
            'name': output['choices'][0]['message']['function_call']['name']
        }
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        logging.error(f"Unexpected response format: {e}")
        return None


filter_function_descriptions = [
    {
        'name': 'filter_features',
        'description': 'Generic function to extract data from a dataset.',
        "parameters": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of column names to keep. Eg. ['language', 'quality']",
                },
            },
            "required": ["features"]
        }
    },
    {
        'name': 'extract_features_and_target',
        'description': 'Use this to extract feature matrix (X) and target vector (y) for model training tasks (regression, classification)',
        "parameters": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of feature columns, e.g., ['n_rooms', 'locality', 'latitude']",
                },
                "target": {
                    "type": "string",
                    "description": "Name of the target column, e.g., 'price'",
                },
            },
            "required": ["features", "target"]
        }
    },
    {
        'name': 'extract_time_series_data',
        'description': "Extract date/time column and numerical column for time series analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_column": {
                    "type": "string",
                    "description": "Name of the date column.",
                },
                "numerical_column": {
                    "type": "string",
                    "description": "Name of the numerical column, e.g., 'price'",
                }
            },
            "required": ["date_column", "numerical_column"]
        }
    },
    ]

def filter_features(data, features):
    """Filters the specified features from the input DataFrame and returns a copy of the filtered data."""
    return data[features].copy()


def extract_features_and_target(data, features, target):
    """Extracts the specified features and target column from the input DataFrame."""
    return data[features].copy(), data[target].copy()


def extract_time_series_data(data, date_column, numerical_column):
    """Extracts the date and numerical columns from the input DataFrame for time series analysis."""
    if date_column not in data.columns or numerical_column not in data.columns:
        logging.error(f"ERROR Columns '{date_column}' or '{numerical_column}' not found in the data.")
    return data[date_column].copy(), data[numerical_column].copy()



# Analysis

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

        # First 3 rows
        analysis['first_3'] = df_cleaned.head(3).to_dict()
        logging.info("Extracted first 3 rows.")

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
        sns.set_theme(style="whitegrid")  # Adds a clean background with gridlines
        num_cols = len(df_cleaned[numeric_cols].columns)
        palette = sns.color_palette("bright", num_cols)
        sns.boxplot(data=df_cleaned[numeric_cols], palette=palette)
        plt.xticks(rotation=45, ha='right')
        plt.title('Outlier Plot', fontsize=18, fontweight='bold')
        plt.xlabel('Features', fontsize=16, fontweight='bold')
        plt.ylabel('Values', fontsize=16, fontweight='bold')
        chart_file_name = os.path.join(name, "outlier_plot.png")
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
        plt.title('Correlation Matrix', fontsize=18, fontweight='bold')
        plt.tight_layout()
        chart_file_name = os.path.join(name, "correlation_matrix.png")
        plt.savefig(chart_file_name, dpi = 100)
        plt.close()
        logging.info(f"Correlation matrix saved as {chart_file_name}")
        return chart_file_name
    else:
        logging.error("No numeric columns available for correlation matrix.")
        return None


# Non-generic analysis

# Perform Regression based on LLM suggestion
def regression(name, df, api_key):
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
    column_info = "\n".join([f"{col}: {dtype}" for col, dtype in df_cleaned.dtypes.items()])
    example_data = df_cleaned.head(1).to_dict(orient="records")

    # Send a prompt to the LLM to suggest appropriate columns for regression
    prompt = f"""\
        Dataset columns: {column_info}. Sample row: {example_data}.
        Task: Identify the relevant feature columns and the target column for a regression task. Exclude the target column from the feature set.
        """
    
    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=filter_function_descriptions)
    
    if not response:
        logging.error("Failed to get column suggestion for regression from LLM.")
        return None
 
    try:
        try:
            params = json.loads(response['arguments'])
            chosen_func = eval(response['name'])  
        except (json.JSONDecodeError, NameError, SyntaxError) as e:
            logging.error(f"Error processing response: {e}")
            return None 

        if 'target' not in params.keys():
            logging.error("Target variable not found in the parameters.")
            return None
        
        params['features'] = list(filter(lambda feature: feature != params['target'], params['features']))

        X, y = chosen_func(data=df_cleaned, **params)
        X = X.select_dtypes(include=['number'])

        if X.empty:
            logging.warning("No numeric columns found in the dataset.")
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a regression pipeline with additional model options (e.g., Ridge, Lasso)
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('regression', LinearRegression())  # Can be changed to Ridge or Lasso for regularization
        ])

        logging.info('Training Linear Regression Model')

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2_score = pipe.score(X_test, y_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        regression_plot(name=name, y_true=y_test, y_pred=y_pred)  
        logging.info(f"""
                     'r2_score': {r2_score},'mae': {mae}, 
                     'mse': {mse},'rmse': {rmse},
                     'coefficient': {pipe['regression'].coef_},
                     'intercept': {pipe['regression'].intercept_},
                     'feature_names_input': {list(X.columns)},
                     'target_name': {y.name},
                     """)
        return {
            'r2_score': r2_score,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'coefficient': pipe['regression'].coef_,
            'intercept': pipe['regression'].intercept_,
            'feature_names_input': list(X.columns),
            'target_name': y.name,
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
        dict: A dictionary containing the accuracy, classification report.
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
    column_info = "\n".join([f"{col}: {dtype}" for col, dtype in df_cleaned.dtypes.items()])
    example_data = df_cleaned.head(1).to_dict(orient="records")

    # Send a prompt to the LLM to suggest appropriate columns for classification
    prompt = f"""\
        Dataset columns: {column_info}. Sample row: {example_data}.
        Task: Identify the relevant feature columns and the target column for a classification task. Exclude the target column from the feature set. Target column should be categorical datatype. Hint: Use function extract_features_and_target.
        """
    
    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=filter_function_descriptions)
    logging.info(f"AI Response: {response}")
    if not response:
        logging.error("Failed to get column suggestion for classification from LLM.")
        return None   
     # Parse the LLM output 
    try:
        params = json.loads(response['arguments'])
        logging.info(f"Parsed Params: {params}")  
    except (json.JSONDecodeError, NameError, SyntaxError) as e:
        logging.error(f"Error processing response: {e}")
        return None 

    if not any('target' == key.strip().lower() for key in params.keys()):
        logging.error("Target variable not found in the parameters.")
        logging.error(params)
        return None
    
    target = params['target']
    features = list(filter(lambda feature: feature != target, params['features']))

    if target not in df_cleaned.columns:
        logging.error(f"The target column '{target}' does not exist in the dataframe.")
        return None
    
    missing_features = [feature for feature in features if feature not in df_cleaned.columns]
    if missing_features:
        logging.error(f"The following feature columns are missing from the dataframe: {missing_features}")
        return None


    try:
        X = df_cleaned[features].select_dtypes(include=['number'])
        y = df_cleaned[target]


        if y.nunique() > 20:
            logging.warning('Target variable has too many unique values for classification.')
            return None

        if X.empty:
            logging.warning("No numeric columns found for features.")
            return None
        
        # Drop rows with missing values in features or target
        df_cleaned = df_cleaned.dropna(subset=features + [target])
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
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])
        
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
        report = classification_report(y_test, y_pred)
        
        # Get unique class labels for multi-class
        labels = sorted(y_test.unique())

        # Ensure the directory exists before saving the plot
        if not os.path.exists(name):
            os.makedirs(name)

        confusion_matrix_plot(y_true=y_test, y_pred=y_pred, labels=labels, name=name)

        # Log the accuracy and classification report
        logging.info(f"Accuracy: {accuracy:.4f}")
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
def kmeans_clustering(name, df, api_key, n_clusters=3):
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
        logging.info("Imputation of missing values applied using mean strategy.")

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
            X_scaled_pca = pca.fit_transform(X_scaled)
        else:
            X_scaled_pca = X_scaled

        sns.set_theme(style="whitegrid")

        # Plot the clusters
        plt.figure(figsize=(10, 8), dpi =100)
        scatter = plt.scatter(X_scaled_pca[:, 0], X_scaled_pca[:, 1], c=df_cleaned['Cluster'], cmap='viridis', marker='o', s=50, edgecolors='k', alpha=0.7, label='Data Points')

        # Plot cluster centers
        if X_scaled_pca.shape[1] == 2:
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                        c='red', s=200, marker='x', label='Cluster Centers', linewidths=2)
        else:
            # Project cluster centers to 2D if PCA was applied
            cluster_centers_2d = pca.transform(kmeans.cluster_centers_)
            plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1],
                        c='red', s=200, marker='x', label='Cluster Centers', linewidths=2)

        # Add labels and title
        plt.title(f"KMeans Clustering with {n_clusters} clusters", fontsize=16, fontweight='bold')
        plt.xlabel("Principal Component 1" if X_scaled_pca.shape[1] == 2 else "Feature 1", fontsize=14, fontweight='bold')
        plt.ylabel("Principal Component 2" if X_scaled_pca.shape[1] == 2 else "Feature 2", fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        # Generate a file name and save the plot
        chart_file_name = os.path.join(name, "clustering_graph.png")
        plt.savefig(chart_file_name, dpi =100)
        plt.close()  

        logging.info(f"Clustering chart saved as {chart_file_name}")

        # Return useful values: cluster centers, labels, and inertia
        return {
            "Cluster centres": kmeans.cluster_centers_,
            "Labels": df_cleaned['Cluster'].values,
            "Inertia": kmeans.inertia_,
            "PCA components": pca.components_ if X_scaled.shape[1] > 2 else None
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
    column_info = "\n".join([f"{col}: {dtype}" for col, dtype in df_cleaned.dtypes.items()])
    example_data = df_cleaned.head(1).to_dict(orient="records")

    # Send a prompt to the LLM to suggest appropriate columns for classification
    prompt = f"""\
        Dataset columns: {column_info}. Sample row: {example_data}.
        Task: Identify the column representing dates/times (date Column) and numerical column for a time series task. Ensure that the numerical Column is not included in the date Column. Use function extract_time_series_data.
        """
    
    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=filter_function_descriptions)
    
    if not response:
        logging.error("Failed to get column suggestion for time series from LLM.")
        return None
    logging.debug(f"LLM Response: {response}")
    # Parse the LLM output 
    try:
        params = json.loads(response['arguments'])
        logging.info(f"Parsed parameters: {params}") 
    except (json.JSONDecodeError, NameError, SyntaxError, KeyError) as e:
        logging.error(f"Error processing response: {e}")
        return None 
    
    # Parse the LLM output 
    try:
        logging.info('Starting Time series analysis...')
        date_col = params.get('date_column', None)
        num_col = params.get('numerical_column', None)
 
        if isinstance(date_col, list):
            date_col = date_col[0]
        if isinstance(num_col, list):
            num_col = num_col[0]

        # Validate the suggestion
        if not date_col or not num_col:
            logging.error("Invalid suggestion from LLM: Missing 'time' or 'value' columns.")
            return None
        
        # Ensure the time column and value column exist in the dataframe
        if date_col not in df_cleaned.columns or num_col not in df_cleaned.columns:
            logging.error(f"The suggested columns '{date_col}' or '{num_col}' do not exist in the dataframe.")
            return None
        logging.debug(f"Extracted date column: {date_col}")
        logging.debug(f"Extracted numerical column: {num_col}")
        
        # Ensure the time column is in datetime format
        df_cleaned[date_col] = df_cleaned[date_col].apply(lambda x: parser.parse(x, fuzzy=True) if isinstance(x, str) else pd.NaT)
        data = df_cleaned.set_index(date_col).sort_index()
        
        # Drop rows where the time column is invalid or value column is NaN
        data = data.dropna(subset=[num_col])

    except Exception as e:
        logging.error(f"Error while parsing LLM suggestion or extracting columns: {e}")
        return None
    try:
        
        # Perform the Augmented Dickey-Fuller (ADF) test
        adf_result = adfuller(data[num_col])
        adf_statistic, p_value, _, _, critical_values, _ = adf_result

        # Decompose the time series
        decomposition = seasonal_decompose(data[num_col], model='multiplicative', period=12)
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        residual = decomposition.resid.dropna()

        # Call time_series_graph to generate and save the graph
        chart_filename = time_series_graph(name=name, data=data, num_col=num_col)

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
        logging.error(f"Error generating Time series graph or performing analysis: {e}")
        return None

# Function to generate and save the time series graph
def time_series_graph(name, data, num_col):
    """
    Plots and saves the time series graph.
    """
    try:
        # Generate the time series plot
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(data.index, data[num_col], marker='o', markersize=5, linestyle='-', color='teal', linewidth=2)

        # Customize plot labels and title
        plt.xlabel("Time", fontsize=16, fontweight='bold')
        plt.ylabel(num_col, fontsize=16, fontweight='bold')
        plt.title(f"Time Series Plot: {name}", fontsize=18, fontweight='bold')
        plt.grid(True, linestyle='--', color='gray', alpha=0.6)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the plot
        chart_filename = os.path.join(name, "time_series.png")
        plt.savefig(chart_filename, dpi=100)
        plt.close()

        logging.info(f"Time series plot successfully saved as {chart_filename}")
        return chart_filename
    except Exception as e:
        logging.error(f"Error generating time series graph: {e}")
        return None

def regression_plot(name, y_true, y_pred, title='Actual vs Predicted', label='y = x'):
    """Plot regression chart.

    Args:
        name (str): The directory where the plot will be saved.
        y_true (array-like): The true target values.
        y_pred (array-like): The predicted values.
        title (str): The title of the plot (default 'Actual vs Predicted').
        label (str): The label for the line (default 'y = x').

    Returns:
        str: The path to the saved regression plot.
    """
    # Ensure the directory exists
    if not os.path.exists(name):
        os.makedirs(name)

    dpi = 100
    plt.figure(figsize=(8, 6), dpi=dpi)

    # Scatter plot of actual vs predicted values
    plt.scatter(y_true, y_pred, alpha=0.8, color='royalblue', label="Predicted vs Actual", s=50)

    # Plot the line y = x
    plt.plot(y_true, y_true, color='crimson', label=label, linewidth=2)

    # Set plot labels and title
    plt.xlabel('Actual', fontsize=16, fontweight='bold')
    plt.ylabel('Predicted', fontsize=16, fontweight='bold')
    plt.title(title, fontsize=18, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # Save the plot to the specified directory
    chart_file_name = os.path.join(name, "regression_plot.png")
    try:
        plt.savefig(chart_file_name, dpi=dpi)
        logging.info(f"Regression plot saved as {chart_file_name}")
    except Exception as e:
        logging.error(f"Failed to save regression plot: {e}")
    finally:
        plt.close()

    return chart_file_name

def confusion_matrix_plot(y_true, y_pred, labels, name):
    """
    Plots and saves the confusion matrix.
    """
    try:
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Visualize the confusion matrix using a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=True, annot_kws={"size": 14, "weight": "bold", "color": "black"}, square=True)
        plt.title("Confusion Matrix", fontsize=18, fontweight='bold')
        plt.xlabel('Predicted', fontsize=16, fontweight='bold')
        plt.ylabel('True', fontsize=16, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(rotation=45, va='top', fontsize=14)
        plt.tight_layout()
        # Save the plot to the specified filename
        chart_filename = os.path.join(name, "confustion_matrix.png")
        plt.savefig(chart_filename, dpi=100)
        plt.close()
        logging.info(f"Confusion matrix plot saved to {chart_filename}")
        
    except Exception as e:
        logging.error(f"Error plotting confusion matrix: {e}")


analysis_map = {
    'regression': regression,
    'kmeans_clustering': kmeans_clustering,
    'classification': classification,
    'time_series': time_series
}

def choose_analysis(name, data, api_key, analyses):
    """Perform all the provided analysis functions."""
    results = {}
    for analysis in analyses:
        analysis_normalized = analysis.strip().lower()
        func = analysis_map.get(analysis_normalized)
        
        if func:
            try:
                res = func(name, data, api_key)
            except Exception as e:
                logging.error(f'Error in {analysis} function: {str(e)}')
                res = None

            if res is not None:
                results[analysis] = res
        else:
            logging.warning(f"Analysis function {analysis} is not recognized.")
    
    return results


def perform_ml_analysis(name, df, api_key):
    """
    Perform in-depth analysis by consulting a language model (LLM) to suggest 
    appropriate machine learning techniques based on the provided dataset.
    """
    # Data cleaning
    df_cleaned = clean_data(df)
    if df_cleaned.empty or df_cleaned.shape[1] == 0:
        logging.error("The dataset is empty or has no usable columns.")
        return None
    
    analyses = ['regression','kmeans_clustering','classification','time_series']

    analysis_function_descriptions = [
        {
            'name': 'choose_analysis',
            'description': 'A function to choose all the relevant analysis to be performed for a dataset.',
            "parameters": {
                "type": "object",
                "properties": {
                    "analyses": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of analysis to perform in order. Eg. ['regression', 'time_series', 'kmeans_clustering']",
                    },
                },
                "required": ["analyses"]
            }
        }
    ]
    # Columns info to provide context to the LLM
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in df_cleaned.dtypes.items()])
    example_data = df_cleaned.head(1).to_dict(orient="records")
    
    order_list_analyses = "\n".join([f'{i+1}. {analysis_name}' for (i, analysis_name) in enumerate(analyses)])
    
    prompt = f"""\
    You are given a file {name}.csv. With features:{columns_info}
    Sample: {example_data}
    Note: Perform only the appropriate analyses.
    Analysis options: {order_list_analyses}
    Call the choose_analysis function with the correct options.
    """
    # Call the LLM API
    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=analysis_function_descriptions)

    try:
        params = json.loads(response['arguments'])
    except (KeyError, json.JSONDecodeError) as e:
        logging.error(f"Error parsing arguments from model response: {e}")
        return None
    analysis_func_name = response.get('name', '').strip().lower()
    # Ensure that the LLM returns 'choose_analysis' and not another function name
    if analysis_func_name != "choose_analysis":
        logging.error(f"Invalid function name returned by the model: {analysis_func_name}. Expected 'choose_analysis'.")
        return None
    logging.info(f"In-depth Analysis suggested by LLM: {params}")
    # Directly call the 'choose_analysis' function
    analysis_results = choose_analysis(name, df_cleaned, api_key, **params)
    return analysis_results


# Narrate a story
def create_story(name, description, image_files, api_key, model='gpt-4o-mini'):
    """
    Generates a compelling data analysis story in markdown format with embedded images using relative URLs
    pointing directly to the image filenames.
    
    Parameters:
    - name (str): Name of the dataset
    - description (str): Generic and in-depth data analysis.
    - image_files (list): List of image file paths (relative) to be included in the markdown.
    - api_key (str): API key for the external AI service.
    - model (str): The AI model to use (default is 'gpt-4o-mini').
    
    Returns:
    - str: The generated markdown content for the README.md.
    """
    if not description or not image_files or not api_key:
        raise ValueError("Description, image files, API key, and image folder are required")

    # API URL for AI model
    url = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'

    # Headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Prepare the images using relative URLs
    image_urls = [f"![Image]({image_file})" for image_file in image_files]

    prompt = f"Given dataset: {name}.csv\nData: {description}. Craft an engaging, in-depth narrative analysis in markdown format, highlighting key insights, trends, and actionable recommendations. Weave a story with creativity, bringing the data to life. Include images at relevant points using relative filenames (no full paths), placed logically within the narrative.The images should be referenced as markdown-style image links: {', '.join(image_urls)}"

    # Payload for the API request
    payload = {
        'model': model,
        "messages": [
            {'role': 'system', 'content': "You are a data analyst and storyteller. Present your findings as a creative and compelling narrative."},
            {'role': 'user', 'content': prompt},
        ],
        'temperature': 0.7,
        'max_tokens': 1800
    }

    try:
        # Send request to the AI API
        logging.info("Sending request to the AI API...")
        response = requests.post(url=url, headers=headers, json=payload, timeout=60)
        
        if response.ok:
            try:
                ai_response = response.json()  # Try to parse the JSON response
                
                # Check if 'choices' and other required keys are present
                if 'choices' in ai_response and ai_response['choices']:
                    result = ai_response["choices"][0]["message"]["content"].strip()
                    
                    logging.info(f"Monthly Cost: {ai_response.get('monthlyCost', 'N/A')}")
                    logging.info(f"Request successful, generated story.")
                    return result
                else:
                    # If the response doesn't have 'choices', log an error and raise an exception
                    logging.error("Invalid AI response: Missing 'choices' or 'message' content.")
                    raise ValueError("Error: Missing 'choices' or 'message' content in the AI response.")
            
            except ValueError as e:
                logging.error(f"Error parsing AI response: {e}")
                raise
            except KeyError as e:
                logging.error(f"Missing expected key in AI response: {e}")
                raise
            except requests.exceptions.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON from AI response: {e}")
                raise
        
        else:
            # If the response status is not OK, log the error and raise an exception
            logging.error(f"Error fetching the summary. Status code: {response.status_code}")
            logging.error(f"Response content: {response.content.decode('utf-8')}")
            raise RuntimeError(f"Error: Unable to fetch the analysis. Status code: {response.status_code}")
    
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        raise RuntimeError(f"Error: Request failed due to {e}")

def clean_image_urls(markdown_content):
    """
    Cleans image URLs by removing directory paths, leaving only filenames.
    For example: '![Image](media\\correlation_matrix.png)' becomes '![Image](correlation_matrix.png)'.
    
    Args:
    - markdown_content (str): The markdown text with image URLs.
    
    Returns:
    - str: The cleaned markdown content with proper image URLs.
    """
    # Regular expression to remove directory path before the image filename
    # This pattern finds `![Image](...)` and removes any directory path (e.g., `media\`)
    cleaned_content = re.sub(r'!\[Image\]\((.*?)\\(.*?)\)', r'![Image](\2)', markdown_content)
    
    return cleaned_content

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
    ml_analysis = perform_ml_analysis(name=name, df=df, api_key=api_key)

    list_chart = []
    for filename in os.listdir(name):
        if filename.endswith('.png'):
            relative_path = os.path.relpath(os.path.join(name, filename), start='.')  # Relative to the current directory
            list_chart.append(relative_path)
    description = {"generic_analysis": analysis, "ml_analysis": ml_analysis}
    content = create_story(name=name, description=description, image_files=list_chart, api_key=api_key)
    processed_content = clean_image_urls(content)
    write_file(name=name, text_content=processed_content)
    logging.info("Autolysis completed.")


if __name__ == "__main__":
    main()
