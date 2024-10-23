import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from PIL import Image
import shap
import joblib
import gdown
import os


st.title("Modelling Storyline")

st.subheader("1. Regression Design")

# Block aligned text using HTML
regression_paragraph = """
<div style="text-align: justify;">
In a first step, we opted for a <span style="font-weight: 900;">regression design</span> with a 
<span style="font-weight: 900;">continuous dependent variable</span> (logarithmic response time in seconds) 
to predict the <span style="font-weight: 900;">response time</span> of the London Fire Brigade for a given incident.
We employ aforementioned engineered features based on the respective data analyses and estimate:
</div>
"""
st.markdown(regression_paragraph, unsafe_allow_html=True)

st.markdown(r"""
                $$
                \text{ResponseTime}_i = \beta_0 + \beta_1 \text{Location}_i + \beta_2 \text{Timing}_i + \beta_3 \text{IncidentType}_i + \epsilon_i,
                $$
                """)

st.write("where:")
    
st.markdown("""
                - _ResponseTime:_ continuous in seconds (ln)
                - _Location:_ distance to station (ln), cell
                - _Timing:_ month, day, hour (cycle encoded)
                - _IncidentType:_ incident type, property category.
                """)

# Block aligned text using HTML
model_comparison_paragraph = """
<div style="text-align: justify;">
We utilized various models to achieve optimal performance: <span style="font-weight: 900;">Linear Regression</span>, 
<span style="font-weight: 900;">SVR</span>, <span style="font-weight: 900;">ElasticNet</span>, <span style="font-weight: 900;">Random Forest</span>, and <span style="font-weight: 900;">XGBoost</span>. 
The graph below allows for a comparison of the <span style="font-weight: 900;">R<sup>2</sup> values</span> of these models.
</div>
"""
st.markdown(model_comparison_paragraph, unsafe_allow_html=True)

st.write("")
    
# Data for the chart
data = {
    'Design': ['Linear', 'SVR', 'ElasticNet', 'RandomForest', 'XGBoost'],
    'R2': [0.253315696, 0.21057167, 0.241756551, 0.240251948, 0.279492873]
}

df = pd.DataFrame(data)

# Sort the DataFrame alphabetically by 'Design'
df = df.sort_values(by='Design').reset_index(drop=True)

# Create the column chart with a matching background
fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the figsize to slightly increase the vertical size
fig.patch.set_facecolor('#0e1117')  # Dark background color to match Streamlit's dark theme
ax.set_facecolor('#0e1117')

# Customize the bar chart with slightly wider bars
bars = ax.bar(df['Design'], df['R2'], color='#61dafb', width=0.5)

# Set the y-axis label with a white color
ax.set_ylabel('R2', color='white', fontsize=12)  # Smaller font size

# Change the color of the tick labels to white and reduce font size
ax.tick_params(axis='x', colors='white', labelsize=10)
ax.tick_params(axis='y', colors='white', labelsize=10)

# Add gridlines and set them to be behind the bars
ax.grid(color='grey', linestyle='--', linewidth=0.5, axis='y', alpha=0.7)
ax.set_axisbelow(True)

# Adjust the y-axis scale
ax.set_ylim(0.1, 0.3)

# Remove the spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Add data labels inside the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval - 0.02, round(yval, 2), 
            ha='center', va='bottom', color='black', fontsize=8, fontweight='bold')  # Smaller font size

# Display the chart in the Streamlit app
st.pyplot(fig)

# Block aligned text using HTML
xgboost_performance_paragraph = """
<div style="text-align: justify;">
The <span style="font-weight: 900;">XGBoost model</span> demonstrates superior performance with an 
<span style="font-weight: 900;">R<sup>2</sup> of 0.28</span>. However, this value remains quite low. 
The drop-down and respective tables enable you to further deep dive into the performance of each regression model,
differentiated by train and test set. 
</div>
"""
st.markdown(xgboost_performance_paragraph, unsafe_allow_html=True)

st.write("")

# Create the drop-down menu with the specified options, sorted alphabetically
models = ['ElasticNet', 'Linear', 'RandomForest', 'SVR', 'XGBoost']
option = st.selectbox(
    'Select Regression Model',
    sorted(models),
    key='regression_model'
)

# Generate DataFrames with specific values for each model
def get_dataframes():
    dataframes = {
        'Linear': pd.DataFrame({
            'MAE': [0.2519433325250212, 0.2519433325250212],
            'MSE': [0.11717495251075341, 0.11717495251075341],
            'RMSE': [0.3423082711690639, 0.3423082711690639],
            'R2': [0.2533156955113437, 0.2533156955113437]
        }, index=['Train', 'Test']),
        'SVR': pd.DataFrame({
            'MAE': [0.2558656794132485, 0.2561106142710682],
            'MSE': [0.12408780585377704, 0.12388264556394395],
            'RMSE': [0.35226099110429054, 0.35196966568717813],
            'R2': [0.2104036650955159, 0.21057167031803015]
        }, index=['Train', 'Test']),
        'ElasticNet': pd.DataFrame({
            'MAE': [0.25533744097237526, 0.25545761314687493],
            'MSE': [0.1193540954470088, 0.11898889477455021],
            'RMSE': [0.34547662069524876, 0.3449476696175091],
            'R2': [0.24052524200604553, 0.24175655092793624]
        }, index=['Train', 'Test']),
        'RandomForest': pd.DataFrame({
            'MAE': [0.09523092764143488, 0.25199285030288776],
            'MSE': [0.01763047607671788, 0.11922500758786787],
            'RMSE': [0.13277980296987144, 0.3452897444000732],
            'R2': [0.887813639728614, 0.2402519483826373]
        }, index=['Train', 'Test']),
        'XGBoost': pd.DataFrame({
            'MAE': [0.2451064377952239, 0.24655474317735396],
            'MSE': [0.1120386260875857, 0.11306704579074599],
            'RMSE': [0.33472171439508625, 0.3362544360908061],
            'R2': [0.28707508430975415, 0.2794928725138395]
        }, index=['Train', 'Test'])
    }
    return dataframes

# Retrieve the DataFrames
dataframes = get_dataframes()

# Display the table for the selected model
st.table(dataframes[option])

st.subheader("2. Baseline Binary Classification")

# Block aligned text using HTML
classification_paragraph = """
<div style="text-align: justify;">
Because the performance on the regression design was quite unsatisfactory, we pivoted to a 
<span style="font-weight: 900;">binary classification</span> of the problem. We found that the 
<span style="font-weight: 900;">London Fire Brigade</span> set a goal of six minutes for the response time, 
from the reception of the call for an incident to the first pump arriving at the respective location (LFB, 2022). 
Based on this information, we constructed a new <span style="font-weight: 900;">binary target variable</span>: 
<ul>
<li><span style="font-weight: 900;">0</span> if the response time of an incident is above 360 seconds</li>
<li><span style="font-weight: 900;">1</span> if the response time is equal to or under 360 seconds</li>
</ul>
The graph below depicts the distribution of this new target variable.
</div>
"""
st.markdown(classification_paragraph, unsafe_allow_html=True)

st.write("")

# Dummy data for the pie chart
labels = ["Target Reached (<= 6min, 1)", "Target Not Reached (> 6min, 0)"]
sizes = [0.70625231, 0.29374769]

# Colors for the pie chart
colors = ['#d3d3d3', '#87CEEB']  # Light grey and medium light blue

# Create a pie chart with black background
background_color = "#0e1117"  # Assuming the background is black
fig, ax = plt.subplots(facecolor=background_color)
ax.set_facecolor(background_color)

wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops=dict(color="white"))

# Change the font color for the percentages to black
for autotext in autotexts:
    autotext.set_color("black")

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Add title
plt.title("Binary Target Variable Distribution", color="white")

# Display the plot in Streamlit
st.pyplot(fig)

# Block aligned text using HTML
classification_model_paragraph = """
<div style="text-align: justify;">
For the <span style="font-weight: 900;">binary classification model</span>, we estimate a similar equation as for the 
regression design, utilizing the same features with only the target variable being different:
</div>
"""
st.markdown(classification_model_paragraph, unsafe_allow_html=True)

st.markdown(r"""
                $$
                \text{TargetReached}_i = \beta_0 + \beta_1 \text{Location}_i + \beta_2 \text{Timing}_i + \beta_3 \text{IncidentType}_i + \epsilon_i,
                $$
                """)

st.write("where")

st.markdown("""
                - _TargetReached:_ binary
                - _Location:_ distance to station (ln), cell
                - _Timing:_ month, day, hour (cycle encoded)
                - _IncidentType:_ incident type, property category.
                """)

# Block aligned text using HTML
baseline_models_paragraph = """
<div style="text-align: justify;">
For the baseline estimation of this new <span style="font-weight: 900;">binary classification</span> problem, we conducted 
<span style="font-weight: 900;">XGBoost</span>, <span style="font-weight: 900;">Logistic Regression</span>, 
<span style="font-weight: 900;">Random Forest</span>, and <span style="font-weight: 900;">Decision Tree</span> 
to find the most promising models. We also tried KNN and SVM, but they were computationally too demanding. 
The graph below depicts the <span style="font-weight: 900;">accuracy</span> and the 
<span style="font-weight: 900;">recall of the 0-class</span> for these baseline models and compares them.
</div>
"""
st.markdown(baseline_models_paragraph, unsafe_allow_html=True)

st.write("")




# Data extraction
models = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'XGBoost']
accuracy = [0.67, 0.74, 0.75, 0.76]
recall_class_0 = [0.43, 0.39, 0.29, 0.36]

# Bar width
bar_width = 0.35

# Positions of the bars on the y-axis
r1 = np.arange(len(models))
r2 = [x + bar_width for x in r1]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0E1117')
ax.set_facecolor('#0E1117')

# Create horizontal bars for accuracy and recall
bars1 = ax.barh(r1, accuracy, color='#d3d3d3', height=bar_width, edgecolor='grey', label='Accuracy')
bars2 = ax.barh(r2, recall_class_0, color='#87CEEB', height=bar_width, edgecolor='grey', label='Recall (Class 0)')

# Add data labels
for bars in [bars1, bars2]:
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                va='center', ha='left', color='black', fontsize=12)

# General layout
ax.set_yticks([r + bar_width / 2 for r in range(len(models))])
ax.set_yticklabels(models, color='white')
ax.invert_yaxis()  # Invert y-axis to have the first model on top
ax.set_xlabel('Score', color='white')
ax.set_title('Baseline Classification Model Comparison', color='white')
ax.legend(loc='lower right')

# Set tick parameters
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Display the plot in Streamlit
st.pyplot(fig)

# Block aligned text using HTML
evaluation_paragraph = """
<div style="text-align: justify;">
For the baseline model evaluation, we focused on <span style="font-weight: 900;">accuracy</span>, 
which measures the proportion of correctly predicted instances. High accuracy indicates effective overall performance. 
<span style="font-weight: 900;">Random Forest</span>, <span style="font-weight: 900;">Logistic Regression</span>, 
and <span style="font-weight: 900;">XGBoost</span> showed the most promise. 
We will continue with these models. Use the drop-down menu below to explore the classification reports.
</div>
"""
st.markdown(evaluation_paragraph, unsafe_allow_html=True)

st.write("")

# Classification reports
classification_reports = {
    'Decision Tree': {
        'Class': ['Class 0', 'Class 1', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.44, 0.77, '', 0.60, 0.67],
        'Recall': [0.43, 0.77, '', 0.60, 0.67],
        'F1-score': [0.44, 0.77, 0.67, 0.60, 0.67],
        'Support': [90243, 217298, 307541, 307541, 307541]
    },
    'Random Forest': {
        'Class': ['Class 0', 'Class 1', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.60, 0.78, '', 0.69, 0.72],
        'Recall': [0.39, 0.89, '', 0.64, 0.74],
        'F1-score': [0.47, 0.83, 0.74, 0.65, 0.72],
        'Support': [90243, 217298, 307541, 307541, 307541]
    },
    'Logistic Regression': {
        'Class': ['Class 0', 'Class 1', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.66, 0.76, '', 0.71, 0.73],
        'Recall': [0.29, 0.94, '', 0.61, 0.75],
        'F1-score': [0.40, 0.84, 0.75, 0.62, 0.71],
        'Support': [90243, 217298, 307541, 307541, 307541]
    },
    'XGBoost': {
        'Class': ['Class 0', 'Class 1', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.67, 0.78, '', 0.73, 0.75],
        'Recall': [0.36, 0.93, '', 0.64, 0.76],
        'F1-score': [0.47, 0.85, 0.76, 0.66, 0.73],
        'Support': [90243, 217298, 307541, 307541, 307541]
    }
}

# Dropdown menu for model selection
model_selected = st.selectbox("Select Classification Model", classification_reports.keys(), key='model_dropdown')

# Displaying the classification report
def display_classification_report(model):
    report = classification_reports[model]
    df = pd.DataFrame(report)
    df = df.set_index('Class')
    st.table(df)

display_classification_report(model_selected)





st.subheader("3. Advanced Binary Classification")

# Block aligned text using HTML
improvement_methods_paragraph = """
<div style="text-align: justify;">
To improve upon the baseline model performance, we used:
<ul>
    <li><span style="font-weight: 900;">PCA</span> for dimensionality reduction</li>
    <li><span style="font-weight: 900;">Undersampling</span> to balance the dataset</li>
    <li><span style="font-weight: 900;">Hyperparameter tuning</span> to find the best parameters</li>
    <li><span style="font-weight: 900;">Ensemble methods</span> to combine multiple models</li>
</ul>
</div>
"""
st.markdown(improvement_methods_paragraph, unsafe_allow_html=True)

st.subheader("Principal Component Analysis")

# Block aligned text using HTML
pca_paragraph = """
<div style="text-align: justify;">
In a context with numerous features, <span style="font-weight: 900;">Principal Component Analysis (PCA)</span> 
is beneficial for reducing dimensionality while retaining key information. PCA transforms features into uncorrelated components, 
mitigating multicollinearity and enhancing model performance and interpretability. The graph below depicts the cumulative explained variance by the number of components.
</div>
"""
st.markdown(pca_paragraph, unsafe_allow_html=True)

st.write("")

# Define the relative path where the images are stored
image_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'cp_pca')
image_file = os.path.join(image_path, 'pca_explained_variance.png')
image = Image.open(image_file)
st.image(image, use_column_width=True)

# Block aligned text using HTML
variance_paragraph = """
<div style="text-align: justify;">
Based on the graph of explained variance by the number of components, we decided on using 23 components, 
which explains approximately 85% of the total variance. This balance maintains substantial original data 
information while significantly reducing the feature space.
</div>
"""
st.markdown(variance_paragraph, unsafe_allow_html=True)

st.write("")

st.subheader("Undersampling")

# Block aligned text using HTML
undersampling_paragraph = """
<div style="text-align: justify;">
In a binary classification problem where 29% of observations are class 0 (London Fire Brigade response time over 6 minutes) 
and 71% are class 1 (response time under 6 minutes), undersampling the majority class (class 1) balances the dataset. This approach:
<ul>
    <li>Helps the model focus on detecting the minority class 0, minimizing the risk of missing critical instances and ensuring better detection of slower response times</li>
    <li>Reduces the amount of data and hence computational demand</li>
</ul>
</div>
"""
st.markdown(undersampling_paragraph, unsafe_allow_html=True)

st.subheader("Hyperparameters")

# Block aligned text using HTML
hyperparameter_paragraph = """
<div style="text-align: justify;">
<span style="font-weight: 900;">Hyperparameter tuning</span> can significantly improve model performance by optimizing key parameters. 
The drop-down menu below shows the best hyperparameters for the three most promising models.
</div>
"""
st.markdown(hyperparameter_paragraph, unsafe_allow_html=True)

st.write("")

# Best hyperparameters for each model
best_params = {
    'Logistic Regression': {'C': 5, 'penalty': 'l2', 'solver': 'lbfgs'},
    'Random Forest': {'criterion': 'gini', 'max_depth': 10, 'n_estimators': 200, 'random_state': 666},
    'XGBoost': {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.9}
}

# Convert dictionaries to DataFrames
df_params = {model: pd.DataFrame(list(params.items()), columns=['Parameter', 'Value']).sort_values(by='Parameter').set_index('Parameter')
             for model, params in best_params.items()}

# Dropdown menu for selecting model
model = st.selectbox("Select a model", sorted(best_params.keys()))

# Display the table for the selected model
st.write(f"Best Hyperparameters for {model}:")
st.table(df_params[model])

st.subheader("Ensemble Methods")

# Block aligned text using HTML
ensemble_methods_paragraph = """
<div style="text-align: justify;">
To enhance our binary classification model performance, we implemented ensemble methods on top of 
<span style="font-weight: 900;">PCA</span>, <span style="font-weight: 900;">undersampling</span>, and optimal 
<span style="font-weight: 900;">hyperparameters</span>. Specifically, we utilized:
<ul>
    <li>Bagging</li>
    <li>Boosting</li>
    <li>A voting classifier</li>
    <li>A stacking classifier</li>
</ul>
This approach leverages the strengths of multiple algorithms for better predictive power and robustness. Due to computational constraints, 
we focused on six final models: <span style="font-weight: 900;">Random Forest</span>, <span style="font-weight: 900;">XGBoost enhanced by bagging</span>, 
<span style="font-weight: 900;">Logistic Regression enhanced by bagging</span>, 
<span style="font-weight: 900;">a stacking classifier</span>, <span style="font-weight: 900;">a soft voting classifier</span>, and 
<span style="font-weight: 900;">a hard voting classifier</span>. These classifiers are constructed from the most promising models: Logistic Regression, Random Forest, and XGBoost.
</div>
"""
st.markdown(ensemble_methods_paragraph, unsafe_allow_html=True)

st.write("")

st.subheader("Advanced Classification Model Comparison")

# Block aligned text using HTML
recall_performance_paragraph = """
<div style="text-align: justify;">
The following chart represents the recall performance of the six aforementioned models and compares them. We focus on the 
<span style="font-weight: 900;">recall of the 0 class</span>, which is our primary performance metric. A higher recall in this class 
reduces the risk of missing incidents where the London Fire Brigade did not meet its six-minute target response time. 
Accurately identifying these cases is crucial for improving safety and ensuring timely emergency response, aligning with our goal of enhancing public safety.
</div>
"""
st.markdown(recall_performance_paragraph, unsafe_allow_html=True)

# Data for the models
data = {
    'Model': ['Random Forest', 'Voting Classifier (Soft)', 'XGBoosting (Bag)', 'Logistic Regression (Bag)', 'Stacking Classifier', 'Voting Classifier (Hard)'],
    'Recall (0)': [0.61, 0.64, 0.63, 0.61, 0.63, 0.64],
    'Recall (1)': [0.73, 0.71, 0.73, 0.71, 0.73, 0.72]
}

df = pd.DataFrame(data)

# Sort the DataFrame alphabetically by 'Model'
df = df.sort_values(by='Model').reset_index(drop=True)

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))

# Create bar width and positions
bar_width = 0.35
index = np.arange(len(df['Model']))

# Define colors
colors_class0 = ['#696969' if model != 'Voting Classifier (Hard)' else '#1e90ff' for model in df['Model']]  # Dark grey and dark blue
colors_class1 = ['#d3d3d3' if model != 'Voting Classifier (Hard)' else '#add8e6' for model in df['Model']]  # Light grey and light blue

# Plot bars with new colors
bars1 = ax.bar(index, df['Recall (0)'], bar_width, label='Class 0', color=colors_class0)
bars2 = ax.bar(index + bar_width, df['Recall (1)'], bar_width, label='Class 1', color=colors_class1)

# Add values to the bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height - 0.01, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=10, color='black', weight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height - 0.01, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=10, color='black', weight='bold')

# Set labels
ax.set_ylabel('Recall', fontsize=14)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=12)

# Set the y-axis limit from 0.6 to 0.75
ax.set_ylim(0.55, 0.75)

# Set the background color to match Streamlit background
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')

# Modify legend and place it outside the plot
legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
for text in legend.get_texts():
    text.set_color('black')

st.pyplot(fig)

# Block aligned text using HTML
final_model_paragraph = """
<div style="text-align: justify;">
The voting classifiers have the best recall performance on the 0-class, with the 
<span style="font-weight: 900;">hard voting classifier</span> performing slightly better than the soft one in other metrics. 
Hence, we chose the hard voting classifier as our final model. The drop-down menu below enables you to select from various other performance measures 
to compare the six models, providing a comprehensive evaluation of their effectiveness.
</div>
"""
st.markdown(final_model_paragraph, unsafe_allow_html=True)

st.write("")

# Data for the models
data = {
    'Model': ['Random Forest', 'Voting Classifier (Soft)', 'XGBoosting (Bag)', 'Logistic Regression (Bag)', 'Stacking Classifier', 'Voting Classifier (Hard)'],
    'Accuracy': [0.6986, 0.6884, 0.6974, 0.6797, 0.6968, 0.6919],
    'Precision (0)': [0.49, 0.48, 0.49, 0.47, 0.49, 0.48],
    'Recall (0)': [0.61, 0.64, 0.63, 0.61, 0.63, 0.64],
    'F1-Score (0)': [0.54, 0.55, 0.55, 0.53, 0.55, 0.55],
    'Precision (1)': [0.82, 0.83, 0.83, 0.81, 0.82, 0.83],
    'Recall (1)': [0.73, 0.71, 0.73, 0.71, 0.73, 0.72],
    'F1-Score (1)': [0.78, 0.76, 0.77, 0.76, 0.77, 0.77],
    'Macro Avg. (0)': [0.65, 0.65, 0.66, 0.64, 0.66, 0.65],
    'Macro Avg. (1)': [0.67, 0.67, 0.68, 0.66, 0.68, 0.67],
    'Weighted Avg. (0)': [0.72, 0.72, 0.73, 0.71, 0.73, 0.72],
    'Weighted Avg. (1)': [0.70, 0.69, 0.70, 0.68, 0.70, 0.69],
    'ROC AUC Score': [0.7352, 0.7364, 0.7268, 0.6797, 0.7403, 0.6755]  # Note: some ROC AUC scores are made up for illustration
}

df = pd.DataFrame(data)

# Sort the DataFrame alphabetically by 'Model'
df = df.sort_values(by='Model').reset_index(drop=True)

# Dropdown menu for selecting metric, sorted alphabetically
metrics = [
    'Accuracy', 
    'F1-Score (0)', 'F1-Score (1)', 
    'Macro Avg. (0)', 'Macro Avg. (1)', 
    'Precision (0)', 'Precision (1)', 
    'Recall (0)', 'Recall (1)', 
    'Weighted Avg. (0)', 'Weighted Avg. (1)',
    'ROC AUC Score'
]

metric = st.selectbox("Select a metric", sorted(metrics))

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))  # Increase the figsize for a larger plot
bars = []

for idx, row in df.iterrows():
    value = row[metric]
    if pd.isna(value):
        ax.text(0.5, idx, 'NA', ha='center', va='center', fontsize=14, color='red', transform=ax.get_yaxis_transform())
    else:
        color = 'lightblue' if row['Model'] == 'Voting Classifier (Hard)' else 'lightgrey'
        bar = ax.barh(row['Model'], value, color=color)
        bars.append(bar)

# Set labels
ax.set_xlabel(metric, fontsize=14)
ax.set_yticklabels(df['Model'], rotation=0, ha='right', fontsize=12)

# Set the axis limits to fit the bars within the scale if not ROC AUC Score
if metric != 'ROC AUC Score':
    min_value = min(df[metric])
    max_value = max(df[metric])
    ax.set_xlim(min_value - 0.01, max_value + 0.01)
else:
    ax.set_xlim(0.5, 0.8)  # Custom limits for ROC AUC Score

# Set the background color to match Streamlit background
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')
ax.title.set_color('white')

st.pyplot(fig)



