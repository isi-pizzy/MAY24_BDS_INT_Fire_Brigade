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

def regression_design():
    st.subheader("1. Regression Design")

    # Block aligned text using HTML
    regression_paragraph = """
    <div style="text-align: justify;">
    In a first step, we opted for a <span style="font-weight: 900;">regression design</span> with a 
    <span style="font-weight: 900;">continuous dependent variable</span> 
    to predict the response time of the London Fire Brigade for a given incident.
    We employ aforementioned engineered features and estimate a structure of the form:
    </div>
    """
    st.markdown(regression_paragraph, unsafe_allow_html=True)

    st.markdown(r"""
                    $$
                    \text{ResponseTime}_i = \beta_0 + \beta_1 \text{Location}_i + \beta_2 \text{DateTime}_i + \beta_3 \text{IncidentCategory}_i + \epsilon_i,
                    $$
                    """)

    st.write("where:")
        
    st.markdown("""
                    - _i:_ individual incident
                    - _ResponseTime:_ logged continuous target variable
                    - _Location:_ logged distance to nearest LFB station, grid cell ID
                    - _DateTime:_ various datetime variables, including ‘IsWeekend’, ‘IsHoliday’ as well as cyclic encoded hour, day, month
                    - _IncidentCategory:_ incident type, property category.
                    """)

    # Block aligned text using HTML
    model_comparison_paragraph = """
    <div style="text-align: justify;">
    We employed various algorithms to achieve optimal performance: <span style="font-weight: 900;">Linear Regression</span>, 
    <span style="font-weight: 900;">SVR</span>, <span style="font-weight: 900;">ElasticNet</span>, <span style="font-weight: 900;">Random Forest</span>, and <span style="font-weight: 900;">XGBoost</span>. 
    The graph below allows for a comparison of their respective <span style="font-weight: 900;">R<sup>2</sup> values</span> on the test set (0.2 split throughout).
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
    <span style="font-weight: 900;">R<sup>2</sup> of 0.28</span>. However, this value remains low. 
    The table below enables to further deep dive into the performance of the regression models. 
    </div>
    """
    st.markdown(xgboost_performance_paragraph, unsafe_allow_html=True)

    st.write("")

    # Create a dictionary to store the test statistics for each model
    models_data = {
        'Model': ['ElasticNet', 'Linear', 'RandomForest', 'SVR', 'XGBoost'],
        'MAE': [
            0.25545761314687493,  # ElasticNet Test
            0.2519433325250212,   # Linear Test
            0.25199285030288776,  # RandomForest Test
            0.2561106142710682,   # SVR Test
            0.24655474317735396   # XGBoost Test
        ],
        'MSE': [
            0.11898889477455021,  # ElasticNet Test
            0.11717495251075341,  # Linear Test
            0.11922500758786787,  # RandomForest Test
            0.12388264556394395,  # SVR Test
            0.11306704579074599   # XGBoost Test
        ],
        'RMSE': [
            0.3449476696175091,   # ElasticNet Test
            0.3423082711690639,   # Linear Test
            0.3452897444000732,   # RandomForest Test
            0.35196966568717813,  # SVR Test
            0.3362544360908061    # XGBoost Test
        ],
        'R2': [
            0.24175655092793624,  # ElasticNet Test
            0.2533156955113437,   # Linear Test
            0.2402519483826373,   # RandomForest Test
            0.21057167031803015,  # SVR Test
            0.2794928725138395    # XGBoost Test
        ]
    }

    # Create a DataFrame from the dictionary
    df_test_statistics = pd.DataFrame(models_data)

    # Set the Model column as the index for better display
    df_test_statistics.set_index('Model', inplace=True)

    # Display the table
    st.table(df_test_statistics)




def baseline_binary_classification():
    st.subheader("2. Binary Classification Design")

    # Block aligned text using HTML
    classification_paragraph = """
    <div style="text-align: justify;">
    Because the performance of the regression design was unsatisfactory, we pivoted to a 
    <span style="font-weight: 900;">binary classification</span> of the problem. The 
    LFB set itself a goal of six minutes for the response time, 
    from the reception of an emergency call to the first pump arriving at the incident (LFB, 2022). 
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
    baseline_models_paragraph = """
    <div style="text-align: justify;">
    For the classification model, we estimate a similar structure as for the 
    regression design, utilizing the same features, 
    with the key difference of a <span style="font-weight: 900;">binary taregt variable</span> as compared to the continuous one.
    We tested several algorithms, to achieve the best possbile performance, of which three proved to be the most viable in terms of complexity and accuracy: 
    <span style="font-weight: 900;">Logistic Regression</span>,
    <span style="font-weight: 900;">Random Forest</span>, and <span style="font-weight: 900;">XGBoost</span>. 
    The graph below depicts the <span style="font-weight: 900;">accuracy</span> and the 
    <span style="font-weight: 900;">recall of the 0-class</span> for these baseline models and compares them.
    </div>
    """
    st.markdown(baseline_models_paragraph, unsafe_allow_html=True)

    st.write("")

    # Data extraction
    models = ['Random Forest', 'Logistic Regression', 'XGBoost']
    accuracy = [0.74, 0.75, 0.76]
    recall_class_0 = [0.39, 0.29, 0.36]

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

    # Set the x-axis limits
    ax.set_xlim(0.2, 0.8)

    ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    # Add data labels
    for bars in [bars1, bars2]:
        for bar in bars:
            # Width represents the value on the x-axis
            width = bar.get_width()
        
            # Adding the value text next to the bar
            ax.text(
                width + 0.01,  # Adjusts the position of the label to the right of the bar
                bar.get_y() + bar.get_height() / 2,  # Vertically centers the label
                f'{width:.2f}',  # Formats the text with two decimal places
                va='center',  # Vertically aligns the text to the center
                ha='left',  # Horizontally aligns the text to the left of the starting point
                color='white',  # Sets the text color
                fontsize=12  # Sets the font size
            )

    # General layout
    ax.set_yticks([r + bar_width / 2 for r in range(len(models))])
    ax.set_yticklabels(models, color='white')
    ax.invert_yaxis()  # Invert y-axis to have the first model on top
    ax.set_xlabel('Score', color='white')
    ax.legend(loc='lower right')

    # Set tick parameters
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Block aligned text using HTML
    recall_paragraph = """
    <div style="text-align: justify;">
    We primarily focus on the <span style="font-weight: 900;">recall of the 0-class</span> to emphasize identifying incidents 
    where the LFB did not meet the 6-minute response time target. By prioritizing recall, our model minimizes 
    the risk associated with failing to accurately predict these critical incidents.
    
    <ul>
        <li><span style="font-weight: 900;">Random Forest</span>: Achieves the highest recall for the 0-class among baseline models, making it the 
        most effective at identifying delayed responses.</li>
        <li><span style="font-weight: 900;">XGBoost</span>: Excels in overall accuracy, ensuring a balanced performance across all classes.</li>
    </ul>

    While recall of the 0-class is crucial for pinpointing delayed responses, <span style="font-weight: 900;">accuracy</span> is also important 
    as it reflects the model's ability to correctly predict across all incidents, thereby maintaining overall reliability.
    </div>
    """
    
    st.markdown(recall_paragraph, unsafe_allow_html=True)

    st.write("")

    # Classification reports
    classification_reports = {
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





def advanced_binary_classification():
    st.subheader("3. Model Optimization")

    # Block aligned text using HTML
    improvement_methods_paragraph = """
    <div style="text-align: justify;">
    To improve upon the baseline classification performance, we used:
    <ul>
        <li><span style="font-weight: 900;">PCA</span> for dimensionality reduction</li>
        <li><span style="font-weight: 900;">Undersampling</span> to balance the dataset</li>
        <li><span style="font-weight: 900;">Hyperparameter tuning</span> to find the best configurations</li>
        <li><span style="font-weight: 900;">Ensemble methods</span> to combine multiple models</li>
    </ul>
    </div>
    """
    st.markdown(improvement_methods_paragraph, unsafe_allow_html=True)

    st.markdown("#### Principal Component Analysis")

    # Block aligned text using HTML
    pca_paragraph = """
    <div style="text-align: justify;">
    In a context with a large feature space, <span style="font-weight: 900;">Principal Component Analysis (PCA)</span> 
    is beneficial for <span style="font-weight: 900;">reducing dimensionality</span> while retaining key information. PCA transforms features into uncorrelated components, 
    enhancing model efficiency and thus allowing us to improve model performance through the subsequent optimization steps. 
    Furthermore, it can assist in <span style="font-weight: 900;">interpretability</span>. The graph below depicts the cumulative explained variance by number of components.
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
    Based on the graph of explained variance by the number of components, we decided on using <span style="font-weight: 900;">23 components</span>, 
    which explain approximately <span style="font-weight: 900;">85% of the original variance</span>. This balance maintains substantial full feature space 
    information while significantly reducing the dimensionality.
    </div>
    """
    st.markdown(variance_paragraph, unsafe_allow_html=True)

    st.write("")

    st.markdown("#### Undersampling")

    # Block aligned text using HTML
    undersampling_paragraph = """
    <div style="text-align: justify;">
    In a binary classification problem where 29% of observations are class 0 (LFB response time over 6 minutes) 
    and 71% are class 1 (LFB response time under 6 minutes), <span style="font-weight: 900;">undersampling the majority class (class 1) balances the dataset</span>. This approach:
    <ul>
        <li>Helps the model focus on detecting the minority class 0, minimizing the risk of missing critical instances with slower response times</li>
        <li>Reduces the amount of data and hence computational demand (as compared to e.g. oversampling)</li>
    </ul>
    Specifically, we made use of so-called <span style="font-weight: 900;">random undersampling</span>.
    </div>
    """
    st.markdown(undersampling_paragraph, unsafe_allow_html=True)

    st.write("")

    st.markdown("#### Hyperparameter Tuning")

    # Block aligned text using HTML
    hyperparameter_paragraph = """
    <div style="text-align: justify;">
    <span style="font-weight: 900;">Hyperparameter tuning</span> can significantly improve model performance by optimizing configurations that guide the training. 
    The drop-down menu below shows our best hyperparameters, found using <span style="font-weight: 900;">GridSearchCV</span>, for the three aformentioned classification algorithms.
    </div>
    """
    st.markdown(hyperparameter_paragraph, unsafe_allow_html=True)

    st.write("")

    # Best hyperparameters for each model
    best_params = {
        'Logistic Regression': {'C': 5, 'penalty': 'l2', 'solver': 'lbfgs'},
        'Random Forest': {'criterion': 'gini', 'max_depth': 10, 'n_estimators': 200},
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

    st.markdown("#### Ensemble Methods")

    # Block aligned text using HTML
    ensemble_methods_paragraph = """
    <div style="text-align: justify;">
    To enhance our binary classification model performance, we implemented ensemble methods on top of 
    <span style="font-weight: 900;">PCA</span>, <span style="font-weight: 900;">undersampling</span>, and 
    <span style="font-weight: 900;">hyperparameter tuning</span>. Specifically, we utilized:
    <ul>
        <li>Bagging</li>
        <li>Boosting</li>
        <li>Voting classifier</li>
        <li>Stacking classifier</li>
    </ul>
    This approach leverages the strengths of multiple algorithms for better predictive power and robustness. Due to computational constraints, 
    we focused on six final models: <span style="font-weight: 900;">Random Forest</span>, <span style="font-weight: 900;">XGBoost enhanced by bagging</span>, 
    <span style="font-weight: 900;">Logistic Regression enhanced by bagging</span>, 
    <span style="font-weight: 900;">a stacking classifier</span>, <span style="font-weight: 900;">a soft voting classifier</span>, and 
    <span style="font-weight: 900;">a hard voting classifier</span>. These classifiers are constructed from the three aforementioned classification models: Logistic Regression, Random Forest, and XGBoost.
    </div>
    """
    st.markdown(ensemble_methods_paragraph, unsafe_allow_html=True)

    st.write("")

    st.subheader("4. Model Evaluation")

    # Block aligned text using HTML
    recall_performance_paragraph = """
    <div style="text-align: justify;">
    The following graph depicts the recall and accuracy performance of the six aforementioned models and compares them. As explained, we mainly focus on the 
    <span style="font-weight: 900;">recall of the 0 class</span>. A higher recall for this class 
    reduces the risk of missing critical incidents where the LFB did not meet its six-minute target. 
    </div>
    """
    st.markdown(recall_performance_paragraph, unsafe_allow_html=True)

    st.write("")

    # Data for the models
    data = {
        'Model': ['Random Forest', 'Voting Classifier (Soft)', 'XGBoosting (Bag)', 'Logistic Regression (Bag)', 'Stacking Classifier', 'Voting Classifier (Hard)'],
        'Recall (0)': [0.61, 0.64, 0.63, 0.61, 0.63, 0.64],
        'Accuracy': [0.70, 0.69, 0.70, 0.68, 0.70, 0.69]
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
    bars1 = ax.bar(index, df['Recall (0)'], bar_width, label='Recall (0)', color=colors_class0)
    bars2 = ax.bar(index + bar_width, df['Accuracy'], bar_width, label='Accuracy', color=colors_class1)

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
    ax.set_ylabel('Score', fontsize=14)
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
    Hence, we chose the hard voting classifier as our final model. The drop-down menu below allows to select from various other performance measures 
    to compare the six models, providing a comprehensive evaluation of their effectiveness.
    </div>
    """
    st.markdown(final_model_paragraph, unsafe_allow_html=True)

    st.write("")

    # Data for the models
    data = {
        'Model': ['Random Forest', 'Voting Classifier (Soft)', 'XGBoosting (Bag)', 'Logistic Regression (Bag)', 'Stacking Classifier', 'Voting Classifier (Hard)'],
        'Accuracy': [0.70, 0.69, 0.70, 0.68, 0.70, 0.69],
        'Precision (0)': [0.49, 0.48, 0.49, 0.47, 0.49, 0.48],
        'Recall (0)': [0.61, 0.64, 0.63, 0.61, 0.63, 0.64],
        'F1-Score (0)': [0.54, 0.55, 0.55, 0.53, 0.55, 0.55],
        'Precision (1)': [0.82, 0.83, 0.83, 0.81, 0.82, 0.83],
        'Recall (1)': [0.73, 0.71, 0.73, 0.71, 0.73, 0.72],
        'F1-Score (1)': [0.78, 0.76, 0.77, 0.76, 0.77, 0.77]
    }
      

    df = pd.DataFrame(data)

    # Sort the DataFrame alphabetically by 'Model'
    df = df.sort_values(by='Model').reset_index(drop=True)

    # Dropdown menu for selecting metric, sorted alphabetically
    metrics = [
        'Accuracy', 
        'F1-Score (0)', 'F1-Score (1)',  
        'Precision (0)', 'Precision (1)', 
        'Recall (0)', 'Recall (1)', 
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

    ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    # Set the background color to match Streamlit background
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    st.pyplot(fig)