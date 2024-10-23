import streamlit as st
from streamlit_shap import st_shap
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from PIL import Image
import shap
import joblib
import os
import base64
from io import BytesIO


# Results

shap.initjs()  # Initialize SHAP JavaScript library at the beginning of your Streamlit app

### 1. Evaluation Comparison
### 1.1 Evaluation Metric Scores

def comp_table():
    data = {
        'Model': [
            'Random Forest', 
            'XGBoost', 
            'Voting Classifier (Soft)', 
            'Voting Classifier (Hard)',
            'Stacking Classifier'
        ],
        'Precision': [0.72, 0.73, 0.72, 0.72, 0.73],
        'Recall': [0.70, 0.70, 0.69, 0.69, 0.70],
        'F1-Score': [0.71, 0.71, 0.70, 0.70, 0.71],
        'ROC AUC': [0.7352, 0.7411, 0.7364, 0, 0.7403],  # 0 Placeholder for Voting Classifier (Hard)
        'Accuracy': [0.70, 0.6986, 0.6884, 0.6919, 0.6968],
        'Balanced Accuracy': [0.6733, 0.6756, 0.6756, 0.6755, 0.6768]
    }

    comp = pd.DataFrame(data)
    comp.set_index('Model', inplace=True)
    st.write("Evaluation Comparison of Final Model Scores")
    st.write("""
            The following table summarizes the performance metrics of the final models used in predicting the response time classes. Metrics such as Precision, Recall, F1-Score, ROC AUC, Accuracy, and Balanced Accuracy are included to provide a comprehensive evaluation of each model's effectiveness.
            """)
    st.dataframe(comp)
    st.write("""
            We can observe that the XGBoost and Stacking Classifier models perform similarly well across most metrics, with slight variations. The Random Forest model also demonstrates strong performance, while the Voting Classifiers (both Soft and Hard) show comparable, albeit slightly lower, scores.
            """)
    
def display_metrics_plot(): 
    image_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_evaluation_comparison')

    score_metrics = {
        "Precision": "precision.png",
        "Recall": "recall.png",
        "F1-Score": "f1_score.png",
        "ROC AUC Score": "roc_auc.png",
        "Accuracy": "accuracy.png",
        "Balanced Accuracy": "balanced_accuracy.png"
    }

    # Create a dropdown menu for selecting the score metric with a default option
    options = ["Select a Score Metric"] + list(score_metrics.keys())
    selected_metric = st.selectbox("Select a Score Metric:", options)

    # Load and display the corresponding image if a valid option is selected
    if selected_metric != "Select a Score Metric":
        image_file = os.path.join(image_path, score_metrics[selected_metric])
        image = Image.open(image_file)
        st.image(image, caption=selected_metric, use_column_width=True)

### 1.2 Recall Evaluation (focus on minority class)

def plot_recall_comp():
    st.write("""
            As pointed out above, we focus on the recall of the minority class, specifically the class where the response time goal of 6 minutes is not reached.
            """)
    st.write("""
             For the LFB, it is crucial to minimize false negatives, as these represent instances where a delayed response is not correctly identified. Reducing false negatives ensures that high-risk situations are accurately flagged, enabling the LFB to allocate resources more effectively and potentially save lives. Thus, we aim to reduce false negatives and thereby increase the recall of the minority class.
            """)
    
    recall_comp = {
        'Model': [
            'Random Forest', 
            'XGBoost', 
            'Voting Classifier (Soft)', 
            'Voting Classifier (Hard)',
            'Stacking Classifier'
        ],
        'Recall (Not Reached (>6 min))': [0.61, 0.62, 0.64, 0.64, 0.63],
        'Recall (Reached (<=6 min))': [0.73, 0.73, 0.71, 0.72, 0.73]
    }

    recall_comp = pd.DataFrame(recall_comp)
    best_model = 'Voting Classifier (Hard)'

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = range(len(recall_comp['Model']))

    # Plot Recall (Not Reached (>6 min))
    bars1 = ax.bar(index, recall_comp['Recall (Not Reached (>6 min))'], bar_width, label='Recall (Not Reached (>6 min))', color='lightcoral')

    # Plot Recall (Reached (<=6 min))
    bars2 = ax.bar([i + bar_width for i in index], recall_comp['Recall (Reached (<=6 min))'], bar_width, label='Recall (Reached (<=6 min))', color='lightgreen')

    # Highlight the bar with the highest Recall (Not Reached (>6 min))
    for i, bar in enumerate(bars1):
        if recall_comp['Model'][i] == best_model:
            bar.set_color('blue')

    # Add labels and title
    ax.set_xlabel('')
    ax.set_ylabel('Recall')
    ax.set_title('Recall Class Comparison')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(recall_comp['Model'], rotation=45, ha='right')
    ax.legend()

    # Adjust y-axis limits
    ax.set_ylim(0.55, 0.8)

    # Set styles
    ax.title.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.legend(frameon=False)

    # Set background to transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Remove grid lines
    ax.grid(False)

    plt.xticks(rotation=45, ha='right')
    plt.setp(ax.get_legend().get_texts(), color='white')

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_evaluation_comparison', 'recall_comparison.png')
    plt.savefig(save_path)
    st.pyplot(fig)

    st.write("""
            From the recall comparison plot, we can see that the Voting Classifier (Hard) has the highest recall for the minority class, indicating the best performance in identifying instances where the response time goal is not met. Based on the overall model evaluation and the need to balance performance across classes, we will use the Voting Classifier (Hard) as our final model to make predictions.
            """)

### Return functions for main streamlit file

def load_eval_functions():
    st.subheader('Evaluation Summary')
    return comp_table(), display_metrics_plot(), plot_recall_comp()



### 2 Prediction
### 2.1 Predicting Labels

def load_X_test_pca():
    parquet_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'X_test_pca.parquet')
    X_test_pca = pd.read_parquet(parquet_path)
    num_components = X_test_pca.shape[1]
    pca_feature_names = [f'PC{i+1}' for i in range(num_components)]
    return X_test_pca, pca_feature_names

def load_y_test_pca():
    parquet_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'y_test_pca.parquet')
    y_test_pca = pd.read_parquet(parquet_path)
    return y_test_pca

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.joblib')
    model = joblib.load(model_path)
    return model

def prediction():
    model = load_model()
    X_test_pca, pca_feature_names = load_X_test_pca()
    pred_values = model.predict(X_test_pca)
    return pred_values, pca_feature_names, X_test_pca

### 2.2 Prediction Plots

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap='Set2', colorbar=False)  # Disable the colorbar

    # Set edge color to white
    for spine in ax.spines.values():
        spine.set_edgecolor('none')

    ax.grid(False)

    # Manually add white lines around the squares
    for i in range(cm.shape[0] + 1):
        ax.axhline(i - 0.5, color='white', linewidth=2)
        ax.axvline(i - 0.5, color='white', linewidth=2)

    # Set background to transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Set text color to white
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Set the color of text in the cells to white
    for text in disp.text_.ravel():
        text.set_color('white')

    return fig

def plot_pred_dist(y_pred):
    fig, ax = plt.subplots()

    # Define colors using seaborn
    colors = sns.color_palette("Set2", 2)

    # Plot histogram
    counts, bins, patches = ax.hist(y_pred, bins=[-0.5, 0.5, 1.5], align='mid', rwidth=0.8)

    # Set colors for the histogram bars
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Goal not Reached (>6 min)', 'Goal reached (<=6 min)'])
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Predictions')

    # Set background to transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Set text color to white
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Remove grid lines
    ax.grid(False)

    return fig

def plot_pca_components(X_test_pca, pc_x, pc_y):
    fig, ax = plt.subplots(figsize=(10, 6))
    hue_labels = {0: 'Class 0', 1: 'Class 1'}
    
    scatter = sns.scatterplot(data=X_test_pca, x=pc_x, y=pc_y, hue='Prediction', palette='Set2', ax=ax)

    # Set styles
    ax.set_title(f'PCA Component Plot: {pc_x} vs {pc_y}')
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Set background to transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Remove grid lines
    ax.grid(False)

    # Manually set the legend to ensure correct labels
    handles, _ = scatter.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Class 0', 'Class 1'], title_fontsize='13', fontsize='11', loc='upper right')
    
    # Set legend styles
    legend = ax.get_legend()
    legend.get_frame().set_alpha(0)  # Set legend background to transparent
    plt.setp(legend.get_texts(), color='white')  # Set legend font color to white
    plt.setp(legend.get_title(), color='white')  # Set legend title color to white

    return fig




# Make Prediction
y_pred, pca_feature_names, X_test_pca = prediction()

# Load True Y Labels
y_true = load_y_test_pca()

# Create Prediction DataFrame
X_test_pca_df = X_test_pca.copy()
X_test_pca_df.columns = pca_feature_names
X_test_pca_df['Prediction'] = y_pred

### Return functions for main streamlit file

def load_pred_functions():
    st.subheader("1. Prediction")
    st.write("""
             In this section, we use our Voting Classifier (Hard) model to predict the response time classes. The model is evaluated on a test dataset of 307,541 data points to assess its generalization on unseen data.
             """)
    
    # Make Prediction Button
    st.write("""
            #### Class Prediction
            Our binary classification model aims to predict the response time classes based on various principal components derived from the input features and random undersampling techniques to balance the classes.
            """)
    
    st.write("""
            Click the *Make Prediction* button below to generate predictions for the test dataset.
            """)

    if st.button('Make Prediction'):
        st.write("Prediction:")
        st.dataframe(X_test_pca_df)

        # Interpretation of the Prediction DataFrame
        st.write("""
                The table above shows the principal components used for prediction and the corresponding predicted response time mode (0 or 1). The 'Prediction' column indicates whether the response time is classified as not reaching the target of 6 minutes (class 0) or reaching the 6-minute goal (class 1). These predictions are made on the test set to assess how well the model generalizes to new, unseen data.
                """)

    # Prediction Plot Dropdown
    st.write("""
            #### Prediction Evaluation
            """)
        
    st.write("""
            We evaluate the performance of our model using various visualization techniques to understand its effectiveness and potential areas for improvement:
            - **Distribution of Predictions**: Displays the frequency of predicted classes, providing insight into the overall prediction distribution and helping to identify any class imbalance in the model's predictions.
            - **Confusion Matrix**: Visualizes the performance of the classification model by showing the number of correct and incorrect predictions for each class, helping to identify how well the model distinguishes between different classes.
            - **PCA Component Plot**: The PCA component plot provides a visual indication whether the model is effectively using the principal components to distinguish between the two classes.
            """)
    
    plot_type = st.selectbox(
        "Choose a plot to display:",
        ["Select Plot", "Prediction Distribution", "Confusion Matrix", "PCA Component Plot"]
        )

    if plot_type == "Prediction Distribution":
        fig = plot_pred_dist(y_pred)
        st.pyplot(fig)
        st.write("""
                The distribution plot indicates that the model predicts more instances of 'Goal Reached (<=6 min)' compared to 'Goal not Reached (>6 min)', providing insight into the model's tendency and potential class imbalance in the predictions.
                """)
    
    elif plot_type == "Confusion Matrix":
        fig = plot_confusion_matrix(y_true, y_pred)
        st.pyplot(fig)
        st.write("""
                The confusion matrix shows that the model correctly predicted 155,420 instances of 'Reached (<=6 min)' but misclassified 32,859 instances of 'Not Reached (>6 min)' as 'Reached (<=6 min)', representing a considerable number of false positives.
                """)
        
    elif plot_type == "PCA Component Plot":
        pc_x = st.selectbox("Select X-axis PC", pca_feature_names, index=0)
        pc_y = st.selectbox("Select Y-axis PC", pca_feature_names, index=1)
        fig = plot_pca_components(X_test_pca_df, pc_x, pc_y)
        st.pyplot(fig)

    return 


### 3 Interpretability

# Define paths to SHAP values and explainers
xgb_shap_values_path = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'shap_values_xgb.pkl')
xgb_explainer_path = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'explainer_xgb.pkl')

rf_shap_values_path = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'shap_values_rf.pkl')
rf_explainer_path = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'explainer_rf.pkl')

logreg_shap_values_path = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'shap_values_logreg.pkl')
logreg_explainer_path = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'explainer_logreg.pkl')

# Load SHAP values and explainers
shap_values = {}
explainers = {}

if os.path.exists(xgb_shap_values_path) and os.path.exists(xgb_explainer_path):
    shap_values['XGBoost'] = joblib.load(xgb_shap_values_path)
    explainers['XGBoost'] = joblib.load(xgb_explainer_path)
    print(f"XGBoost SHAP values loaded successfully. Shape: {shap_values['XGBoost'].shape}")

if os.path.exists(rf_shap_values_path) and os.path.exists(rf_explainer_path):
    shap_values['Random Forest'] = joblib.load(rf_shap_values_path)
    explainers['Random Forest'] = joblib.load(rf_explainer_path)
    print(f"Random Forest SHAP values loaded successfully. Shape: {shap_values['Random Forest'].shape}")

if os.path.exists(logreg_shap_values_path) and os.path.exists(logreg_explainer_path):
    shap_values['Logistic Regression'] = joblib.load(logreg_shap_values_path)
    explainers['Logistic Regression'] = joblib.load(logreg_explainer_path)
    print(f"Logistic Regression SHAP values loaded successfully. Shape: {shap_values['Logistic Regression'].shape}")


### 3.1 Mean Shap Value Comparison in the Voting Classifier

def plot_shap_across_models():
    mean_abs_shap_comparison = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'mean_abs_shap_comparison.png')
    if os.path.exists(mean_abs_shap_comparison):
        default_plot_image = Image.open(mean_abs_shap_comparison)
        return default_plot_image
    else:
        return None

### 3.2 Mean Shap Value Plot

def plot_shap_values(model_name):
    if model_name == 'XGBoost':
        shap_sum_plot_xgboost = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_sum_plot_xgboost.png')
        if os.path.exists(shap_sum_plot_xgboost):
            plt_xgboost = Image.open(shap_sum_plot_xgboost)
            return plt_xgboost
        else:
            return None
    elif model_name == 'Random Forest':
        shap_sum_plot_rf = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_sum_plot_rf.png')
        if os.path.exists(shap_sum_plot_rf):
            plt_rf = Image.open(shap_sum_plot_rf)
            return plt_rf
        else:
            return None
    elif model_name == 'Logistic Regression':
        shap_sum_plot_logreg = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_sum_plot_logreg.png')
        if os.path.exists(shap_sum_plot_logreg):
            plt_logreg = Image.open(shap_sum_plot_logreg)
            return plt_logreg
        else:
            return None
    else:
        return None

### 3.3 SHAP Violin Plots for for Each Model

def plot_violin_votingclf():
    violin_votingclf = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_violin_plot_votingclf.png')
    if os.path.exists(violin_votingclf):
        violin_votingclf_img = Image.open(violin_votingclf)
        return violin_votingclf_img
    else:
        return None

def plot_shap_violins(model_name):
    if model_name == 'XGBoost':
        shap_violin_xgboost = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_violin_plot_xgboost.png')
        if os.path.exists(shap_violin_xgboost):
            violin_xgboost = Image.open(shap_violin_xgboost)
            return violin_xgboost
        else:
            return None
    elif model_name == 'Random Forest':
        shap_violin_rf = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_violin_plot_rf.png')
        if os.path.exists(shap_violin_rf):
            violin_rf = Image.open(shap_violin_rf)
            return violin_rf
        else:
            return None
    elif model_name == 'Logistic Regression':
        shap_violin_logreg = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'bs_shap_values', 'shap_violin_plot_logreg.png')
        if os.path.exists(shap_violin_logreg):
            violin_logreg = Image.open(shap_violin_logreg)
            return violin_logreg
        else:
            return None
    else:
        return None

### 3.4 Local Interpretation for one Observation

def local_interpretation(model_name, X_test_pca, pca_feature_names, index=3):
    try:
        if model_name in shap_values and model_name in explainers:
            shap_values_model = shap_values[model_name]
            explainer_model = explainers[model_name]

            # Debug: Print shapes
            print(f"SHAP values shape for {model_name}: {shap_values_model.shape}")
            print(f"Explainer expected value shape for {model_name}: {explainer_model.expected_value.shape}")

            # Handle Random Forest separately to specify the class index
            if model_name == 'Random Forest':

                rounded_shap_values_rf = np.round(shap_values_model, 3)
                rounded_X_test_pca = np.round(X_test_pca.iloc[index, :], 3)
                
                force_plot = shap.force_plot(
                    explainer_model.expected_value[1],
                    rounded_shap_values_rf[index, :],
                    rounded_X_test_pca,
                    feature_names=pca_feature_names
                )
                return force_plot

            else:
                rounded_shap_values = np.round(shap_values_model[index, :], 3)
                rounded_X_test_pca = np.round(X_test_pca.iloc[index, :], 3)

                force_plot = shap.force_plot(
                    explainer_model.expected_value,
                    rounded_shap_values,
                    rounded_X_test_pca,
                    feature_names=pca_feature_names
                )

            return force_plot

        elif model_name == 'Combined Voting Classifier':
            if all(m in shap_values and m in explainers for m in ['XGBoost', 'Random Forest', 'Logistic Regression']):
                shap_values_combined = (shap_values['XGBoost'] + shap_values['Random Forest'] + shap_values['Logistic Regression']) / 3
                rounded_shap_values_combined = np.round(shap_values_combined[index, :], 3)
                rounded_X_test_pca = np.round(X_test_pca.iloc[index, :], 3)

                force_plot_combined = shap.force_plot(
                    explainers['XGBoost'].expected_value,
                    rounded_shap_values_combined,
                    rounded_X_test_pca,
                    feature_names=pca_feature_names
                )

                return force_plot_combined

        print(f"SHAP values or explainer not found for {model_name}")
        return None
    except Exception as e:
        print(f"Error in generating SHAP force plot for {model_name} (Observation {index}): {e}")
        return None
    
### 3.5 Interactive Plot for first 1000 Observations

def interactive_force_plot(model_name, X_test_pca, pca_feature_names, start_index=0, num_observations=1000):
    try:
        if model_name in shap_values and model_name in explainers:
            shap_values_model = shap_values[model_name]
            explainer_model = explainers[model_name]

            end_index = start_index + num_observations
            X_range = X_test_pca.iloc[start_index:end_index]

            # Handle Random Forest separately without specifying class index
            if model_name == 'Random Forest':
                shap_values_range = shap_values_model[start_index:end_index]
                expected_value = explainer_model.expected_value[1]  # Assuming class index 1 for simplicity

                force_plot = shap.force_plot(
                    expected_value,
                    shap_values_range,
                    X_range,
                    feature_names=pca_feature_names
                )
            else:
                shap_values_range = shap_values_model[start_index:end_index]
                expected_value = explainer_model.expected_value

                force_plot = shap.force_plot(
                    expected_value,
                    shap_values_range,
                    X_range,
                    feature_names=pca_feature_names
                )

            return force_plot

        elif model_name == 'Combined Voting Classifier':
            end_index = start_index + num_observations  # Ensure end_index is defined here as well
            if all(m in shap_values and m in explainers for m in ['XGBoost', 'Random Forest', 'Logistic Regression']):
                shap_values_combined = (shap_values['XGBoost'] + shap_values['Random Forest'] + shap_values['Logistic Regression']) / 3
                shap_values_combined_range = shap_values_combined[start_index:end_index]
                expected_value = explainers['XGBoost'].expected_value

                force_plot_combined = shap.force_plot(
                    expected_value,
                    shap_values_combined_range,
                    X_test_pca.iloc[start_index:end_index],
                    feature_names=pca_feature_names
                )

                return force_plot_combined

        print(f"SHAP values or explainer not found for {model_name}")
        return None
    except Exception as e:
        print(f"Error in generating SHAP force plot for {model_name} (Observations {start_index} to {end_index}): {e}")
        return None

def generate_force_plot_html(force_plot):
    # Add CSS to change background color to white
    force_plot_html = f"""
    <html>
    <head>
        {shap.getjs()}
        <style>
            .shap-container {{
                background-color: white !important;
                padding: 20px;
                border-radius: 10px;
            }}
            .shap-plot {{
                background-color: white !important;
            }}
            .shap-plot > div {{
                background-color: white !important;
            }}
        </style>
    </head>
    <body>
        <div class="shap-container">
            {force_plot.html()}
        </div>
    </body>
    </html>
    """
    return force_plot_html


### Return interpretability functions to main streamlit file

def load_interpret_functions():
    st.subheader("2. Model Interpretation")
    st.write("""
    Model interpretability is crucial for understanding how our classification model makes predictions. It helps us identify which features are most influential in determining the response time mode, ensuring transparency and trust in the model's decisions. By analyzing interpretability metrics such as SHAP values, we can gain insights into the model's behavior, detect potential biases, and improve model performance by focusing on key features.
            """)

    st.write('#### Mean Absolute SHAP Values')
    st.write("""
    Mean Absolute SHAP Values accros each model in the voting classifier
             """)
    
    # Display mean_abs_shap_comparison
    mean_abs_shap_comparison = plot_shap_across_models()
    
    if mean_abs_shap_comparison:
        st.image(mean_abs_shap_comparison, caption="Mean absolute SHAP values accros each model in the voting classifier", use_column_width=True)
    else:
        st.write("Default plot image not found.")
    st.write("""
            The mean absolute SHAP values plot shows that PC6 and PC3 are the most influential features across the voting classifier models, with XGBoost and Logistic Regression assigning high importance to these features. This highlights the significant impact of these principal components in predicting the response variable across different models.
            """)
    # SHAP Summary Plots
    st.write('#### SHAP Summary Plots')
    st.write("""
            The SHAP summary plot explains the contribution of each principal component to the model's predictions. Higher SHAP values indicate greater importance in predicting whether the response time goal is reached. This helps us understand which features most influence the model's decision-making process in our classification problem.
            """)
    shap_model_name = st.selectbox(
        "Choose a model to display SHAP values:",
        ["Select", "XGBoost", "Random Forest", "Logistic Regression"]
    )

    if shap_model_name != "Select":
        shap_plot = plot_shap_values(shap_model_name)
        if shap_plot:
            st.image(shap_plot, caption=f"SHAP summary plot for {shap_model_name}", use_column_width=True)
        else:
            st.write("SHAP summary plot not found.")
    
    # SHAP Violin Plots
    st.write('#### SHAP Violin Plots')
    st.write("""
            The SHAP violin plot explains the distribution and impact of the principal components on the model's predictions. Each violin represents the distribution of SHAP values for a feature, indicating both the magnitude and the direction (positive or negative) of the feature's influence on the prediction. This visualization helps in understanding which features are most important in determining the response time mode.
             """)
    
    # Display violin plot of voting classifier
    voilin_votingclf = plot_violin_votingclf()
    
    if voilin_votingclf:
        st.image(voilin_votingclf, caption="Violin plot of SHAP values for combined models (Voting Classifier)", use_column_width=True)
    else:
        st.write("Default plot image not found.")
    
    st.write("""
             The violin plot of SHAP values for the Voting Classifier model shows that PC6 has the highest impact on model output, indicating it is the most influential feature. The wider the distribution, the greater the variation in the feature's impact, with high SHAP values denoting significant contributions to the prediction.
             """)

    st.write("""
            To gain more insights, the next plot displays the highest SHAP values and their variation on the feature importance for the individual models of the Voting Classifier.
             """)
    
    shap_model_name = st.selectbox(
        "Choose a model to display SHAP violin plot:",
        ["Select", "XGBoost", "Random Forest", "Logistic Regression"]
    )

    if shap_model_name != "Select":
        shap_plot = plot_shap_violins(shap_model_name)
        if shap_plot:
            st.image(shap_plot, caption=f"SHAP violin plot for {shap_model_name}", use_column_width=True)
        else:
            st.write("SHAP violin plot not found.")

    # SHAP Local Interpretation
    st.write('#### Local Interpretation')
    st.write("""
            The local interpretation helps in understanding the contribution of each feature for a model's single prediction.
            """)

    shap_model_name_local = st.selectbox(
        "Choose a model for local interpretation:",
        ["Select", "XGBoost", "Random Forest", "Logistic Regression", "Combined Voting Classifier"],
        key="local_interpretation"
    )

    if shap_model_name_local != "Select":
        observation_number = st.number_input(
            f"Enter the observation number (between 0 and {X_test_pca.shape[0] - 1}):",
            min_value=0, 
            max_value=X_test_pca.shape[0] - 1,
            step=1
        )

        if st.button('Generate SHAP Force Plot'):
            force_plot = local_interpretation(shap_model_name_local, X_test_pca, pca_feature_names, index=observation_number)
            if force_plot:
                force_plot_html = generate_force_plot_html(force_plot)
                components.html(force_plot_html, height=200)
            else:
                st.write("SHAP force plot not found or error in generating the plot.")

    # Interactive SHAP Force Plot for a range of observations
    st.write("""
            The next graph displays an interactive SHAP plot that helps in understanding the contribution of features over a range of observations.
            """)

    shap_model_name_interactive = st.selectbox(
        "Choose a model for interactive SHAP force plot:",
        ["Select", "XGBoost", "Random Forest", "Logistic Regression", "Combined Voting Classifier"],
        key="interactive_plot"
    )

    if shap_model_name_interactive != "Select":
        num_observations = st.number_input(
            "Enter the number of observations to visualize (e.g., 1000):",
            min_value=1, 
            max_value=X_test_pca.shape[0], 
            value=1000
        )

        start_index = st.slider(
            f"Select the start index for the range (0 to {X_test_pca.shape[0] - num_observations}):",
            min_value=0,
            max_value=X_test_pca.shape[0] - num_observations,
            step=1
        )

        if st.button('Generate Interactive SHAP Force Plot'):
            force_plot = interactive_force_plot(shap_model_name_interactive, X_test_pca, pca_feature_names, start_index=start_index, num_observations=num_observations)
            if force_plot:
                force_plot_html = generate_force_plot_html(force_plot)
                components.html(force_plot_html, height=420)
            else:
                st.write("SHAP force plot not found or error in generating the plot.")

# 4 Feature Interpretation
### 4.1 PC Load Reconstruction


# Load the PCA object
pca_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'pca_object.pkl')
pca = joblib.load(pca_path)

# Load the column names
X_columns_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'X_columns.pkl')
X_columns = joblib.load(X_columns_path)

def calculate_loadings(pca, X_columns):
    loadings = pca.components_.T  # Transpose to get the correct shape
    loadings_df = pd.DataFrame(loadings, index=X_columns, columns=[f'PC{i+1}' for i in range(loadings.shape[1])])
    return loadings_df

# Plot the loadings for a specific PC
def plot_loadings(loadings_df, pc, top_n=5):
    pc_loadings = loadings_df[pc].sort_values(ascending=False)
    pc_loadings_top_bottom = pd.concat([pc_loadings.head(top_n), pc_loadings.tail(top_n)])

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.set_palette('Set2')
    ax = sns.barplot(x=pc_loadings_top_bottom.values, y=pc_loadings_top_bottom.index)
    ax.set_title(f'Top and Bottom PCA Loadings for {pc}', fontsize=16, color='white')
    ax.set_xlabel('Loading Value', fontsize=14, color='white')
    ax.set_ylabel('Original Feature', fontsize=14, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.xticks(fontsize=12, color='white')
    plt.yticks(fontsize=12, color='white')
    plt.gca().set_facecolor('none')  # Set the background of the plot to be transparent

    buf = BytesIO()
    plt.savefig(buf, format='png', transparent=True)  # Save plot to the buffer
    plt.close()
    buf.seek(0)
    return buf


### 4.2 PC & Feature Correlation Matrix

def plot_loadings_heatmap(loadings_df, top_n_features=10):
    loadings_df_rounded = loadings_df.round(2)

    # Truncate feature labels to a maximum of 20 characters
    truncated_index = [feature[:20] for feature in loadings_df_rounded.index]
    loadings_df_rounded.index = truncated_index

    # Select top features
    top_features = loadings_df_rounded.abs().nlargest(top_n_features, loadings_df_rounded.columns).index

    # Create a heatmap data frame
    heatmap_df = loadings_df_rounded.loc[top_features, :]

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    ax = sns.heatmap(
        heatmap_df, annot=True, cmap='coolwarm', cbar_kws={'label': 'Loading Value'},
        annot_kws={"color": "black", "fontsize" : 10}, vmin=-1, vmax=1
    )
    ax.set_title(f'Top {top_n_features} Features for PCA Components', fontsize=16, color='white')
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(fontsize=10, color='white')
    plt.yticks(fontsize=12, color='white')
    plt.gca().set_facecolor('none')  # Set the background of the plot to be transparent

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    buf = BytesIO()
    plt.savefig(buf, format='png', transparent=True)  # Save plot to the buffer
    plt.close()
    buf.seek(0)
    return buf


### Return pc load functions to main streamlit file

def load_pc_load_functions():
    st.subheader("3. Feature Interpretation")
    st.write("""
    Feature interpretation using PCA loadings is crucial for understanding how original features contribute to the principal components. By examining the loadings, we can identify the most influential features that drive the variance captured by each PC. This analysis provides valuable insights into the underlying structure of the data, enabling more informed decisions and interpretations.
             """)

    st.write('#### Principal Component Loading')
    st.write("""
    The PCA Loading Plot shows how each original feature contributes to a selected principal component. Each bar represents a loading value, indicating the influence of a feature on the chosen PC.
            """)
    
    loadings_df = calculate_loadings(pca, X_columns)

    pc = st.selectbox('Select the PC you are interested in:', [f'PC{i+1}' for i in range(loadings_df.shape[1])])
    top_n = st.number_input('Enter the number of top and bottom features to display:', min_value=1, max_value=50, value=5)

    if st.button('Generate Loading Plot'):
        buf = plot_loadings(loadings_df, pc, top_n=top_n)
        st.image(buf, use_column_width=True)

        st.write("""
        The PCA Loading Plot helps us identify key features driving the variance in the selected principal component. Features with high positive or negative loadings are the most influential in shaping the patterns captured by the PC, providing insights into the data's underlying structure.
            """)
    
    st.write('#### Principal Component Correlation Matrix')
    st.write("""
    The PCA Loadings Heatmap visually represents the relationship between the original features and the principal components. Each cell shows the loading value, indicating the contribution of a feature to a specific principal component.
             """)

    top_n_features = st.number_input('Enter the number of top features to display:', min_value=1, max_value=loadings_df.shape[0], value=10)

    if st.button('Generate Loadings Heatmap'):
        buf = plot_loadings_heatmap(loadings_df, top_n_features=top_n_features)
        st.image(buf, use_column_width=True)
    
        st.write("""
        From the heatmap, we can identify which features most significantly impact each principal component.  Higher absolute values (positive or negative) indicate features with a strong influence, helping us understand the key factors that shape the data's structure.
            """)

