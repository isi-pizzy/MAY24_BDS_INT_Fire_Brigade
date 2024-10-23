import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from PIL import Image
from pathlib import Path
import os

image_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'eda')
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def show_data_exploration():
    st.title("Exploratory Data Analysis")
    show_data_sets()
    target_variable()
    show_feature_engineering()
    show_data_visualization()
    show_preprocessing_steps()
    show_final_dataset()

def show_data_visualization():
    st.write('### 4. Statistical Analysis')
    st.write('We conducted several statistical analyses to evaluate the relevance of our data concerning the target variable.')
    
    images = {
        'Hour of the day': {
            'file_name': 'total_response_time_by_hour.png',
            'caption': 'Hour Impact on the Response Time',
            'description': 'The influence of the hour of the day on the response time of incidents.'
        },
        'Incident Categories': {
            'file_name': 'total_response_time_by_incident_group.png',
            'caption': 'Incident Categories Impact on Response Time',
            'description': 'The impact of various incident categories on response time. False Alarms are deliberately excluded.'
        },
        'Bank Holidays': {
            'file_name': 'bank_holidays_impact.png',
            'caption': 'Bank Holidays Impact on Response Time',
            'description': 'The notable impact of bank holidays on incident response times.'
        }
    }
    
    selected_image = st.selectbox('Select an analysis to view', list(images.keys()))
    
    if selected_image:
        image_info = images[selected_image]
        show_image(image_info['file_name'], image_info['caption'], image_info['description'])

def show_data_sets():
    st.write("### 1. Data Sets")
    st.write("""
    The exploratory data analysis (EDA) for this project involves four primary datasets, each serving a unique purpose:

    1. **[LFB Incident Dataset](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)**: Encompasses incidents handled by the London Fire Brigade (LFB) since January 2009, with 39 columns and 716,551 rows.
    2. **[LFB Mobilisation Dataset](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)**: Details the fire engines dispatched to incidents, with 22 columns and 2,373,348 rows, providing insights into LFB's response activities.
    3. **[Google Maps](https://www.google.com/maps/d/u/0/viewer?mid=1f3Kgp7Qx5v0w-sXKomdR8DzD9u4&ll=51.50696950000004%2C-0.2769568999999672&z=11)**: Provides the locations of fire stations across London, essential for spatial analysis.
    4. **[Bank Holidays Dataset](https://www.dmo.gov.uk/media/bfknrcrn/ukbankholidays-jul19.xls)**: Includes all bank holidays in London, useful for examining the impact on incident response times and patterns.
    """)
    
    incidents = pd.read_csv(os.path.join(data_path, 'incidents_prev.csv'))
    incidents['IncidentNumber'] = np.round(incidents['IncidentNumber'])
    mob = pd.read_csv(os.path.join(data_path, 'mob_prev.csv'))
    mob = mob.drop(['IncidentNumber.1'], axis=1)
    dataset = st.selectbox('Select Dataset', ('Incidents', 'Mobilisation'))

    if dataset == 'Incidents':
        st.write("Incidents")
        st.write(incidents.head(10))
    else:
        st.write("Mobilisation")
        st.write(mob.head(10))

def target_variable():
    st.write("### 2. Setting the Target Variable")
    st.write("""
    The target variable for this analysis is the total response time of the first pump arriving at the incident location, derived from the variable AttendanceTimeSeconds in the mobilisation dataset. We renamed it TotalResponseTime.
    """)
    show_image('mean_total_response_time_by_year.png', 'Average Total Response Time Over the Years', 
               'This graph illustrates the changes in average response time over the years.')
    show_image('total_response_time_distribution.png', 'Total Response Time Distribution', 
               'Distribution of total response times before logarithmic transformation to understand data spread.')

def show_feature_engineering():
    st.write("### 3. Feature Engineering")
    st.write("""
    Feature engineering is essential for transforming raw data into meaningful features that enhance model predictive power. In this analysis, several features were engineered:

    1. **From Maps Data**: Calculated the distance of each incident to the nearest fire station, aiding in understanding proximity's influence on response times.
    2. **From Geo Data**: Mapped London into a cell grid for granular analysis based on geographic regions.
    3. **From Bank Holidays**: Created a boolean feature indicating incidents occurring on bank holidays, investigating the effect on incident frequency and response.
    4. **From Date Column**: Extracted temporal features such as weekend occurrence, time of day, and day of the week, capturing temporal patterns in incidents.
    """)
    
    show_image('mean_total_response_time_heatmap_beforebinary.png', 'Mean Total Response Time per Grid Cell', 'This heatmap shows the mean total response times per grid cell.')

def show_preprocessing_steps():
    st.write("### 5. Removing Irrelevant Data")
    st.write("""
    To streamline analysis and enhance model performance, irrelevant data was removed:

    1. Focused solely on the first pump arriving at the incident to avoid noise from subsequent arrivals.
    2. Removed irrelevant columns to the target variable for a cleaner dataset.
    3. Eliminated rows with missing values (NaNs) to maintain data integrity.

    ### Preparing Data for Modeling
    Preprocessing steps for modeling:

    1. **One-Hot-Encoding**: Applied to prepare categorical features for modeling.
    2. **Cyclic Encoding**: Used for time-related features to correctly interpret cyclical patterns.
    3. **Logarithmic Transformation**: Applied to 'DistanceToStation' and 'TotalResponseTime' for a more normal distribution, beneficial for many machine learning algorithms.

    These steps collectively enhance data quality and relevance, establishing a solid foundation for modeling and analysis.
    """)

def show_final_dataset():
    st.write("### 6. Final Dataset")
    df_final_prev = pd.read_csv(os.path.join(data_path, 'df_prev.csv'))
    num_rows, num_cols = df_final_prev.shape
    num_rows = '1.537.704'
    st.write(f"The final DataFrame has {num_rows} rows and {num_cols} columns.")
    st.write(df_final_prev)
    
def show_image(file_name, caption, description):
    image = Image.open(os.path.join(image_path, file_name))
    st.image(image, caption=caption, use_column_width=True)
    st.write(description)

if __name__ == "__main__":
    show_data_exploration()
