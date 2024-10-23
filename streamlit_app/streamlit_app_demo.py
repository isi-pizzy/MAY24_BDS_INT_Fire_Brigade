import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'tabs'))
import eda
import results
import conclusion
import model2
import home

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = Path(__file__).resolve().parent / 'css' / 'style.css'
local_css(css_path)

# Streamlit UI
st.sidebar.title("Table of Contents")
pages = ["Home", "Data", "Model", "Prediction", "Conclusion", "About"]
page = st.sidebar.radio("Go to", pages)
st.sidebar.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
image_path = os.path.join(os.path.dirname(__file__), 'assets', 'logo-datascientest.png')
st.sidebar.image(image_path, use_column_width=True)

# Home
if page == pages[0]:
    home.homepage()
    
# Data
if page == pages[1]:
    eda.show_data_exploration()
    

# Model
if page == pages[2]:
    st.title("Modelling Storyline")
    model2.regression_design()
    model2.baseline_binary_classification()
    model2.advanced_binary_classification()


# Prediction
if page == pages[3]:
    st.title("Prediction & Interpretation")
   
    #results.load_eval_functions()
    results.load_pred_functions()
    results.load_interpret_functions()
    results.load_pc_load_functions()
    
    
# Conclusion
if page == pages[4]:
    st.title("Conclusion")

    conclusion.conclusion_text()


# About
if page == pages[5]:
    st.title("About")
    
    st.write("""
            The classification of the **London Fire Brigade's** response times is a capstone machine learning project from the Data Scientist bootcamp at [DataScientest.com](https://datascientest.com), in cooperation with [Panthéon-Sorbonne University](https://www.pantheonsorbonne.fr/).
            """)

    st.write("### Project Members:")
    st.write("""
            - **Ismarah MAIER** [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://www.linkedin.com/in/ismarah-maier-18496613b/)
            - **Clemens PAULSEN** [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://www.linkedin.com/in/clemens-paulsen-a65a5a155/)
            - **Dr. Benjamin SCHELLINGER** [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://www.linkedin.com/in/benjaminschellinger/)
            """)

    st.write("### Project Mentor:")
    st.write("""
    - **Yazid MSAADI** (DataScientest) [![Email](https://img.shields.io/badge/Email-red)](mailto:yazid.m@datascientest.com)
    """)

    st.write("### Github:")
    st.write("[LFB project](https://github.com/DataScientest-Studio/MAY24_BDS_INT_Fire_Brigade.git)")

    st.write("### References:")
    st.write("""
            - Buffington, T. and Ezekoye, O. A. (2019). *Statistical Analysis of Fire Department Response Times and Effects on Fire Outcomes in the United States.* Fire Technol 55, 2369–2393. https://doi.org/10.1007/s10694-019-00870-4
            - Google Maps (2024). *London's fire stations (V2)*, url: https://www.google.com/maps/d/u/0/viewer?mid=1f3Kgp7Qx5v0w-sXKomdR8DzD9u4&ll=51.50696950000004%2C-0.2769568999999672&z=11
            - Hewitt, M., Biermann, F., and Greatbatch, I. (2022). *The Economic and Social Value of UK Fire and Rescue Services.*, National Fire Chiefs Council, url: https://nfcc.org.uk/wp-content/uploads/2023/09/The-Economic-and-Social-Value-of-UK-Fire-and-Rescue-Services.pdf
            - Home Office (2023). *Detailed analysis of response times to fires attended by fire and rescue services: England, April 2022 to March 2023.* GOV.UK., url: https://www.gov.uk/government/statistics/detailed-analysis-of-response-times-to-fires-year-to-march-2023/detailed-analysis-of-response-times-to-fires-attended-by-fire-and-rescue-services-england-april-2022-to-march-2023
            - Jaldell, H. (2013). *Cost-benefit analyses of sprinklers in nursing homes for elderly.* Journal of Benefit-Cost Analysis. 2013;4(2):209-235. doi:10.1515/jbca-2012-0004
            - LFB Information Management (2024a). *London Fire Brigade Incident Records*, url: https://data.london.gov.uk/dataset/london-fire-brigade-incident-records
            - LFB Information Management (2024b). *London Fire Brigade Mobilisation Records*, url: https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records
            - LFB – London Fire Brigade. (2022). Measuring our success 2023-2029. url: https://www.london-fire.gov.uk/media/6686/crmp-metrics-30-may.pdf
            - Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* In Advances in Neural Information Processing Systems (pp. 4765–4774). url: https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf
            - Strumbelj, E., & Kononenko, I. (2010). *An Efficient Explanation of Individual Classifications using Game Theory.* Journal of Machine Learning Research, 11, 1–18. url: http://jmlr.org/papers/volume11/strumbelj10a/strumbelj10a.pdf
            - United Kingdom Debt Management Office (2024). *Bank Holidays Dataset*, url: https://www.dmo.gov.uk/media/bfknrcrn/ukbankholidays-jul19.xls
              """)