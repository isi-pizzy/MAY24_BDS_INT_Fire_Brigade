import streamlit as st
from PIL import Image
import os

image_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images')
# image source: https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.magirusgroup.com%2Fde%2Fen%2Fserving-heroes%2Fdeliveries%2Fdetail%2Fdelivery%2Fthree-m64l-for-london-11-2021%2F&psig=AOvVaw0EtxRpWJgRz23bLB3_HHf0&ust=1722337783378000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCLjHxMCOzIcDFQAAAAAdAAAAABAE

def homepage():
    st.title("Response Time Prediction for the LONDON FIRE BRIGADE")

    st.write("""##""")
                 
    show_image('LFB_Truck.jpg')
    
    st.write("""
             ### 1. Context
             The London Fire Brigade (LFB) Response Time project is dedicated to analyzing, predicting, and optimizing the response times of the LFB, the busiest fire and rescue service in the United Kingdom and one of the largest in the world. Swift and precise responses are vital for mitigating damage caused by fires and other emergencies.
             """)
    st.write("""
             ### 2. Motivation & Objectives
             Leveraging advanced machine learning techniques, this project seeks to significantly enhance the operational efficiency of the LFB, contributing to the economic and scientific advancement of emergency response services. Through comprehensive data analysis and modeling, we provide insights that can improve the effectiveness of fire brigade operations, ultimately fostering safer communities.
             
             Economically, optimizing response times offers both direct and indirect financial benefits. Reduced response times can decrease the extent of fire damage, thereby lowering repair and insurance costs.
             """)
    st.write("""
            ### 3. Focus
             To minimize noise and enhance our results, we focus exclusively on the first pump arriving at the incident following the initial call to the LFB. This approach excludes subsequent calls and additional fire pumps that may arrive later. This decision is grounded in the [Community Risk Management Plan](https://www.london-fire.gov.uk/media/6686/crmp-metrics-30-may.pdf) published by the LFB in May 2022, which states: "We expect to respond to an incident with the first appliance arriving within 6 minutes." Consequently, our project is defined as a binary classification problem after initial tests with baseline regression models.
             """)
    st.write("""
             ### 4. Results
             Our final outcome is a Voting Classifier model designed to predict whether the first pump will arrive at the incident location within 6 minutes or not.
             """)
    
def show_image(file_name):
    image = Image.open(os.path.join(image_path, file_name))
    st.image(image, use_column_width=True)