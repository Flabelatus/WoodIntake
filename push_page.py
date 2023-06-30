import os
import pandas as pd
import random
import numpy as np
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from reservation_system import reservation_design
from matching import matching_design, MC_match, match_requirement_dataset_improved

class push_page:
    """A class to read the saved csv data from the wood intake process"""

    def __init__(self, Digital_intake_class):
        DigIn = Digital_intake_class 
        self.matching_list = None
        self.wood_list = None
        self.requirement_list = None
        self.fields = DigIn.fields
        dataset = DigIn.data

        st.title("Form fitting - a general design")
        st.subheader("Example - how it works")

        st.text("A preset design is available and used")
        # show an image of the overview
        form_fitting_1_image = Image.open('form_fitting_1_image.png')

        st.image(form_fitting_1_image, caption='Form fitting image', width = 500)

        form_fitting_2_image = Image.open('form_fitting_2_image.png')

        st.image(form_fitting_2_image, caption='Form fitting image', width = 500)


        st.text('This can then be reserved')

        # image = Image.open('stool_image.jpg')
        # st.image(image, caption='Image of a stool')


        tab1, tab2, tab3 = st.tabs(["Generate requirements", "Match design", "Reserve matched wood"])

        with tab1:
            st.subheader("Generate a general requirements set of a stool")
            n_stools = st.slider('Number of stools:', 0, 5, 1)
            project_id = st.text_input('project id', 'Stool_1')
            
            # if st.button('Generate requirements'):
            #     st.write('General requirements are generated')
            #     DigIn.generate_requirements_stool(n_stools)
            #     # DigIn.generate_requirements(size = 4, n_planks = 30)
            #     requirement_df = pd.DataFrame(DigIn.requirement_list)
            #     # print(DigIn.requirement_list)
            #     st.write(requirement_df)
            #     # st.write("Requirements are saved in a CSV file")
            #     # requirement_df.to_csv('requirements.csv', index=False)

            if st.button('Generate requirements of a stool through API'):
                st.write('Requirements are generated through an API call')

                DigIn.generate_requirements_stool_api(n_stools = n_stools, project_id = project_id)
                self.requirement_list = DigIn.requirement_list.copy()
                requirement_df = pd.DataFrame(DigIn.requirement_list)
                st.write(requirement_df)

        with tab2:
            requirement_list = DigIn.read_requirements_from_client(project_id = project_id)
            # print(requirement_list)
            # print(wood_list)
            wood_list = DigIn.get_data_api()
            self.matching_df, unmatched_df, optimal_list = matching_design(requirement_list, wood_list)

        
        with tab3:
            matching_df = pd.read_csv('matching_df.csv')
            reservation_design(DigIn, matching_df)


        
