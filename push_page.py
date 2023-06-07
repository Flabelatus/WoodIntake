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
from reservation_system import reservation

class push_page:
    """A class to read the saved csv data from the wood intake process"""

    def __init__(self, Digital_intake_class):
        DigIn = Digital_intake_class 
        self.matching_list = None
        self.wood_list = None
        self.requirement_list = None
        self.fields = DigIn.fields
        dataset = DigIn.data

        image = Image.open('stool_image.jpg')
        st.image(image, caption='Image of a stool')

        st.title("Push - a general design")
        st.subheader("Example - how it works")
        st.text("Here is an example of how it works")
        st.subheader("Generate a general requirements set of a stool")
        n_stools = st.slider('Number of stools:', 0, 5, 1)
        if st.button('Generate requirements'):
            st.write('General requirements are generated')
            DigIn.generate_requirements_stool(n_stools)
            # DigIn.generate_requirements(size = 4, n_planks = 30)
            requirement_df = pd.DataFrame(DigIn.requirement_list)
            # print(DigIn.requirement_list)
            st.write(requirement_df)
            st.write("Requirements are saved in a CSV file")
            requirement_df.to_csv('requirements.csv', index=False)

        if st.button('Generate requirements of a stool through API'):
            st.write('Requirements are generated through an API call')
            print(n_stools)
            DigIn.generate_requirements_stool_api(n_stools = n_stools)
            requirement_df = pd.DataFrame(DigIn.requirement_list)
            st.write(requirement_df)



        option = st.selectbox(
            'How would you like to optimize the matching algorithm?',
            ('Minimum waste', 'Keep long planks', 'Most parts found in database', 'Minimum cuts needed'))

        n_runs = st.slider('Number of runs for Monte Carlo:', 1, 100, 30)
        if st.button('Match the requirements with the available wood - improved - Monte Carlo'):
            matching_df, unmatched_df = self.MC_match(DigIn, n_runs, option)

            if st.button('Send POST call (to Grasshopper?/DB) (test button) with this matching df'):
                st.write(matching_df)
                st.write("For now this doesn't do anything yet")


        if st.button('Match the requirements with the available wood - simple'):
            self.simple_match(DigIn)
            
        # st.subheader("Match the requirements with the available wood")
        st.subheader('Reserve the wood based on matched requirements')
        res_name = st.text_input('Reservation name', 'Javid')
        res_number = st.text_input('Reservation number', '1')
        st.write('The reservation will be on ', res_name + '-' + res_number)
        reservation(DigIn, res_name, res_number)

        st.header("Checking for specific pieces to use in Grasshopper to check if they match")

        DigIn.wood_list = DigIn.get_data_api()
        dataset = pd.DataFrame(DigIn.wood_list)
        st.write(dataset)

        st.subheader(f"Filter desired pieces")
        filtered_df = dataset.copy()

        # for filter_criteria in ['Length', 'Width', 'Height']:
        filter_criteria = 'Length'
        slider_min_val = min(sorted(dataset[filter_criteria]))
        slider_max_val = max(sorted(dataset[filter_criteria]))
        slider_length = st.select_slider('{} in mm'.format(filter_criteria),
                                         options=range(slider_min_val, slider_max_val + 1),
                                         value=(slider_min_val, slider_max_val), key=filter_criteria)
        filtered = [
            row for index, row in filtered_df.iterrows()
            if slider_length[1] >= row[filter_criteria] >= slider_length[0]
        ]
        filtered_df = pd.DataFrame(filtered)

        filter_criteria = 'Width'
        slider_min_val = min(sorted(dataset[filter_criteria]))
        slider_max_val = max(sorted(dataset[filter_criteria]))
        slider_width = st.select_slider('{} in mm'.format(filter_criteria),
                                        options=range(slider_min_val, slider_max_val + 1),
                                        value=(slider_min_val, slider_max_val), key=filter_criteria)
        filtered = [
            row for index, row in filtered_df.iterrows()
            if slider_width[1] >= row[filter_criteria] >= slider_width[0]
        ]
        filtered_df = pd.DataFrame(filtered)

        filter_criteria = 'Height'
        slider_min_val = min(sorted(dataset[filter_criteria]))
        slider_max_val = max(sorted(dataset[filter_criteria]))
        slider_height = st.select_slider('{} in mm'.format(filter_criteria),
                                         options=range(slider_min_val, slider_max_val + 1),
                                         value=(slider_min_val, slider_max_val), key=filter_criteria)
        filtered = [
            row for index, row in filtered_df.iterrows()
            if slider_height[1] >= row[filter_criteria] >= slider_height[0]
        ]
        filtered_df = pd.DataFrame(filtered)

        filter_criteria = 'Type'
        slider_min_val = 'Softwood'
        slider_max_val = 'Hardwood'
        slider_height = st.select_slider('{} of wood'.format(filter_criteria), options=['Softwood', 'Hardwood'],
                                         value=(slider_min_val, slider_max_val), key=filter_criteria)
        filtered = [
            row for index, row in filtered_df.iterrows()
            if row[filter_criteria] == slider_height[1] or row[filter_criteria] == slider_height[0]
        ]
        filtered_df = pd.DataFrame(filtered)

        filter_criteria = 'Reserved'
        slider_min_val = True
        slider_max_val = False
        slider_height = st.radio('Is the piece reserved already?'.format(filter_criteria), options=[True, False])
        filtered = [
            row for index, row in filtered_df.iterrows()
            if row[filter_criteria] == slider_height[1] or row[filter_criteria] == slider_height[0]
        ]
        filtered_df = pd.DataFrame(filtered)

        if len(filtered_df):
            st.write("The filtered items in the table are the ID values"
                     " of the pieces under the selected criteria")
            st.write(filtered_df)
        else:
            st.write('No piece found matching the desired filter')
        # st.write(dataset)
        if st.button('Reserve the filtered wood'):
            matching_list = filtered_df.to_dict('records')

            for index, row in enumerate(matching_list):
                DigIn.wood_list[row['Index']]['Reservation name'] = str(res_name + '-' + res_number)
                DigIn.wood_list[row['Index']]['Reserved'] = True
                DigIn.wood_list[row['Index']]['Reservation time'] = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.write('The matched wood is reserved!')
            st.write(pd.DataFrame(DigIn.wood_list))

            pd.DataFrame(DigIn.wood_list).to_csv('Generated_wood_data.csv', index=False)

        st.download_button('Download Selection', filtered_df.to_csv(), mime='text/csv')
