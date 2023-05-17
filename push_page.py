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
