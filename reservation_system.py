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

class reservation:
    """A class to read the saved csv data from the wood intake process"""

    def __init__(self, Digital_intake_class, res_name, res_number):
        DigIn = Digital_intake_class 
        self.matching_list = None
        self.wood_list = None
        self.requirement_list = None
        self.fields = DigIn.fields
        self.res_name = res_name
        self.res_number = res_number

        if st.button('Reserve the matched wood'):
            self.reserve(DigIn)
            st.write('The matched wood is reserved!')
            st.write(pd.DataFrame(DigIn.wood_list))

        # st.subheader('Unreserve items')
        if st.button('Unreserve all wood'):
            self.unreserve(DigIn)

    def reserve(self, DigIn):
        
        matching_df = pd.read_csv('matching_df.csv')
        matching_list = matching_df.to_dict('records')

        for index, row in enumerate(matching_list):
            DigIn.wood_list[row['Wood list index']]['Reservation name'] = str(self.res_name + '-' + self.res_number)
            DigIn.wood_list[row['Wood list index']]['Reserved'] = True
            DigIn.wood_list[row['Wood list index']]['Reservation time'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        pd.DataFrame(DigIn.wood_list).to_csv('Generated_wood_data.csv', index=False)

    def unreserve(self, DigIn):
        
        dataset = pd.DataFrame(DigIn.wood_list)
        dataset['Reserved'] = False
        dataset['Reservation name'] = ''
        dataset['Reservation time'] = ''

        dataset.to_csv('Generated_wood_data.csv', index=False)
        DigIn.wood_list = dataset.to_dict('records')

        unres_name = st.text_input('Unreservation name', 'Javid')
        unres_number = st.text_input('Unreservation number', '1')
        st.write('The reservation on ', unres_name + '-' + unres_number, 'you want to unreserve')
        if st.button('Unreserve the items on this name - number'):
            dataset = pd.DataFrame(DigIn.wood_list)

            dataset.loc[dataset['Reservation name'] == str(
                unres_name + '-' + unres_number), 'Reserved'] = False
            dataset.loc[dataset['Reservation name'] == str(
                unres_name + '-' + unres_number), 'Reservation time'] = None
            dataset.loc[dataset['Reservation name'] == str(
                unres_name + '-' + unres_number), 'Reservation name'] = None

            dataset.to_csv('Generated_wood_data.csv', index=False)
            DigIn.wood_list = dataset.to_dict('records')

