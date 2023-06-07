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
from graphical_elements import GraphicalElements

class pull_page:
    """A class to read the saved csv data from the wood intake process"""

    def __init__(self, Digital_intake_class):
        DigIn = Digital_intake_class 
        self.matching_list = None
        self.wood_list = None
        self.requirement_list = None
        self.fields = DigIn.fields

        st.title("Pull design from Grasshopper and match with available wood")

        st.subheader("Example how it works")
        st.text("Here is a visual shown of how it works")

        st.subheader('Pull a project saved in the database')

        project_id = st.text_input('What is the project id?', 'Stool_1')
        if st.button('Pull requirements through API'):
            st.write('Requirements are generated through an API call')
            DigIn.read_requirements_from_client(project_id = project_id)
            requirement_df = pd.DataFrame(DigIn.requirement_list)
            st.write(requirement_df)


        tab1, tab2, tab3 = st.tabs(["Show design", "Match design", "Reserve matched wood"])

        with tab1:
            if os.path.exists('requirements.csv'):
                requirement_df = pd.read_csv('requirements.csv')
                # length_values_req = requirement_df['Length']
                st.subheader('Length and width distribution in mm of the requirements\n')
                fig = GraphicalElements.barchart_plotly_one(requirement_df, color='red', requirements=True)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
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
                
        

        # def match_requirement_dataset_improved(self, requirement_list, n_runs = 10):

        
        with tab3:
        # st.subheader("Match the requirements with the available wood")
            st.subheader('Reserve the wood based on matched requirements')
            res_name = st.text_input('Reservation project', 'Stool')
            res_number = st.text_input('Reservation id', '1')
            st.write('The reservation will be on ', res_name + '-' + res_number)
            reservation(DigIn, res_name, res_number)
        


    def MC_match(self, DigIn, n_runs, option):
        if os.path.exists('requirements.csv'):
            requirement_df = pd.read_csv('requirements.csv')
            requirement_list = requirement_df.to_dict('records')

            # st.write(requirement_df)
            # print(requirement_list)
            matching_df, unmatched_df = DigIn.match_requirement_dataset_improved(requirement_list,
                                                                                 n_runs=n_runs, option=option)
            dataset = pd.DataFrame(DigIn.wood_list)

            # Visualize the matched planks
            if len(matching_df):
                
                st.write("Matching Dataframe is saved in a CSV file")
                matching_df.to_csv('matching_df.csv', index=False)
                st.subheader('Showing all the planks in the dataset and the matching requirements')
                fig = GraphicalElements(dataset).barchart_plotly_two_improved(matching_df, requirement_df)
                # fig = Graphical_elements(dataset).barchart_plotly_two(matching_df, requirement_df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write('Could not match any unreserved planks with the requirements')

            if option == 'Minimum waste':
                st.write('Waste is limited to {}'.format('?? FEATURE TO BE ADDED ??'))

            if len(unmatched_df):
                st.subheader(
                    'The following chart shows all the required planks that cannot be found in the database based on'
                    ' the used matching algorithm')
                fig = GraphicalElements(dataset).barchart_plotly_one(dataset=unmatched_df,
                                                                     color='maroon', requirements=True)
                st.plotly_chart(fig, use_container_width=True)

            return matching_df, unmatched_df
        else:
            st.write('Requirements are not found')


    def simple_match(self, DigIn):
        if os.path.exists('requirements.csv'):
            requirement_df = pd.read_csv('requirements.csv')
            requirement_list = requirement_df.to_dict('records')

            # st.write(requirement_df)
            # print(requirement_list)
            matching_df, unmatched_df = DigIn.match_requirements_dataset(requirement_list)
            dataset = pd.DataFrame(DigIn.wood_list)

            # Visualize the matched planks
            if len(matching_df):
                st.write(matching_df)
                st.write("Matching Dataframe is saved in a CSV file")
                matching_df.to_csv('matching_df.csv', index=False)
                st.subheader('Showing all the planks in the dataset and the matching requirements')
                fig = GraphicalElements(dataset).barchart_plotly_two_improved(matching_df, requirement_df)
                # fig = Graphical_elements(dataset).barchart_plotly_two(matching_df, requirement_df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write('Could not match any unreserved planks with the requirements')

            if len(unmatched_df):
                st.subheader(
                    'The following chart shows all the required planks that cannot be found in the database'
                    ' based on the used matching algorithm')
                fig = GraphicalElements(dataset).barchart_plotly_one(dataset=unmatched_df,
                                                                     color='maroon', requirements=True)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.write('Requirements are not found')
