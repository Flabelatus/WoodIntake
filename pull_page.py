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
from graphical_elements import GraphicalElements
from matching import matching_design, MC_match, match_requirement_dataset_improved

class pull_page:
    """A class to read the saved csv data from the wood intake process"""

    def __init__(self, Digital_intake_class):
        DigIn = Digital_intake_class 
        self.matching_list = None
        self.wood_list = None
        self.requirement_list = None
        self.fields = DigIn.fields

        st.title("Get design from Grasshopper and match with available wood")

        st.subheader("Example how it works")

        st.text("A design is made in Grasshopper")
        # show an image of the overview
        alternate_design_image = Image.open('alternate_design_image.png')

        st.image(alternate_design_image, caption='Design in Grasshopper image', width = 300)

        st.text('This design can be matched with the available wood and reserved')

        st.subheader('Get a design of a project saved in the database')

        project_id = st.text_input('What is the project id?', 'Stool_1')
        if st.button('Pull requirements through API'):
            st.write('Requirements are taken through an API call from Grasshopper')
            self.requirement_list = DigIn.read_requirements_from_client(project_id = project_id)
            requirement_df = pd.DataFrame(self.requirement_list)
            if len(self.requirement_list) > 0:
                st.text('These are the requirements in a table:')
                st.write(requirement_df)
            else:
                st.text('There are no requirements currently available')



        tab1, tab2, tab3 = st.tabs(["Show design", "Match design", "Reserve matched wood"])

        with tab1:
            # if os.path.exists('requirements.csv'):
                # requirement_df = pd.read_csv('requirements.csv')
                # length_values_req = requirement_df['Length']
            if 'requirement_df' in locals() and len(self.requirement_list) > 0:
                st.subheader('Length and width distribution in mm of the requirements\n')
                fig = GraphicalElements.barchart_plotly_one(requirement_df, color='red', requirements=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.text('There are no requirements to show')

        with tab2:
            wood_list = DigIn.get_data_api()
            requirement_list = DigIn.read_requirements_from_client(project_id = project_id)
            # print(requirement_list)
            # print(wood_list)
            self.matching_df, unmatched_df, optimal_list = matching_design(requirement_list, wood_list)
            # if type(self.matching_df) == list:
            #     self.matching_df.to_csv(pd.DataFrame({}))
            # else:
            try:
                self.matching_df.to_csv('matching_df.csv', index=False)
            except:
                ...
        
        with tab3:
            matching_df = pd.read_csv('matching_df.csv')
        # st.subheader("Match the requirements with the available wood")
            reservation_design(DigIn, matching_df)
        


    
    # def simple_match(self, DigIn):
    #     if os.path.exists('requirements.csv'):
    #         requirement_df = pd.read_csv('requirements.csv')
    #         requirement_list = requirement_df.to_dict('records')

    #         # st.write(requirement_df)
    #         # print(requirement_list)
    #         matching_df, unmatched_df = DigIn.match_requirements_dataset(requirement_list)
    #         dataset = pd.DataFrame(DigIn.wood_list)

    #         # Visualize the matched planks
    #         if len(matching_df):
    #             st.write(matching_df)
    #             st.write("Matching Dataframe is saved in a CSV file")
    #             matching_df.to_csv('matching_df.csv', index=False)
    #             st.subheader('Showing all the planks in the dataset and the matching requirements')
    #             fig = GraphicalElements(dataset).barchart_plotly_two_improved(matching_df, requirement_df)
    #             # fig = Graphical_elements(dataset).barchart_plotly_two(matching_df, requirement_df)
    #             st.plotly_chart(fig, use_container_width=True)
    #         else:
    #             st.write('Could not match any unreserved planks with the requirements')

    #         if len(unmatched_df):
    #             st.subheader(
    #                 'The following chart shows all the required planks that cannot be found in the database'
    #                 ' based on the used matching algorithm')
    #             fig = GraphicalElements(dataset).barchart_plotly_one(dataset=unmatched_df,
    #                                                                  color='maroon', requirements=True)
    #             st.plotly_chart(fig, use_container_width=True)

    #     else:
    #         st.write('Requirements are not found')
