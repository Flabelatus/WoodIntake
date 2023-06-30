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

from reservation_system import unreserve

class database_page:
    """A class to read the saved csv data from the wood intake process"""

    def __init__(self, Digital_intake_class):
        DigIn = Digital_intake_class 
        self.matching_list = None
        self.wood_list = None
        self.requirement_list = None
        self.fields = DigIn.fields
        self.data = DigIn.get_data_api()

        # Fetch data from the database
        DigIn.wood_list = DigIn.get_data_api()

        # Setup Streamlit on Local URL: http://localhost:8501
        st.title("Available Wood Table")
        st.text("The following table demonstrates real-time data captured by\n"
                "the digital intake process in the Robot Lab as means of building\n"
                "a data base of residual wood.")

        
        # show an image of the overview
        database_image = Image.open('database_image.png')

        st.image(database_image, caption='Database image')


        st.subheader("Digital Intake Results from the Robot Lab")
        dataset = DigIn.data
        # style = 'API'

        

        if st.button('Refresh data'):
            dataset = DigIn.data

        st.write(pd.DataFrame(DigIn.wood_list))
        st.write(f'TOTAL Number of wood scanned: {len(dataset["Index"])}')
        st.download_button('Download Table', dataset.to_csv(), mime='text/csv')

        if st.button('Unreserve all wood'):
            unreserve(DigIn, unreserve_all = True)

        

        tab1, tab2 = st.tabs(["Database", "Requirements"])

        with tab1:
            st.subheader('Length and width distribution in mm of the dataset\n')
            fig = self.barchart_plotly_one(
                dataset.sort_values(by=['Length'], ascending=False), color='blue')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            project_id = st.text_input('Project id of requirement', 'All')
            # if project_id == 'All'
            requirement_list = DigIn.read_requirements_from_client(project_id)
                # length_values_req = requirement_df['Length']
            st.subheader('Length and width distribution in mm of the requirements\n')
            requirement_df = pd.DataFrame(requirement_list)
            if len(requirement_list) > 0:
                fig = self.barchart_plotly_one(requirement_df, color='red', requirements=True)
                st.plotly_chart(fig, use_container_width=True)
                if st.button('Delete all requirements through API'):
                    DigIn.delete_requirements_from_client(delete_all = True)
            else:
                st.text('There are no requirements currently available')

        # with tab3:
        #    st.header("An owl")
        #    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


        # st.subheader('Length Distribution in mm of the dataset in Plotly')
        # fig = self.distplot_plotly(dataset = dataset, x_column="Length", y_column="Width",
        #                                                  color="Type")
        # st.plotly_chart(fig, use_container_width=True)

        # self.show_color(dataset, style = 'API')


    def barchart_plotly_one(self, dataset, color, requirements='False'):

        if requirements is True:
            fig = go.Figure(data=[go.Bar(
                x=(dataset['width'].cumsum() - dataset['width'] / 2).tolist(),
                y=dataset['length'],
                width=(dataset['width']).tolist(),  # customize width here
                marker_color=color,
                opacity=0.8,
                customdata=dataset['part'].tolist(),
                name='requirement',
                hovertemplate='Width (mm): %{width:.f}, Length (mm): %{y:.f}, Part: %{customdata:.s}'
            )])
        else:
            fig = go.Figure(data=[go.Bar(
                x=(dataset['Width'].cumsum() - dataset['Width'] / 2).tolist(),
                y=dataset['Length'],
                width=(dataset['Width']).tolist(),  # customize width here
                marker_color=color,
                opacity=0.8,
                name='wood in database',
                hovertemplate='Width (mm): %{width:.f}, Length (mm): %{y:.f}'
            )])

        return fig


    def distplot_plotly(self, dataset, x_column, y_column, color):

        fig = px.histogram(dataset, x=x_column, color=color,
                           marginal="box",  # or violin, rug
                           hover_data=dataset.columns)

        return fig


    def show_color(self, dataset, style):

        colors = dataset['Color']
        rgb_column = [row.split(',') for row in list(colors)]
        rgb = []
        for rgb_list in rgb_column:
            rgb.append(tuple([int(value) for value in rgb_list]))

        img = []
        if style == 'csv':
            for index, color in enumerate(colors):
                img.append((
                    Image.new('RGB', (100, 200), tuple(colors[index])),
                    str(dataset["Index"][index]))
                )
        elif style == 'API':
            for index, colors in enumerate(rgb):
                img.append((
                    Image.new('RGB', (100, 200), rgb[index]),
                    str(dataset["Index"][index]))
                )

        st.subheader('Available Colors in the Database')
        # string
        input_text = st.text_input('Please enter the Index'
                                   ' of the desired element to '
                                   'view te color')

        st.image([
            img[index][0] for index in range(len(img))
            if img[index][1] == input_text
        ],
            caption=[
                img[index][1] for index in range(len(img))
                if img[index][1] == input_text
            ],
            use_column_width=False,
            width=100
        )

    
