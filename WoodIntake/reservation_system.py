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
import json


    

def reservation_design(DigIn, matching_df):

    st.subheader('Reserve the wood based on matched requirements')
    res_name = st.text_input('Reservation project id', 'Stool_1')
    # res_number = st.text_input('Reservation id', '1')
    st.write('The reservation will be on ', res_name)
    
    print(matching_df)
    if st.button('Reserve the wood'):
        if 'matching_df' in locals():
            matching_list = matching_df.to_dict('records')

            reserve(DigIn, matching_list, res_name)
            st.write('The wood is reserved')
        else:
            print()
            st.write('It was not possible to reserve, due to an error')

def reserve(DigIn, matching_list, res_name):
    
    db_url = os.environ.get("DATABASE_URL")
    
    wood_endpoint = db_url + "/residual_wood"

    # matching_df = pd.read_csv('matching_df.csv')
    # matching_list = matching_df.to_dict('records')

    for index, row in enumerate(matching_list):

        # DigIn.wood_list[row['Wood list index']]['Reservation name'] = str(self.res_name + '-' + self.res_number)
        # DigIn.wood_list[row['Wood list index']]['Reserved'] = True
        # DigIn.wood_list[row['Wood list index']]['Reservation time'] = datetime.now().strftime("%Y-%m-%d %H:%M")

        response = requests.get(url=wood_endpoint + '/' + str(row['Wood list index']), 
            headers={"Content-Type": "application/json"})
        wood_row = response.json()
        print(wood_row)
        wood_row['reserved'] = True
        wood_row['reservation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        wood_row['reservation_name'] = str(res_name)

        response = requests.put(url=wood_endpoint + '/'+ str(wood_row.pop('id')), 
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(wood_row))
    
    
    # pd.DataFrame(DigIn.wood_list).to_csv('Generated_wood_data.csv', index=False)

def unreserve(DigIn, unreserve_all = True):
    
    wood_list = DigIn.get_data_api()
    dataset = pd.DataFrame(wood_list)

    # dataset['Reserved'] = False
    # dataset['Reservation name'] = ''
    # dataset['Reservation time'] = ''

    wood_list = dataset.to_dict('records')

    # unres_name = st.text_input('Unreservation name', 'Stool_1')
    # unres_number = st.text_input('Unreservation number', '1')


    db_url = os.environ.get("DATABASE_URL")
    
    wood_endpoint = db_url + "/residual_wood"
    

    # if st.button('Unreserve all wood'):
    response = requests.get(url=wood_endpoint, headers={"Content-Type": "application/json"})
    # wood_df = pd.DataFrame(response.json())
    # print(pd.DataFrame(response.json())['id'])
    if unreserve_all == True:
        for row in response.json():
            if row['type'] is None:
                row['type'] = "unknown"
            print(row)
            row['reserved'] = False
            row['reservation_time'] = ''
            row['reservation_name'] = ''
            # print(json.dumps(row))
            response = requests.put(url=wood_endpoint + '/'+ str(row.pop('id')), headers={"Content-Type": "application/json"},
                data=json.dumps(row))



    # st.write('The reservation on ', unres_name + '-' + unres_number, 'you want to unreserve')
    # if st.button('Unreserve the items on this project id'):
    #     dataset = pd.DataFrame(DigIn.wood_list)

    #     dataset.loc[dataset['Reservation name'] == str(
    #         unres_name + '-' + unres_number), 'Reserved'] = False
    #     dataset.loc[dataset['Reservation name'] == str(
    #         unres_name + '-' + unres_number), 'Reservation time'] = None
    #     dataset.loc[dataset['Reservation name'] == str(
    #         unres_name + '-' + unres_number), 'Reservation name'] = None

    #     dataset.to_csv('Generated_wood_data.csv', index=False)
    #     DigIn.wood_list = dataset.to_dict('records')

