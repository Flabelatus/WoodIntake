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
from dotenv import load_dotenv

from database_page import database_page
from pull_page import pull_page
from push_page import push_page

load_dotenv()

class DigitalIntake:
    """A class to read the saved csv data from the wood intake process"""

    def __init__(self, name='Datawood.csv'):
        self.matching_list = None
        self.wood_list = None
        self.requirement_list = None
        self.name = name
        self.fields = [
            "Index",
            "Width",
            "Length",
            "Height",
            "Type",
            "Color",
            "Timestamp",
            "Density",
            "Weight",
            'Reserved',
            "Reservation name",
            "Reservation time",
            "Requirements",
            "Source",
            "Price",
            "Info"

        ]

        self.data = self.get_data_api()

    def get_data_api(self):

        # api-endpoint
        URL = os.environ.get("DATABASE_URL") + "/residual_wood"
        # sending get request and saving the response as response object
        r = requests.get(url=URL)

        # extracting data in json format
        data = r.json()
        dataset = pd.DataFrame(data)

        # convert to right type
        dataset = dataset.astype({"width": "int", "length": "int", "height": "int",
                                  "density": "int", "weight": "int", "price": "int"})
        # get the right order of columns to use
        dataset.columns = dataset.columns.str.title()
        dataset.rename(columns={'Id': 'Index', 'Reservation_Time': 'Reservation time',
                                'Reservation_Name': 'Reservation name'}, inplace=True)

        dataset = dataset[self.fields]
        self.data = dataset

        wood_list = self.data.to_dict('records')
        self.wood_list = wood_list
        return wood_list

    def generate_new_wood(self, n=10, min_width=100, max_width=300,
                          min_length=300, max_length=1000,
                          min_height=30, max_height=50,
                          min_density=200, max_density=700):
        wood_list = []
        for index in range(n):
            width = random.randint(min_width, max_width)
            length = random.randint(min_length, max_length)
            height = random.randint(min_height, max_height)
            Type = random.choice(['Soft Wood', 'Hard Wood'])
            Color = np.random.randint(100, 200, size=3)
            Indexed = datetime.now().strftime("%Y-%m-%d %H:%M")
            Density = random.randint(min_density, max_density)
            Weight = int(width * length * height * Density / 10000 / 1000)

            row = {'Index': index, 'Width': width, 'Length': length, 'Height': height,
                   'Type': Type, 'Color': Color, 'Timestamp': Indexed, 'Density': Density, 'Weight': Weight,
                   'Reserved': False, 'Reservation name': '', 'Reservation time': '', "Requirements": 0,
                   "Source": "Robot Lab", "Price": 5, "Info": ""}
            wood_list.append(row)

        self.wood_list = wood_list
        return wood_list

    def generate_requirements_stool_api(
            self,
            n_stools=1,
            top_parts=(5, {'length': 360, 'width': 70, 'height': 18}),
            side_parts=(4, {'length': 355, 'width': 61, 'height': 18}),
            leg_parts=(4, {'length': 530, 'width': 90, 'height': 25}),
            project_id="1",
            tag="stool",
    ):
        """Simplified function the Same as generate_requirements_stool method. However, with this addition
         that this function stores the new requirements in the db"""

        requirement_list = []
        index = 1

        for stool in range(n_stools):
            for n in range(leg_parts):
                length, width, height = leg_parts[1]['length'], leg_parts[1]['width'], leg_parts[1]['height']
                row = {
                    'part_index': index,
                    'width': width,
                    'length': length,
                    'height': height,
                    'part': 'Leg',
                    'project_id': project_id,
                    'tag': tag
                }
                # insert into db via post call
                response = requests.post(
                    url=os.environ.get("DATABASE_URL") + "/requirements/client",
                    data=row,
                    headers={
                        "Content-Type": "application/json"
                    }
                )
                if response.status_code != 201:
                    raise Exception(
                        "something went wrong inserting row in the db status code: {0}".format(response.status_code)
                    )
                requirement_list.append(row)
                index += 1

            for n in range(top_parts):
                length, width, height = top_parts[1]['length'], top_parts[1]['width'], top_parts[1]['height']
                row = {
                    'part_index': index,
                    'width': width,
                    'length': length,
                    'height': height,
                    'part': 'Leg',
                    'project_id': project_id,
                    'tag': tag
                }
                # insert into db via post call
                response = requests.post(
                    url=os.environ.get("DATABASE_URL") + "/requirements/client",
                    data=row,
                    headers={
                        "Content-Type": "application/json"
                    }
                )
                if response.status_code != 201:
                    raise Exception(
                        "something went wrong inserting row in the db status code: {0}".format(response.status_code)
                    )
                requirement_list.append(row)
                index += 1

            for n in range(side_parts):
                length, width, height = side_parts[1]['length'], side_parts[1]['width'], side_parts[1]['height']
                row = {
                    'part_index': index,
                    'width': width,
                    'length': length,
                    'height': height,
                    'part': 'Leg',
                    'project_id': project_id,
                    'tag': tag
                }
                # insert into db via post call
                response = requests.post(
                    url=os.environ.get("DATABASE_URL") + "/requirements/client",
                    data=row,
                    headers={
                        "Content-Type": "application/json"
                    }
                )
                if response.status_code != 201:
                    raise Exception(
                        "something went wrong inserting row in the db status code: {0}".format(response.status_code)
                    )
                requirement_list.append(row)
                index += 1

        self.requirement_list = requirement_list
        return requirement_list

    def generate_requirements_stool(self, n_stools=1):
        """
        This function generates a standard stool generated by Javid in Grasshopper. The dimensions are set. 
        n_stools: the number of stools to generate.
        """

        requirement_list = []

        index = 0
        top_parts = 5
        side_parts = 4
        leg_parts = 4
        for stool in range(n_stools):
            for n in range(leg_parts):
                length, width, height = 530, 90, 25
                row = {'Index': index, 'Width': width, 'Length': length, 'Height': height, 'Part': 'Leg'}
                requirement_list.append(row)
                index += 1

            for n in range(top_parts):
                length, width, height = 360, 70, 18
                row = {'Index': index, 'Width': width, 'Length': length, 'Height': height, 'Part': 'Top'}
                requirement_list.append(row)
                index += 1

            for n in range(side_parts):
                length, width, height = 355, 61, 18
                row = {'Index': index, 'Width': width, 'Length': length, 'Height': height, 'Part': 'Side'}
                requirement_list.append(row)
                index += 1

        self.requirement_list = requirement_list
        return requirement_list

    def generate_requirements_api(
            self,
            size=1,
            part="-",
            tag="stool",
            project_id="1",
            n_planks=20,
            complexity=5,
            width=(100, 300),
            length=(300, 1000),
            height=(30, 50)
    ):
        """Same as generate_requirements method, however, stores it in the db
               schema = {
            "part_index": "1",
            "length": "530.00",
            "width": "90.00",
            "height": "25.00",
            "part": "leg",
            "tag": "chair",
            "project_id": "1"
        }
        part index is a string value referring to the index of the part in the design (not the id of the
        element in the db)
        """

        requirement_list = []

        min_width, max_width = width
        min_length, max_length = length
        min_height, max_height = height
        max_size = 5
        index = 1

        db_url = os.environ.get("DATABASE_URL")

        for plank in range(int(n_planks / complexity)):
            width = int(random.randint(min_width, max_width) / (max_size - size))
            length = int(random.randint(min_length, max_length) / (max_size - size))
            height = min_height

            #         height = random.randint(min_height, max_height)

            for _ in range(complexity):
                index += 1
                row = {
                    'part_index': index,
                    'width': width,
                    'length': length,
                    'height': height,
                    'part': part,
                    'tag': tag,
                    'project_id': project_id
                }
                requirement_list.append(row)
                # add them in the db
                response = requests.post(url=db_url + "/requirements/client", data=row, headers={
                    "Content-Type": "application/json"
                })
                if response.status_code != 201:
                    raise Exception(
                        "something went wrong inserting row in db status code: {0}".format(response.status_code)
                    )

        self.requirement_list = requirement_list
        return requirement_list

    def generate_requirements(self, size=1, n_planks=20, complexity=5,
                              width=[100, 300], length=[300, 1000], height=[30, 50]):
        """
        This function uses size and n_planks as inputs to generate requirements.
        
        Size: the size input goes from 1 to 5, for size 5 it will generate longer lengths and widths 
        N_planks: the number of planks that is needed for this project
        Complexity: the number of differing lengths and widths that is needed for this project
        """

        requirement_list = []

        min_width, max_width = width
        min_length, max_length = length
        min_height, max_height = height
        max_size = 5
        index = 0

        for plank in range(int(n_planks / complexity)):
            width = int(random.randint(min_width, max_width) / (max_size - size))
            length = int(random.randint(min_length, max_length) / (max_size - size))
            height = min_height

            #         height = random.randint(min_height, max_height)

            for _ in range(complexity):
                index += 1
                row = {'Index': index, 'Width': width, 'Length': length, 'Height': height, 'Part': 'Unknown'}
                requirement_list.append(row)
        self.requirement_list = requirement_list
        return requirement_list

    # def read_requirements_from_client(self):
    #     """Get the requirements from client e.g. Grasshopper or Dashboard"""
    #     # Get the requirements that share the same project ID from the API
    #     db_url = os.environ.get("DATABASE_URL")
    #
    #     requirements_endpoint = db_url + "/requirements/client"
    #     response = requests.get(url=requirements_endpoint, headers={"Content-Type": "application/json"})
    #     if response.status_code != 200:
    #         raise Exception(
    #         f"something went wrong fetching data for requirements status code: {response.status_code}"
    #         )
    #     requirements_list = response.json()
    #
    #     # Get the woods from the API
    #     woods_endpoint = db_url + "/residual_woods"
    #     resp = requests.get(url=woods_endpoint, headers={"Content-Type": "application/json"})
    #     if resp.status_code != 200:
    #         raise Exception(f"something went wrong fetching data for residual wood status code: {resp.status_code}")
    #     wood_list = resp.json()

    def match_requirements_dataset(self, requirement_list):
        """This function matches the requirements from the generated requirements and selects fitting  planks
        from the available dataset."""

        wood_list = self.wood_list.copy()
        matching_list = []
        unmatched_list = []
        for i, row_req in enumerate(requirement_list):
            for index, row_wood in enumerate(wood_list):
                if (row_req['Width'] < row_wood['Width'] and row_req['Length'] < row_wood['Length']
                        and not row_wood['Reserved']):
                    matching_row = {'Requirements list index': row_req['Index'], 'Width req': row_req['Width'],
                                    'Length req': row_req['Length'], 'Height req': row_req['Height'],
                                    'Wood list index': row_wood['Index'], 'Width DB': row_wood['Width'],
                                    'Length DB': row_wood['Length'], 'Height DB': row_wood['Height']}

                    wood_list[index]['Reserved'] = True
                    matching_list.append(matching_row)
                    break

        self.matching_list = matching_list
        matching_df = pd.DataFrame(matching_list)
        if len(matching_list):
            for index, row_req in enumerate(requirement_list):
                # if matching_df['Index'].isin(index):
                if index not in matching_df['Requirements list index'].values:
                    unmatched_list.append(row_req)
        unmatched_df = pd.DataFrame(unmatched_list)

        return matching_df, unmatched_df

    def match_requirement_dataset_improved(self, requirement_list, n_runs=30, option="Minimum waste"):

        cuts_list = []
        wood_cuts_list = []
        stock_pieces_list = []
        matching_list_all = []
        success_list = []
        waste_list = []
        unmatched_list = []

        for run in range(n_runs):
            stock_pieces = pd.DataFrame(self.wood_list).copy()
            parts = pd.DataFrame(requirement_list).copy()
            # parts = parts_df.copy()
            # stock_pieces = stock_pieces_start_df.copy()

            # shuffle

            if option == 'Keep long planks':
                stock_pieces = stock_pieces.sort_values(by='Length', ascending=True).reset_index(drop=True)
                parts = parts.sample(frac=1)
            else:
                stock_pieces = stock_pieces.sample(frac=1).reset_index(drop=True)
            wood_cuts = list([0] * len(stock_pieces))

            matching_list = []
            cuts = 0
            success = 0
            waste = []
            # print(run, 'run')
            for i, part in parts.iterrows():
                # print(i, 'part')

                for j, stock in stock_pieces.iterrows():
                    # print(j, ' stock')

                    if part['Length'] <= stock['Length'] and part['Width'] <= stock['Width']:  # and
                        # part['Height'] <= stock['Height']):
                        #                 matching_list.append((part, stock))
                        # print([stock['Index']])
                        wood_cuts[stock['Index']] += 1
                        stock_pieces.loc[j, 'Length'] = stock['Length'] - part['Length']
                        stock_pieces.loc[j, 'Width'] = stock['Width']
                        stock_pieces.loc[j, 'Height'] = stock['Height']

                        cuts += 1
                        success += 1
                        waste.append((stock['Width'] - part['Width']) * (stock['Height'] - part['Height']))

                        matching_row = {'Requirements list index': part['Index'], 'Wood list index': stock['Index']}
                        matching_list.append(matching_row)
                        break

            success_list.append(success == len(parts))
            cuts_list.append(cuts)
            waste_list.append(waste)
            wood_cuts_list.append(wood_cuts)
            stock_pieces_list.append(stock_pieces)
            matching_list_all.append(matching_list)

        if option == 'Minimum waste':
            index_min = waste_list.index(min(waste_list))
        elif option == 'Keep long planks':
            index_min = waste_list.index(min(waste_list))
        elif option == 'Most parts found in database':
            index_min = success_list.index(max(success_list))
        elif option == 'Minimum cuts needed':
            index_min = wood_cuts_list.index(min(wood_cuts_list))

        matching_df = pd.DataFrame(matching_list_all[index_min])

        if len(matching_list):
            for index, row_req in enumerate(requirement_list):
                # if matching_df['Index'].isin(index):
                if index not in matching_df['Requirements list index'].values:
                    unmatched_list.append(row_req)
        unmatched_df = pd.DataFrame(unmatched_list)

        return matching_df, unmatched_df

    def match_euc_dis(self, requirement_list):
        """This function matches the requirements from the generated requirements based on the Euclidean Distance
        and selects fitting planks from the available dataset."""

        wood_list = self.wood_list.copy()
        matching_list = []
        unmatched_list = []
        # k = len(requirement_list)
        for index, row_req in enumerate(requirement_list):
            # distances = []
            wood_db = []
            wood_db_index = []

            for index, row_wood in enumerate(wood_list):
                if (
                        # row_req['Width'] < row_wood['Width'] and row_req['Length'] < row_wood['Length'] and
                        not row_wood['Reserved']):
                    wood_db.append(np.array([row_wood['Width'], row_wood['Length'], row_wood['Height']]))
                    wood_db_index.append(row_wood['Index'])

            wood_piece = np.array([row_req['Width'], row_req['Length'], row_req['Height']])
            if wood_db:
                distances = (np.sqrt(np.sum((wood_db - wood_piece) ** 2, axis=1)))

                # Find the indices of the k nearest neighbors in the database
                # k_nearest = np.argsort(distances)[:k]

                # Take the minimum of the k nearest neighbors to determine the best match
                best_match_index = np.argmin(distances)

                # Add the matched piece to the matched_indices list
                index = wood_db_index[best_match_index]

                #             print('wood piece: ' + str(wood_piece) + 'wood_db: ' + str(wood_list[index]))

                matching_row = {'Requirements list index': row_req['Index'], 'Wood list index': index}
                wood_list[index]['Reserved'] = True
                matching_list.append(matching_row)

        matching_df = pd.DataFrame(matching_list)
        if len(matching_df):
            for index, row_req in enumerate(requirement_list):
                # if matching_df['Index'].isin(index):
                if index not in matching_df['Requirements list index'].values:
                    unmatched_list.append(row_req)
        unmatched_df = pd.DataFrame(unmatched_list)

        return matching_df, unmatched_df





def main():
    # Initialize the DigitalIntake
    DigIn = DigitalIntake()
    dataset = DigIn.data

    ## Adding a sidebar
    sidebar = st.sidebar.radio('Which page?',("Database", "Pull", "Push"))

    if sidebar == 'Database':
        DB_page = database_page(DigIn)
        DigIn.wood_list = DB_page.get_data_api()

    elif sidebar == 'Pull':
        object_based_page = pull_page(DigIn)

    elif sidebar == 'Push':
        design_based_page = push_page(DigIn)
    


if __name__ == "__main__":
    main()
