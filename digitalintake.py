import os
import pandas as pd
import random
import numpy as np
import streamlit as st
from datetime import datetime

import requests
from dotenv import load_dotenv
import json

from database_page import database_page
from pull_page import pull_page
from push_page import push_page
from form_finding import form_finding_page

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
            "Original piece",
            "Production phase",
            "Design",
            "Kind of wood",
            "Verified measurements",
            # "Requirements",
            "Source",
            "Price",
            "Info"

        ]

        self.data = self.get_data_api()

    def get_data_api(self):

        # api-endpoint
        url = os.environ.get("DATABASE_URL") + "/residual_wood"
        # sending get request and saving the response as response object
        r = requests.get(url=url)

        # extracting data in json format
        data = r.json()
        dataset = pd.DataFrame(data)

        # convert to right type
        dataset = dataset.astype({"width": "int", "length": "int", "height": "int",
                                  "density": "int", "weight": "int", "price": "float"})

        # get the right order of columns to use
        dataset.columns = dataset.columns.str.title()
        dataset.rename(columns={'Id': 'Index', 'Reservation_Time': 'Reservation time',
                                'Reservation_Name': 'Reservation name'}, inplace=True)
        st.write(dataset)
        dataset.rename(columns={"Original_piece": "Original piece",
                                "Production_phase": "Production phase",
                                "Kind_of_wood": "Kind of wood",
                                "Verified_measurements": "Verified measurements"}, inplace=True)

        try:
            dataset = dataset[self.fields]
        except KeyError as err:
            print(err)

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
            _type = random.choice(['Soft Wood', 'Hard Wood'])
            color = np.random.randint(100, 200, size=3)
            indexed = datetime.now().strftime("%Y-%m-%d %H:%M")
            density = random.randint(min_density, max_density)
            weight = int(width * length * height * density / 10000 / 1000)

            row = {'Index': index, 'Width': width, 'Length': length, 'Height': height,
                   'Type': _type, 'Color': color, 'Timestamp': indexed, 'Density': density, 'Weight': weight,
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
            for n in range(leg_parts[0]):
                part = "Leg"
                length, width, height = leg_parts[1]['length'], leg_parts[1]['width'], leg_parts[1]['height']
                row = self.make_json_requirement(index, width, length, height, part, tag, project_id)

                # insert into db via post call
                self.post_call_requirement(row)

                requirement_list.append(row)
                index += 1

            for n in range(top_parts[0]):
                part = "Top"
                length, width, height = top_parts[1]['length'], top_parts[1]['width'], top_parts[1]['height']
                row = self.make_json_requirement(index, width, length, height, part, tag, project_id)

                # insert into db via post call
                self.post_call_requirement(row)

                requirement_list.append(row)
                index += 1

            for n in range(side_parts[0]):
                part = "Side"
                length, width, height = side_parts[1]['length'], side_parts[1]['width'], side_parts[1]['height']
                row = self.make_json_requirement(index, width, length, height, part, tag, project_id)

                # insert into db via post call
                self.post_call_requirement(row)

                requirement_list.append(row)
                index += 1

        self.requirement_list = requirement_list

        return requirement_list

    def make_json_requirement(self, index, width, length, height, part, tag, project_id):
        row = {
            "part_index": index,
            "width": width,
            "length": length,
            "height": height,
            "part": part,
            "tag": tag,
            "project_id": project_id
        }
        return row

    @staticmethod
    def post_call_requirement(row):

        url = os.environ.get("DATABASE_URL")

        print(row)
        json_row = json.dumps(row)
        print(json.dumps(row))
        # for key, value in row.items():
        #     print(key, value)
        #     print(type(key), type(value))
        # print(type(row))
        # try:
        response = requests.post(
            url=url + "/requirements/client",
            data=json_row,
            headers={"Content-Type": "application/json"}
        )
        # print(response.content)
        print(response)

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
                    "part_index": index,
                    "width": width,
                    "length": length,
                    "height": height,
                    "part": part,
                    "tag": tag,
                    "project_id": project_id
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

    def read_requirements_from_client(self, project_id="1"):
        """Get the requirements from client e.g. Grasshopper or Dashboard"""
        # Get the requirements that share the same project ID from the API
        db_url = os.environ.get("DATABASE_URL")

        # something to configure
        # if project_id == 'All':
        requirements_endpoint = db_url + "/requirements/client"
        # else:
        # requirements_endpoint = db_url + "/requirements/client" + '/' + str(project_id)

        response = requests.get(url=requirements_endpoint, headers={"Content-Type": "application/json"})
        if response.status_code != 200:
            raise Exception(
                f"something went wrong fetching data for requirements status code: {response.status_code}"
            )
        requirement_list = response.json()
        self.requirement_list = requirement_list

        # # Get the woods from the API
        # woods_endpoint = db_url + "/residual_woods"
        # resp = requests.get(url=woods_endpoint, headers={"Content-Type": "application/json"})
        # if resp.status_code != 200:
        #     raise Exception(f"something went wrong fetching data for residual wood status code: {resp.status_code}")
        # wood_list = resp.json()

        return requirement_list

    def delete_requirements_from_client(self, part_index=1000, delete_all=False):

        db_url = os.environ.get("DATABASE_URL")

        requirements_endpoint = db_url + "/requirements/client"
        delete_endpoint = db_url + "/requirement/client"
        response = None
        if delete_all:
            response = requests.get(url=requirements_endpoint, headers={"Content-Type": "application/json"})
            print(pd.DataFrame(response.json()))
            print(pd.DataFrame(response.json())['id'])
            for index in pd.DataFrame(response.json())['id']:
                response = requests.delete(url=delete_endpoint + '/' + str(index),
                                           headers={"Content-Type": "application/json"})
                print(response.json())
        else:

            requests.delete(url=delete_endpoint + '/' + str(part_index), headers={"Content-Type": "application/json"})
        if response.status_code != 200:
            raise Exception(
                f"something went wrong fetching data for requirements status code: {response.status_code}"
            )


def main():
    # Initialize the DigitalIntake
    DigIn = DigitalIntake()

    # Adding a sidebar
    sidebar = st.sidebar.radio('Pages', (
        "Database",
        "Form fitting - free design",
        "Form fitting - set design",
        "Form finding"))

    if sidebar == 'Database':
        database_page(DigIn)
        DigIn.wood_list = DigIn.get_data_api()

    elif sidebar == 'Form fitting - free design':
        pull_page(DigIn)

    elif sidebar == 'Form fitting - set design':
        push_page(DigIn)

    elif sidebar == "Form finding":
        form_finding_page(DigIn)


if __name__ == "__main__":
    main()
