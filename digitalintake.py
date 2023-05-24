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


class GraphicalElements:

    def __init__(self, dataset):
        self.dataset = dataset

    def distplot_plotly(self, x_column, y_column, color):

        fig = px.histogram(self.dataset, x=x_column, color=color,
                           marginal="box",  # or violin, rug
                           hover_data=self.dataset.columns)

        return fig

    @staticmethod
    def barchart_plotly_one(dataset, color, requirements='False'):

        if requirements is True:
            fig = go.Figure(data=[go.Bar(
                x=(dataset['Width'].cumsum() - dataset['Width'] / 2).tolist(),
                y=dataset['Length'],
                width=(dataset['Width']).tolist(),  # customize width here
                marker_color=color,
                opacity=0.8,
                customdata=dataset['Part'].tolist(),
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

    def barchart_plotly_two(self, matching_df, requirement_df):
        """
        This function makes a barchart in which the length and with of the planks are visualized for both
        the required planks as the wood in the database.
        """

        wood_df = self.dataset.copy()
        # join the two, keeping all of df1's indices
        joined = pd.merge(matching_df, wood_df.loc[:, ['Index', 'Width', 'Length', 'Height']],
                          left_on='Wood list index', right_on='Index', how='inner')
        joined = pd.merge(joined, requirement_df, left_on=['Requirements list index'], right_on=['Index'], how='inner')

        # Make a trace with a bar chart based on a cumsum to make sure all planks are next to each other. 
        # Furthermore, the cumsum of the requirements is added, as these planks need to be in between the planks
        # of the database. Lastly, the x starts before a first requirement plank is added, and it is
        # lined up in the middle.
        trace1 = go.Figure(data=[go.Bar(
            x=((joined['Width_x'].cumsum() + joined['Width_y'].cumsum() - joined['Width_y'] - joined[
                'Width_x'] / 2)).tolist(),
            y=joined['Length_x'],
            width=(joined['Width_x']).tolist(),  # customize width here
            marker_color='blue',
            opacity=0.8,
            name='wood in database',
            hovertemplate='Width (mm): %{width:.f}, Length (mm): %{y:.f}'
        )])

        trace2 = go.Figure(data=[go.Bar(
            x=(joined['Width_x'].cumsum() + joined['Width_y'].cumsum() - joined['Width_y'] / 2).tolist(),
            y=joined['Length_y'],
            width=(joined['Width_y']).tolist(),  # customize width here
            marker_color='red',
            customdata=joined['Part'].tolist(),
            opacity=0.8,
            name='requirement',
            hovertemplate='Width (mm): %{width:.f}, Length (mm): %{y:.f}, Part: %{customdata:.s}'

        )])

        fig = go.Figure(data=trace1.data + trace2.data)

        return fig

    def barchart_plotly_two_improved(self, matching_df, requirement_df):

        wood_df = self.dataset.copy()

        # joining together all the data from the requirements dataframe and from the wood in stock and merge those
        joined = pd.merge(matching_df, wood_df.loc[:, ['Index', 'Width', 'Length', 'Height']],
                          left_on='Wood list index', right_on='Index', how='inner')
        joined = pd.merge(joined, requirement_df, left_on=['Requirements list index'], right_on=['Index'], how='inner')

        # check where the required pieces come from the same piece of wood
        dupl_joined = joined.drop_duplicates()['Wood list index'].value_counts().gt(1)
        joined.loc[joined['Wood list index'].isin(dupl_joined[dupl_joined].index)]

        # Giving all the pieces a width cumsum, so that they can be visualised next to each other. 
        # And forward fill to fill gaps where needed.
        joined_2 = joined.drop_duplicates(['Wood list index']).copy()
        joined_2['Width_x_cumsum'] = joined_2['Width_x'].cumsum()
        joined['Width_x_cumsum'] = joined_2['Width_x'].cumsum()
        joined = joined.ffill(axis=0)

        # Getting the variable width pieces sorted to get the widest planks visualised at the bottom, but the narrowest
        # pieces visualised first. This will give the best graphic output. Also get the cumsum of length y to visualise
        # the pieces above each other.
        joined_3 = joined.loc[joined['Wood list index'].isin(dupl_joined[dupl_joined].index)].copy()
        df_differing_required_pieces = self.get_diff_size_piece_visual_dataframe(all_wood_df=joined_3)
        joined_5 = df_differing_required_pieces.copy()

        trace1 = go.Figure(data=[go.Bar(
            x=np.ceil((joined_2['Width_x_cumsum'] - joined_2['Width_x'] / 2)).tolist(),
            y=joined_2['Length_x'],
            width=(joined_2['Width_x']).tolist(),  # customize width here
            marker_color='blue',
            opacity=0.4,
            name='Wood in database',
            xaxis='x',
            hovertemplate='Width: %{width:.f}, Length: %{y:.f}',
        )])

        trace2 = go.Figure(data=[go.Bar(
            x=(joined['Width_x_cumsum'] - joined['Width_x'] + joined['Width_y'] / 2).tolist(),
            y=joined['Length_y'],
            width=(joined['Width_y']).tolist(),  # customize width here
            marker_color='red',
            opacity=1,
            name='Requirements',
            xaxis='x',
            customdata=joined['Part'].tolist(),
            hovertemplate='Width (mm): %{width:.f}, Length (mm): %{y:.f}, Part: %{customdata:.s}'
        )])

        if len(joined_5) > 0:
            trace3 = go.Figure(data=[go.Bar(
                x=np.ceil((joined_5['Width_x_cumsum'] - joined_5['Width_x'] + joined_5['Width_y'] / 2)).tolist(),
                y=joined_5['Length_y'],  # joined_3['Length_y_cumsum'] -
                #     y=joined_4['Length_y_cumsum'] - joined_4['Length_y'], #joined_3['Length_y_cumsum'] -
                width=(joined_5['Width_y']).tolist(),  # customize width here
                marker_color='green',
                opacity=0.8,
                name='Variable req. on 1 piece',
                xaxis='x',
                customdata=joined['Part'].tolist(),
                hovertemplate='Width (mm): %{width:.f}, Length (mm): %{y:.f}, Part: %{customdata:.s}',
            )])

        layout = go.Layout(
            barmode='group',
            xaxis=dict(
                title='x actual',
                rangemode="tozero",
                #         anchor='x',
                #         overlaying='x',
                side="left",
                range=[0, max(joined_2['Width_x_cumsum'])]
            ),
        )
        if len(joined_5) > 0:
            fig = go.Figure(
                data=trace1.data + trace3.data + trace2.data,
                layout=layout
            )
        else:
            fig = go.Figure(
                data=trace1.data + trace2.data,
                layout=layout
            )
        return fig

    @staticmethod
    def get_diff_size_piece_visual_dataframe(all_wood_df):
        """
        Check for parts that are different sizes but use the same stock piece and get a dataframe with those pieces.
        Here we make a dataframe with the different size pieces but add the length of the all pieces. This will
        make the visual better.
        """
        wood_list = []
        for part in all_wood_df.to_dict(orient="records"):

            for part2 in all_wood_df.to_dict(orient="records"):
                if (part['Wood list index'] == part2['Wood list index'] and
                        part['Requirements list index'] != part2['Requirements list index'] and
                        part['Width_y'] < part2['Width_y']):
                    part3 = part2.copy()
                    part3['Width_y'] = part['Width_y']

                    # make sure that we only take the element once to check for other pieces. If there are two other 
                    # pieces with similar lengths, only take once the element with different length.
                    if part['Requirements list index'] not in (pd.DataFrame(wood_list, columns=all_wood_df.columns
                                                                            )['Requirements list index'].tolist()):
                        wood_list.append(part)
                        wood_list.append(part3)
                    else:
                        wood_list.append(part3)
        new_df = pd.DataFrame(wood_list)
        if len(new_df) > 0:
            new_df = new_df.sort_values(['Wood list index', 'Width_y', 'Length_y'], ascending=True)
        return new_df


def main():
    # Initialize the DigitalIntake
    DigIn = DigitalIntake()
    # Fetch data from the database
    DigIn.wood_list = DigIn.get_data_api()

    # Setup Streamlit on Local URL: http://localhost:8501
    st.title("Available Wood Table")
    st.text("The following table demonstrates real-time data captured by\n"
            "the digital intake process in the Robot Lab as means of building\n"
            "a data base of residual wood.")

    st.subheader("Digital Intake Results from the Robot Lab")

    dataset = DigIn.data
    # style = 'API'

    dataset.to_csv("Generated_wood_data.csv", index=False)
    st.write(pd.DataFrame(DigIn.wood_list))
    st.write(f'TOTAL Number of wood scanned: {len(dataset["Index"])}')
    st.download_button('Download Table', dataset.to_csv(), mime='text/csv')

    # if st.button('Generate new wood dataset'):
    #     st.write('New Wood dataset is generated')
    #     wood_list = DigIn.generate_new_wood(n=50)
    #     dataset = pd.DataFrame(wood_list)
    #     dataset.to_csv("Generated_wood_data.csv", index=False)

    image = Image.open('stool_image.jpg')
    st.image(image, caption='Image of a stool')

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

    # print(DigIn.requirement_list)
    # length_values = dataset['Length']
    st.subheader('Length and width distribution in mm of the dataset\n')
    fig = GraphicalElements(dataset).barchart_plotly_one(
        dataset.sort_values(by=['Length'], ascending=False), color='blue')
    st.plotly_chart(fig, use_container_width=True)

    if os.path.exists('requirements.csv'):
        requirement_df = pd.read_csv('requirements.csv')
        # length_values_req = requirement_df['Length']
        st.subheader('Length and width distribution in mm of the requirements\n')
        fig = GraphicalElements(dataset).barchart_plotly_one(requirement_df, color='red', requirements=True)
    st.plotly_chart(fig, use_container_width=True)

    # st.subheader("Match the requirements with the available wood")
    st.subheader('Reserve the wood based on matched requirements')
    res_name = st.text_input('Reservation name', 'Javid')
    res_number = st.text_input('Reservation number', '1')
    st.write('The reservation will be on ', res_name + '-' + res_number)

    # def match_requirement_dataset_improved(self, requirement_list, n_runs = 10):

    option = st.selectbox(
        'How would you like to optimize the matching algorithm?',
        ('Minimum waste', 'Keep long planks', 'Most parts found in database', 'Minimum cuts needed'))

    n_runs = st.slider('Number of runs for Monte Carlo:', 1, 100, 30)
    if st.button('Match the requirements with the available wood - improved - Monte Carlo'):

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
                st.write(matching_df)
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

        else:
            st.write('Requirements are not found')

    if st.button('Match the requirements with the available wood - simple'):

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

    if st.button('Match the requirements with the available wood - Euclidean Distance'):

        if os.path.exists('requirements.csv'):
            requirement_df = pd.read_csv('requirements.csv')
            requirement_list = requirement_df.to_dict('records')

            # st.write(requirement_df)
            # print(requirement_list)
            matching_df, unmatched_df = DigIn.match_euc_dis(requirement_list)
            dataset = pd.DataFrame(DigIn.wood_list)

            # Visualize the matched planks
            if len(matching_df):
                st.write(matching_df)
                st.write("Matching Dataframe is saved in a CSV file")
                matching_df.to_csv('matching_df.csv', index=False)
                st.subheader('Showing all the planks in the dataset and the matching requirements')
                fig = GraphicalElements(dataset).barchart_plotly_two_improved(matching_df, requirement_df)
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
            st.write('requirements are not found')

    if st.button('Reserve the matched wood'):
        matching_df = pd.read_csv('matching_df.csv')
        matching_list = matching_df.to_dict('records')

        for index, row in enumerate(matching_list):
            DigIn.wood_list[row['Wood list index']]['Reservation name'] = str(res_name + '-' + res_number)
            DigIn.wood_list[row['Wood list index']]['Reserved'] = True
            DigIn.wood_list[row['Wood list index']]['Reservation time'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.write('The matched wood is reserved!')
        st.write(pd.DataFrame(DigIn.wood_list))

        pd.DataFrame(DigIn.wood_list).to_csv('Generated_wood_data.csv', index=False)

    st.subheader('Unreserve items')
    if st.button('Unreserve all wood'):
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
        # print(dataset[dataset['Reservation name']])
        print(dataset.loc[dataset['Reservation name'] == str(
            unres_name + '-' + unres_number)])

        dataset.loc[dataset['Reservation name'] == str(
            unres_name + '-' + unres_number), 'Reserved'] = False
        dataset.loc[dataset['Reservation name'] == str(
            unres_name + '-' + unres_number), 'Reservation time'] = None
        dataset.loc[dataset['Reservation name'] == str(
            unres_name + '-' + unres_number), 'Reservation name'] = None

        dataset.to_csv('Generated_wood_data.csv', index=False)
        DigIn.wood_list = dataset.to_dict('records')

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
    slider_height = st.select_slider('{}?'.format(filter_criteria), options=[True, False],
                                     value=(slider_min_val, slider_max_val), key=filter_criteria)
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

    style = 'API'
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

    st.subheader('Length Distribution in mm of the dataset in Plotly')
    fig = GraphicalElements(dataset).distplot_plotly(x_column="Length", y_column="Width",
                                                     color="Type")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
