""" Simple demo of an interface from CSV digital intake 
Javid Jooshesh <j.jooshesh@hva.nl>"""

import os
import pandas as pd
import random
import numpy as np
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from os import getcwd
from PIL import Image


class DigitalIntake:
    """A class to read the saved csv data from the wood intake process"""

    def __init__(self, name):
        self.name = name
        self.fields = [
            "Index",
            "Width",
            "Length",
            "Height",
            "Type",
            "Color",
            "Indexed",
            "Density",
            "Weight",
            'Reserved',
            "Reservation name",
            "Reservation time",

        ]
        self.data = pd.read_csv(self.name, usecols=self.fields, delimiter=',')
        # self.wood_list = []
        # self.requirement_list = []
        # self.matching_list = []


    def __str__(self):
        return f"Data base {self.name}, containing the data in CSV format."

    def display_column(self, field=None, head=5):
        with pd.option_context(
                'display.max_rows', None, 'display.max_columns', None):
            if field in self.fields:
                print(f"{self.name}\n{self.data[field].head(head)}")
            elif field is None:
                print(f"{self.name}\n{self.data.head(head)}")
            else:
                print("The field does not exist in the data base")

    def generate_new_wood(self, n = 10, min_width = 100, max_width = 300, 
                      min_length = 300, max_length = 1000, 
                     min_height = 30, max_height = 50, 
                      min_density = 200, max_density = 700):
        wood_list = []
        for index in range(n):
            width = random.randint(min_width, max_width)
            length = random.randint(min_length, max_length)
            height = random.randint(min_height, max_height)
            Type = random.choice(['Soft Wood', 'Hard Wood'])
            Color = np.random.randint(100,200, size = 3)
            Indexed = datetime.now().strftime("%Y-%m-%d %H:%M")
            Density = random.randint(min_density, max_density)
            Weight = int(width * length * height * Density / 10000 / 1000)
            
            row = {'Index': index, 'Width': width, 'Length': length, 'Height': height,
                    'Type': Type, 'Color': Color, 'Indexed': Indexed, 'Density': Density, 'Weight': Weight, 
                    'Reserved': False, 'Reservation name': '', 'Reservation time': ''}
            wood_list.append(row)

        self.wood_list = wood_list
        return wood_list

    def generate_requirements(self, size = 1, n_planks = 20, complexity = 5, 
                          width = [100,300], length = [300, 1000], height = [30,50]):
        '''
        This function uses size and n_planks as inputs to generate requirements.
        
        Size: the size input goes from 1 to 5, for size 5 it will generate longer lengths and widths 
        N_planks: the number of planks that is needed for this project
        Complexity: the number of differing lengths and widths that is needed for this project
        '''
        
        requirement_list = []
        
        min_width, max_width = width
        min_length, max_length = length
        min_height, max_height = height
        max_size = 5
        index = 0
        
        for plank in range(int(n_planks/complexity)):
            width = int(random.randint(min_width, max_width) / (max_size-size))
            length = int(random.randint(min_length, max_length) / (max_size-size))
            height = min_height
            
    #         height = random.randint(min_height, max_height)
            
            
            for _ in range(complexity):
                index += 1
                row = {'Index': index, 'Width': width, 'Length': length, 'Height': height}
                requirement_list.append(row)
        self.requirement_list = requirement_list
        return requirement_list

    def match_requirements_dataset(self, requirement_list):
        ''' 
        This function matches the rquirements from the generated requirements and selects fitting  planks
        from the available dataset.

        '''
        wood_list = self.wood_list.copy()
        matching_list = []
        for index, row_req in enumerate(requirement_list):
            for index, row_wood in enumerate(wood_list):
                if (row_req['Width'] < row_wood['Width'] and row_req['Length'] < row_wood['Length'] 
                and not row_wood['Reserved']):
                    matching_row = {'Requirements list index': row_req['Index'], 'Width req':row_req['Width'], 
                        'Length req':row_req['Length'], 'Height req':row_req['Height'],
                        'Wood list index': row_wood['Index'], 'Width DB': row_wood['Width'],
                        'Length DB': row_wood['Length'], 'Height DB': row_wood['Height']}

                    wood_list[index]['Reserved'] = True
                    matching_list.append(matching_row)
                    break

        self.matching_list = matching_list
        matching_df = pd.DataFrame(matching_list)
        return matching_df

    def match_euc_dis(self, requirements_list):
        ''' 
        This function matches the rquirements from the generated requirements based on the Euclidean Distance 
        and selects fitting  planks from the available dataset. 

        '''
        wood_list = self.wood_list.copy()
        matching_list = []
        k = len(requirements_list)
        for index, row_req in enumerate(requirements_list):
            distances = []
            wood_db = []
            wood_db_index = []

            for index, row_wood in enumerate(wood_list):
                if (
    #                 row_req['Width'] < row_wood['Width'] and row_req['Length'] < row_wood['Length'] and
                not row_wood['Reserved']):
                    wood_db.append(np.array([row_wood['Width'], row_wood['Length'], row_wood['Height']]))
                    wood_db_index.append(row_wood['Index'])

            wood_piece = np.array([row_req['Width'], row_req['Length'], row_req['Height']])
            if wood_db != []:
                distances = (np.sqrt(np.sum((wood_db - wood_piece)**2, axis=1)))        

                # Find the indices of the k nearest neighbors in the database
                k_nearest = np.argsort(distances)[:k]

                # Take the minimum of the k nearest neighbors to determine the best match
                best_match_index = np.argmin(distances)

                # Add the matched piece to the matched_indices list
                index = wood_db_index[best_match_index]

    #             print('wood piece: ' + str(wood_piece) + 'wood_db: ' + str(wood_list[index]))

                matching_row = {'Requirements list index': row_req['Index'], 'Wood list index': index}
                wood_list[index]['Reserved'] = True
                matching_list.append(matching_row)


        matching_df = pd.DataFrame(matching_list)
        return matching_df
    

    @st.cache
    def fetch(self, row=None):
        fetched_data = pd.read_csv(self.name, nrows=row, usecols=self.fields)
        return fetched_data

    @st.cache
    def convert(self):
        return self.data

class Graphical_elements:

    def __init__(self, dataset):
        self.dataset = dataset

    def distplot_plotly(self, x_column, y_column, color):

        
        fig = px.histogram(self.dataset, x=x_column, color=color,
                           marginal="box", # or violin, rug
                           hover_data=self.dataset.columns)

        return fig
        
    def barchart_plotly_one(self, dataset, color):

        fig = go.Figure(data=[go.Bar(
            x=((dataset['Width'].cumsum() - dataset['Width']/2)).tolist(),
            y=dataset['Length'],
            width=(dataset['Width']).tolist(), # customize width here
            marker_color=color,
            opacity = 0.8,
            name = 'wood in database',
            hovertemplate = 'Width (mm): %{width:.f}, Length (mm): %{y:.f}'
        )])

        return fig


    def barchart_plotly_two(self, matching_df, requirement_df):
        '''
        This function makes a barchart in which the length and with of the planks are visualized for both
        the required planks as the wood in the database.
        '''


        wood_df = self.dataset.copy()
        # join the two, keeping all of df1's indices
        joined = pd.merge(matching_df, wood_df.loc[:,['Index', 'Width', 'Length', 'Height']],
                               left_on='Wood list index', right_on = 'Index', how='inner')
        joined = pd.merge(joined, requirement_df, left_on=['Requirements list index'], right_on=['Index'], how='inner')

        # Make a trace with a bar chart based on a cumsum to make sure all planks are next to each other. 
        # Furthermore the cumsum of the requirements is added, as these planks need to be in between the planks of the database
        # Lastly, the x starts before a first requirement plank is added and it is lined up in the middle.
        trace1 = go.Figure(data=[go.Bar(
            x=((joined['Width_x'].cumsum()+ joined['Width_y'].cumsum() - joined['Width_y'] - joined['Width_x']/2)).tolist(),
            y=joined['Length_x'],
            width=(joined['Width_x']).tolist(), # customize width here
            marker_color='blue',
            opacity = 0.8,
            name = 'wood in database',
            hovertemplate = 'Width (mm): %{width:.f}, Length (mm): %{y:.f}'
        )])

        trace2 = go.Figure(data=[go.Bar(
            x=((joined['Width_x'].cumsum() + joined['Width_y'].cumsum() - joined['Width_y']/2)).tolist(),
            y=joined['Length_y'],
            width=(joined['Width_y']).tolist(), # customize width here
            marker_color='red',
            opacity = 0.8,
            name = 'requirement',
            hovertemplate = 'Width (mm): %{width:.f}, Length (mm): %{y:.f}'

        )])

        fig = go.Figure(data = trace1.data + trace2.data)

        return fig


def main():
    # Initializing data frame from CSV
    csv_file = getcwd() + '/Generated_wood_data.csv'
    DigIn = DigitalIntake(csv_file)
    dataset = DigitalIntake(csv_file).convert()
    DigIn.wood_list = dataset.to_dict('records')


    
    # wood_list = DigIn.generate_new_wood(n = 50)
    # dataset = pd.DataFrame(wood_list)

    # Setup Streamlit on Local URL: http://localhost:8501
    st.title("Available Wood Table")
    st.text("The following table demonstrates real-time data captured by\n"
            "the digital intake process in the Robot Lab as means of building\n"
            "a data base of residual wood.")

    st.subheader("Digital Intake Results (Waste Wood from CW4N)")
    

    if st.button('Update the dataset'):
        st.write(pd.DataFrame(DigIn.wood_list))
    else:
        st.write(pd.DataFrame(DigIn.wood_list))

    st.write(f'TOTAL Number of wood scanned: {len(dataset["Index"])}')
    st.download_button('Download Table', dataset.to_csv(), mime='text/csv')

    if st.button('Generate new wood dataset'):
        st.write('New Wood dataset is generated')
        wood_list = DigIn.generate_new_wood(n = 50)
        dataset = pd.DataFrame(wood_list)
        dataset.to_csv("Generated_wood_data.csv", index = False)


    image = Image.open('stool_image.jpg')
    st.image(image, caption='Image of a stool')

    st.subheader("Generate a general requirements set (maybe a stool)")
    if st.button('Generate requirements'):
        st.write('General requirements are generated')
        DigIn.generate_requirements(size = 4, n_planks = 30)
        requirement_df = pd.DataFrame(DigIn.requirement_list)
        # print(DigIn.requirement_list)
        st.write(requirement_df)
        st.write("Requirements are saved in a CSV file")
        requirement_df.to_csv('~/Downloads/WoodIntake-master/requirements.csv', index = False)

    # print(DigIn.requirement_list)
    length_values = dataset['Length']
    st.subheader('Length Distribution in mm of the dataset\n')
    fig = Graphical_elements(dataset).barchart_plotly_one(
                dataset.sort_values(by=['Length'], ascending = False), color = 'blue')
    st.plotly_chart(fig, use_container_width=True)

    if os.path.exists('requirements.csv'):
        requirement_df = pd.read_csv('requirements.csv')
        length_values_req = requirement_df['Length']
        st.subheader('Length Distribution in mm of the requirements\n')
        fig = Graphical_elements(dataset).barchart_plotly_one(requirement_df, color = 'red')
    st.plotly_chart(fig, use_container_width=True)
    
    # st.subheader("Match the requirements with the available wood")
    st.subheader('Reserve the wood based on matched requirements')
    res_name = st.text_input('Reservation name', 'Javid')
    res_number = st.text_input('Reservation number', '1')
    st.write('The reservation will be on ', res_name + '-' + res_number)

    if st.button('Match the requirements with the available wood - simple'):

        if os.path.exists('requirements.csv'):
            requirement_df = pd.read_csv('requirements.csv')
            requirement_list = requirement_df.to_dict('records')

            # st.write(requirement_df)
            # print(requirement_list)
            matching_df = DigIn.match_requirements_dataset(requirement_list)
            dataset = pd.DataFrame(DigIn.wood_list)
            st.write(matching_df)
            st.write("Matching Dataframe is saved in a CSV file")
            matching_df.to_csv('~/Downloads/WoodIntake-master/matching_df.csv', index = False)
            #Visualize the matched planks
            st.subheader('Showing all the planks in the dataset and the matching requirements')
            fig = Graphical_elements(dataset).barchart_plotly_two(matching_df, requirement_df)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.write('requirements are not found')

    if st.button('Match the requirements with the available wood - Euclidean Distance'):

        if os.path.exists('requirements.csv'):
            requirement_df = pd.read_csv('requirements.csv')
            requirement_list = requirement_df.to_dict('records')

            # st.write(requirement_df)
            # print(requirement_list)
            matching_df = DigIn.match_euc_dis(requirement_list)
            dataset = pd.DataFrame(DigIn.wood_list)
            st.write(matching_df)
            st.write("Matching Dataframe is saved in a CSV file")
            matching_df.to_csv('~/Downloads/WoodIntake-master/matching_df.csv', index = False)
            #Visualize the matched planks
            st.subheader('Showing all the planks in the dataset and the matching requirements')
            fig = Graphical_elements(dataset).barchart_plotly_two(matching_df, requirement_df)
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

        pd.DataFrame(DigIn.wood_list).to_csv('Generated_wood_data.csv', index = False)

    
    st.subheader('Unreserve items')
    if st.button('Unreserve all wood'):
        dataset = pd.DataFrame(DigIn.wood_list)
        dataset['Reserved'] = False
        dataset['Reservation name'] = ''
        dataset['Reservation time'] = ''

        dataset.to_csv('Generated_wood_data.csv', index = False)
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

        dataset.to_csv('Generated_wood_data.csv', index = False)
        DigIn.wood_list = dataset.to_dict('records')


    st.subheader('Length Distribution in mm of the dataset in Plotly')
    fig = Graphical_elements(dataset).distplot_plotly(x_column = "Length", y_column = "Width", 
            color = "Type")
    st.plotly_chart(fig, use_container_width=True)

    filter_criteria = 'Length'
    st.subheader(f"Filter Desired Pieces Based on {filter_criteria}")
    slider_min_val = min(sorted(dataset[filter_criteria]))
    slider_max_val = max(sorted(dataset[filter_criteria]))
    length_slider = st.slider('Length in mm', min_value=slider_min_val, max_value=slider_max_val,
                        value = slider_max_val)
    st.write("The Items in the table are the ID values"
             " of the pieces under the selected length")

    filtered = [
        row for index, row in dataset.iterrows()
        if row[filter_criteria] < length_slider
    ]

    filtered_df = pd.DataFrame(filtered)
    # print(filtered_df)
    st.write(filtered_df)
    st.download_button('Download Selection', filtered_df.to_csv(), mime='text/csv')

    colors = dataset['Color']
    # rgb_column = [row.split(',') for row in list(colors)]
    # rgb = []
    # for rgb_list in rgb_column:
    #     rgb.append(tuple([int(value) for value in rgb_list]))

    img = []
    for index, color in enumerate(colors):
        img.append((
            Image.new('RGB', (100, 200), tuple(colors[index])),
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


if __name__ == "__main__":
    main()
