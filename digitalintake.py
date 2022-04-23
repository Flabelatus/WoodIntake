# import sqlite3
import pandas as pd
# import numpy as np
import streamlit as st
from os import getcwd


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
            "Weight"
        ]
        self.data = pd.read_csv(self.name, usecols=self.fields, delimiter=',')


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

    @st.cache
    def fetch(self, row=None):
        fetched_data = pd.read_csv(self.name, nrows=row, usecols=self.fields)
        return fetched_data

    @st.cache
    def convert(self):
        return self.data


# class DataBase(object):

#     def __init__(self, data):
#         self.index = None
#         self.data = data
#         self.properties = {}
#         db = sqlite3.connect("data_wood.sqlite")
#
#     def add_data(self, data):
#         pass


if __name__ == "__main__":
    directory = getcwd()
    csv_file = '/Datawood.csv'
    filename = directory + csv_file
    wood_intake = DigitalIntake(filename)
    wood_intake.display_column(head=3)
    info = wood_intake.convert()

    # Streamlit on Local URL: http://localhost:8501
    st.title("Available Wood Table")
    st.text("The following table demonstrates real-time data captured by\n"
            "the digital intake process in the Robot Lab as means of building\n"
            "a data base of residual wood.")

    st.subheader("Digital Intake Results (Waste Wood from CW4N)")
    st.write(info)
    st.subheader(f'TOTAL Number of wood scanned: {len(info["Index"])}\n\n')
    st.download_button('Download Table', str(info), mime='text/csv')

    length_values = info['Length']
    st.subheader('Length Distribution in mm\n')
    st.bar_chart(length_values)

    criteria = 'Length'
    st.subheader(f"Filter Desired Pieces Based on {criteria}")
    slider_min_val = min(sorted(info[criteria]))
    slider_max_val = max(sorted(info[criteria]))
    selected_size = st.slider('Length in mm', value=sorted())
    st.write("The Items in the table are the ID values"
             " of the pieces under the selected length")
    st.write(selected_size)

    data = None
    for index, sizes in enumerate(sorted(info[criteria])):
        if sizes < selected_size:
            data = wood_intake.fetch(index)

    selected_df = pd.DataFrame(data)
    print(selected_df)
    st.write(selected_df)
    st.download_button('Download Selection', str(data), mime='text/csv')
