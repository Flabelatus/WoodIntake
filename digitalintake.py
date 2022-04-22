# import sqlite3
import pandas as pd
# import numpy as np
import streamlit as st


class DigitalIntake:
    """A class to read the saved csv data from the wood intake process"""

    def __init__(self, name):
        self.name = name
        self.data = pd.read_csv(self.name, usecols=[
            "Index",
            "Width",
            "Length",
            "Height",
            "Type",
            "Color",
            "Indexed",
            "Density",
            "Weight"
        ], delimiter=',')
        self.fields = [item for item in self.data]

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
        fetched_data = pd.read_csv(self.name, nrows=row, usecols=[
            "Index",
            "Width",
            "Length",
            "Height",
            "Type",
            "Color",
            "Indexed",
            "Density",
            "Weight"
        ])
        return fetched_data


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
    directory = 'C:\\Users\\scanlab\\Documents\\' \
                'HvA\\DATA-WOOD\\Test Folder CSV\\'
    csv_file = 'Datawood.csv'
    filename = directory + csv_file
    wood_intake = DigitalIntake(filename)
    wood_intake.display_column(head=10)
    info = wood_intake.data

    # Streamlit on Local URL: http://localhost:8501
    st.title("Data Wood Table")
    st.subheader("Digital Intake Results (Waste Wood from CW4N)")
    st.text("The following table demonstrates real-time data captured by\n"
            "the digital intake process in the Robot Lab as means of building\n"
            "a data base of residual wood.")
    st.write(info)
    st.download_button('Download Table', str(info))

    st.subheader(f'TOTAL Number of wood scanned: {len(info["Index"])}\n\n')
    hist_values = info['Density']
    st.write('Density Distribution\n')
    st.bar_chart(hist_values)

    st.subheader("Filter Desired Pieces Based on Length")
    selected_size = st.slider('Length in mm', min(info['Length']), max(info['Length'] + 1))
    st.write("The Items in the table are the ID values"
             " of the pieces under the selected length")

    data = None
    for index, sizes in enumerate(sorted(info['Length'])):
        if sizes < selected_size:
            data = wood_intake.fetch(index)

    selected_df = pd.DataFrame(data)
    st.write(selected_df)
