import pandas as pd
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


if __name__ == "__main__":
    # Initializing data frame from CSV
    csv_file = getcwd() + '/Datawood.csv'
    csv_content = DigitalIntake(csv_file).convert()
    df = pd.DataFrame(csv_content)

    # Setup Streamlit on Local URL: http://localhost:8501
    st.title("Available Wood Table")
    st.text("The following table demonstrates real-time data captured by\n"
            "the digital intake process in the Robot Lab as means of building\n"
            "a data base of residual wood.")

    st.subheader("Digital Intake Results (Waste Wood from CW4N)")
    st.write(csv_content)
    st.subheader(f'TOTAL Number of wood scanned: {len(csv_content["Index"])}\n\n')
    st.download_button('Download Table', str(csv_content), mime='text/csv')

    length_values = csv_content['Length']
    st.subheader('Length Distribution in mm\n')
    st.bar_chart(length_values)

    filter_criteria = 'Length'
    st.subheader(f"Filter Desired Pieces Based on {filter_criteria}")
    slider_min_val = min(sorted(csv_content[filter_criteria]))
    slider_max_val = max(sorted(csv_content[filter_criteria]))
    length_slider = st.slider('Length in mm', min_value=slider_min_val, max_value=slider_max_val)
    st.write("The Items in the table are the ID values"
             " of the pieces under the selected length")

    filtered = [
        row for index, row in df.iterrows()
        if row[filter_criteria] < length_slider
    ]

    filtered_df = pd.DataFrame(filtered)
    print(filtered_df)
    st.write(filtered_df)
    st.download_button('Download Selection', str(filtered), mime='text/csv')
