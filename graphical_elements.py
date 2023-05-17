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