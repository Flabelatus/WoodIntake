import os
import pandas as pd
import random
import numpy as np
import streamlit as st
from datetime import datetime
import requests
from dotenv import load_dotenv
import json

from graphical_elements import GraphicalElements



def matching_design(requirement_list, wood_list):
    matching_df, unmatched_df, optimal_list = [], [], []
    option = st.selectbox(
            'How would you like to optimize the matching algorithm?',
            ('Minimum waste', 'Keep long planks', 'Minimum amount of planks needed' #,
                #'Same thickness for top parts'
                ))
    if option == 'Keep long planks':
        threshold = st.number_input('What length defines a long plank (in mm) - what is the threshold?', min_value = 300, max_value = 10000, value = 1000)

    n_runs = st.slider('Number of runs for Monte Carlo:', 1, 1000, 100)
    if st.button('Match the requirements with the available wood'):
        matching_df, unmatched_df, optimal_list = MC_match(requirement_list, wood_list, n_runs, option)
    return matching_df, unmatched_df, optimal_list

def MC_match(requirement_list, wood_list, n_runs, option):
    # if os.path.exists('requirements.csv'):
        # requirement_df = pd.read_csv('requirements.csv')
        # requirement_list = requirement_df.to_dict('records')



    matching_df, unmatched_df, optimal_list = match_requirement_dataset_improved(requirement_list, wood_list,
                                                                         n_runs=n_runs, option=option)
    dataset = pd.DataFrame(wood_list).copy()
    requirement_df = pd.DataFrame(requirement_list)

    st.write(matching_df)
    # Visualize the matched planks
    if len(matching_df):
        
        # st.write("Matching Dataframe is saved in a CSV file")
        # matching_df.to_csv('matching_df.csv', index=False)


        st.subheader('Showing all the planks in the dataset and the matching requirements')
        fig = GraphicalElements(dataset).barchart_plotly_two_improved(matching_df, requirement_df)
        # fig = Graphical_elements(dataset).barchart_plotly_two(matching_df, requirement_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write('Could not match any unreserved planks with the requirements')

    if len(unmatched_df):
        st.write(
            'The following chart shows all the required planks that cannot be found in the database based on'
            ' the used matching algorithm')
        fig = GraphicalElements(dataset).barchart_plotly_one(dataset=unmatched_df,
                                                             color='maroon', requirements=True)
        st.plotly_chart(fig, use_container_width=True)

    st.write("The selected run has the following features:")
    st.table(optimal_list.head(1))

    return matching_df, unmatched_df, optimal_list
    # else:
        # st.write('Requirements are not found')

def match_requirement_dataset_improved(requirement_list, wood_list, n_runs=30, option="Minimum waste", threshold = 1000):

    cuts_list = []
    wood_cuts_list = []
    stock_pieces_list = []
    matching_list_all = []
    success_list = []
    waste_list = []
    unmatched_list = []
    used_planks_list = []
    long_planks_list = []

    # st.write(requirement_list)
    # print(requirement_list)
    # print(wood_list)

    for run in range(n_runs):
        stock_pieces = pd.DataFrame(wood_list).copy()
        parts = pd.DataFrame(requirement_list).copy()
        # parts = parts_df.copy()
        # stock_pieces = stock_pieces_start_df.copy()

        # shuffle for different options, for keeping long planks, use the smaller planks first.
        # for needing minimum amount of planks, first use the longer planks

        if option == 'Keep long planks':
            stock_pieces = stock_pieces.sort_values(by='Length', ascending=True).reset_index(drop=True)
            parts = parts.sample(frac=1)
        elif option == 'Minimum amount of planks needed':
            stock_pieces = stock_pieces.sort_values(by='Length', ascending=False).reset_index(drop=True)
            parts = parts.sample(frac=1)
        else:
            stock_pieces = stock_pieces.sample(frac=1).reset_index(drop=True)
        # wood_cuts = list([0] * len(stock_pieces))

        matching_list = []
        cuts = 0
        success = 0
        waste = []
        # print(run, 'run')
        for i, part in parts.iterrows():
            # print(i, 'part')

            for j, stock in stock_pieces.iterrows():
                # print(j, ' stock')

                if part['length'] <= stock['Length'] and part['width'] <= stock['Width']:  # and
                    
                    # edit the stock piece by taking of the part length (sawed of)
                    stock_pieces.loc[j, 'Length'] = stock['Length'] - part['length']
                    stock_pieces.loc[j, 'Width'] = stock['Width']
                    stock_pieces.loc[j, 'Height'] = stock['Height']

                    cuts += 1
                    success += 1
                    waste.append((stock['Width'] - part['width']) * (stock['Height'] - part['height']))

                    matching_row = {'Requirements list index': part['part_index'], 'Wood list index': stock['Index']}
                    matching_list.append(matching_row)
                    break

        # count the amount of planks being used by counting the unique id's of the matching list
        if len(matching_list) > 0:
            used_planks = pd.DataFrame(matching_list)['Wood list index'].unique()
            used_planks_count = np.count_nonzero(used_planks)
        else:
            used_planks = []
            used_planks_count = 0
        used_planks_list.append(used_planks_count)


        unused_stock = stock_pieces[~stock_pieces['Index'].isin(used_planks)]
        long_planks_list.append(np.sum(unused_stock['Length'] > threshold))

        success_list.append(success == len(parts))
        cuts_list.append(cuts)
        waste_list.append(waste)
        stock_pieces_list.append(stock_pieces)
        matching_list_all.append(matching_list)

    # get the sum of waste per run
    waste_per_run_list =[sum(run) for run in waste_list]

    # make a dataframe of all variables to be able to sort
    optimal_list = pd.DataFrame({"Run": range(n_runs), "Waste": waste_per_run_list, "Used long planks": long_planks_list, 
         "Amount of planks used": used_planks_list})
    
    if option == 'Minimum waste':
        index_min = optimal_list.sort_values('Waste', ascending = False).head(1).index[0]
        # optimized_feature = int(optimal_list.iloc[[index_min]]['Waste'])
    elif option == 'Keep long planks':
        index_min = optimal_list.sort_values(['Used long planks', 'Waste']).head(1).index[0]
        # optimized_feature = int(optimal_list.iloc[[index_min]]['Used long planks'])
    elif option == 'Minimum amount of planks needed':
        index_min = optimal_list.sort_values(['Amount of planks used', 'Waste']).head(1).index[0]
        # optimized_feature = int(optimal_list.iloc[[index_min]]['Amount of planks used'])
    # elif option == 'Same thickness for top parts':
    #     index_min = wood_cuts_list.index(min(wood_cuts_list))
    print(optimal_list)

    matching_df = pd.DataFrame(matching_list_all[index_min])



    if len(matching_list):
        for index, row_req in enumerate(requirement_list):
            # if matching_df['Index'].isin(index):
            if index not in matching_df['Requirements list index'].values:
                unmatched_list.append(row_req)
    unmatched_df = pd.DataFrame(unmatched_list)

    return matching_df, unmatched_df, optimal_list


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

