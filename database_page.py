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
import streamlit_authenticator as stauth

import matplotlib.pyplot as plt
from shapely import LineString
from shapely.geometry import LinearRing
from shapely.plotting import plot_line, plot_points

import ifcopenshell
import ifcopenshell.util.element
from ifcopenshell.api import run
from ifcopenshell.util import representation
from ifcopenshell.util import shape_builder
import figures
import mathutils

from reservation_system import unreserve


class database_page:
    """A class to read the saved csv data from the wood intake process"""

    def __init__(self, Digital_intake_class):
        DigIn = Digital_intake_class 
        self.matching_list = None
        self.wood_list = None
        self.requirement_list = None
        self.fields = DigIn.fields
        self.data = DigIn.get_data_api()

        # Fetch data from the database
        DigIn.wood_list = DigIn.get_data_api()

        # Setup Streamlit on Local URL: http://localhost:8501
        st.title("Available Wood Table")
        st.text("The following table demonstrates real-time data captured by\n"
                "the digital intake process in the Robot Lab as means of building\n"
                "a data base of residual wood.")

        
        # show an image of the overview
        database_image = Image.open('database_image.png')

        st.image(database_image, caption='Database image')


        st.subheader("Digital Intake Results from the Robot Lab")
        dataset = DigIn.data
        # style = 'API'

        
        dataset_tab1, dataset_tab2, dataset_tab3 = st.tabs(["Current database", "Add elements to the dataset", "Login"])

        with dataset_tab1:

            if st.button('Refresh data'):
                dataset = DigIn.data

            # add Derako wood manually
            info_string = {"Project": "Grill",
                    "Woodspecies": "Grenen PEFC",
                    "Paint":"DB-0022",
                    "Outdoor application": True,
                    "Size": "20x40",
                    "Fire treatment": True,
                    "Grill project - Hole from start": 50}


            Derako_row = {
            "Index":100,
            "Width":40,
            "Length":700,
            "Height":20,
            "Type":"unknown",
            "Color":"65,0,10",
            "Timestamp":"2022-12-21 11:37:56",
            "Density":550,
            "Weight":329,
            "Reserved":False,
            "Reservation name":"",
            "Reservation time":"",
            "Original piece": 100,
            "Production phase": "",
            "Design": "",
            "Kind of wood": "Residual",
            "Verified measurements": False,
            "Source":"Derako",
            "Price":5,
            "Info":json.dumps(info_string, indent = 2)}

            self.database_tabs(dataset, DigIn, Derako_row)

        with dataset_tab3:
            import yaml
            from yaml.loader import SafeLoader
            with open('authentication.yaml') as file:
                config = yaml.load(file, Loader=SafeLoader)
            authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
            config['preauthorized']
            )
            name, authentication_status, username = authenticator.login('Login', 'main')

            if authentication_status:
                authenticator.logout('Logout', 'main')
                st.write(f'Welcome *{name}*')
                # st.title('Some content')
            elif authentication_status == False:
                st.error('Username/password is incorrect')
            elif authentication_status == None:
                st.warning('Please enter your username and password')

        
        with dataset_tab2:
            if not authentication_status:
                st.write("You have to login to add elements to the database")
            else:
            


                col1, col2 = st.columns(2)
                with col1:
                    width_input = st.number_input('Width', min_value=None, max_value=None, value="min", step=None, format=None)
                    length_input = st.number_input('Length', min_value=None, max_value=None, value="min", step=None, format=None)
                    height_input = st.number_input('Height', min_value=None, max_value=None, value="min", step=None, format=None)
                    color_input = st.text_input("Color", value = "65,0,10")
                    has_holes_radio_1 = st.radio("Does the wood piece have holes?", ["Yes", "No"], key = 'has_holes_radio_1')
                    type_input = st.text_input('Type of wood', value = "unknown")
                    density_input = st.number_input('Density', min_value=None, max_value=None, value="min")
                    weight_input = st.number_input('Weight', min_value=None, max_value=None, value="min")
                    source_input = st.text_input('Source', value = "unknown")
                    

                with col2:
                    if has_holes_radio_1 == "Yes":
                        slider_hole_start = st.slider("Hole from start", min_value=10, max_value=300, value=50, step=10, 
                            key = 'slider_hole_start_1')
                        slider_hole_edge = st.slider("Hole from edge", min_value=10, max_value=20, value=10, step=2, 
                            key = 'slider_hole_edge_1')
                        slider_radius_hole = st.slider("Radius hole", min_value=1, max_value=10, value=5, step=1, 
                            key = 'slider_radius_hole_1')

                        new_info_string = info_string = {"Project": "Grill",
                            "Woodspecies": "Grenen PEFC",
                            "Paint":"DB-0022",
                            "Outdoor application": True,
                            "Size": "20x40",
                            "Fire treatment": True,
                            "Grill project - Hole from start": slider_hole_start}
                    else:
                        slider_hole_start = 50
                        slider_hole_edge = 10
                        slider_radius_hole = 5
                        new_info_string = {""}


                st.text(pd.DataFrame(DigIn.wood_list)['Index'].max()+1)
                new_row = {
                    "Index":pd.DataFrame(DigIn.wood_list)['Index'].max()+1,
                    "Width":width_input,
                    "Length":length_input,
                    "Height":height_input,
                    "Type":type_input,
                    "Color":color_input,
                    "Timestamp":"2022-12-21 11:37:56",
                    "Density":density_input,
                    "Weight":weight_input,
                    "Reserved":False,
                    "Reservation name":"",
                    "Reservation time":"",
                    "Source":source_input,
                    "Price":0,
                    "Info":json.dumps(new_info_string, indent = 2)}

            if st.button('Add wood piece to data'):
                DigIn.wood_list.append(new_row)
            

        

    def database_tabs(self, dataset, DigIn, Derako_row):



        DigIn.wood_list.append(Derako_row)
        

        st.write(pd.DataFrame(DigIn.wood_list))
        st.write(f'TOTAL Number of wood scanned: {len(dataset["Index"])}')
        st.download_button('Download Table', dataset.to_csv(), mime='text/csv')

        if st.button('Unreserve all wood'):
            unreserve(DigIn, unreserve_all = True)

        

        tab1, tab2, tab3, tab4 = st.tabs(["Database", "Requirements", "Known information about the wood", "Visual of a specific piece"])

        with tab1:
            st.subheader('Length and width distribution in mm of the dataset\n')
            fig = self.barchart_plotly_one(
                dataset.sort_values(by=['Length'], ascending=False), color='blue')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            project_id = st.text_input('Project id of requirement', 'All')
            # if project_id == 'All'
            requirement_list = DigIn.read_requirements_from_client(project_id)
                # length_values_req = requirement_df['Length']
            st.subheader('Length and width distribution in mm of the requirements\n')
            requirement_df = pd.DataFrame(requirement_list)
            if len(requirement_list) > 0:
                fig = self.barchart_plotly_one(requirement_df, color='red', requirements=True)
                st.plotly_chart(fig, use_container_width=True)
                if st.button('Delete all requirements through API'):
                    DigIn.delete_requirements_from_client(delete_all = True)
            else:
                st.text('There are no requirements currently available')
        with tab3:

            st.text("Here is the explanation of known information about the wood (if provided)")
            explanation_info = pd.DataFrame(
                {"Project":"G= Grill-project / L= linear project", 
                "Woodspecies with certification": "FSC/PEFC/non-certified",
                "Paint": "transparent/application",
                "Outdoor application?": "OS",
                "Size": "in mm",
                "Fire treatment": "HM"},
                index = ["Explanation"])
            st.table(explanation_info)

            st.text("This is the information of the wood, for the selected piece")

            st.table(pd.DataFrame(json.loads(Derako_row["Info"]), index = [Derako_row["Index"]]))

            
            

        with tab4:
            st.text("Make a specific piece (measurements in mm)")
            col1, col2 = st.columns(2)
            with col1:
                slider_width = st.slider("Width", min_value=20, max_value=100, value=40, step=5)
                slider_length = st.slider("Length", min_value=100, max_value=1000, value=500, step=50)
                slider_height = st.slider("Height", min_value=10, max_value=50, value=20, step=5)
                has_holes_radio = st.radio("Does the wood piece have holes?", ["Yes", "No"])

            with col2:
                if has_holes_radio == "Yes":
                    slider_hole_start = st.slider("Hole from start", min_value=10, max_value=300, value=50, step=10)
                    slider_hole_edge = st.slider("Hole from edge", min_value=10, max_value=20, value=10, step=2)
                    slider_radius_hole = st.slider("Radius hole", min_value=1, max_value=10, value=5, step=1)
                else:
                    slider_hole_start = 50
                    slider_hole_edge = 10
                    slider_radius_hole = 5
            
            
            

            info_string = {"Project": "Grill",
                "Woodspecies": "Grenen PEFC",
                "Paint":"DB-0022",
                "Outdoor application": True,
                "Size": "20x40",
                "Fire treatment": True,
                "Grill project - Hole from start": slider_hole_start}

            Derako_row = {
                "Index":100,
                "Width":slider_width,
                "Length":slider_length,
                "Height":slider_height,
                "Type":"unknown",
                "Color":"65,0,10",
                "Timestamp":"2022-12-21 11:37:56",
                "Density":550,
                "Weight":329,
                "Reserved":False,
                "Reservation name":"",
                "Reservation time":"",
                "Source":"Derako",
                "Price":5,
                "Info":json.dumps(info_string, indent = 2)}


            st.text("The wood has holes in these specific areas")
            self.make_shapely_figure(width = Derako_row['Width'], length = Derako_row['Length'], 
                has_holes = has_holes_radio == "Yes",
                hole_from_start = json.loads(Derako_row["Info"])["Grill project - Hole from start"],
                radius_hole = slider_radius_hole, hole_from_edge = slider_hole_edge)

            model = self.make_ifc_object(width = Derako_row['Width'], length = Derako_row['Length'], 
                height = Derako_row['Height'], 
                has_holes = has_holes_radio == "Yes",
                hole_from_start = json.loads(Derako_row["Info"])["Grill project - Hole from start"],
                radius_hole = slider_radius_hole, hole_from_edge = slider_hole_edge)

            if st.button('Download IFC Object'):
                model.write("/Users/jerome/Downloads/my_beam_with_holes.ifc")

        # with tab3:
        #    st.header("An owl")
        #    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


        # st.subheader('Length Distribution in mm of the dataset in Plotly')
        # fig = self.distplot_plotly(dataset = dataset, x_column="Length", y_column="Width",
        #                                                  color="Type")
        # st.plotly_chart(fig, use_container_width=True)

        # self.show_color(dataset, style = 'API')


    def barchart_plotly_one(self, dataset, color, requirements='False'):

        if requirements is True:
            fig = go.Figure(data=[go.Bar(
                x=(dataset['width'].cumsum() - dataset['width'] / 2).tolist(),
                y=dataset['length'],
                width=(dataset['width']).tolist(),  # customize width here
                marker_color=color,
                opacity=0.8,
                customdata=dataset['part'].tolist(),
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


    def distplot_plotly(self, dataset, x_column, y_column, color):

        fig = px.histogram(dataset, x=x_column, color=color,
                           marginal="box",  # or violin, rug
                           hover_data=dataset.columns)

        return fig


    def show_color(self, dataset, style):

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

    def make_shapely_figure(self, width, length, hole_from_start, radius_hole, 
        has_holes = True, hole_from_edge = 10, dis_between_holes = 300):


        fig = plt.figure(figsize=(10,10))


        ring = LinearRing([(0, 0), (length, 0), (length, width), (0, width), (0, 0)])

        plot_line(ring, add_points=True, color='blue', alpha=0.7)
        # plot_points(ring, color='gray', alpha=0.7)
        # st.write(ring)

        if has_holes:
            wood_left = hole_from_start

            while wood_left < length:

                # st.write(wood_left)

                ring = LinearRing([(wood_left-radius_hole/2, hole_from_edge-radius_hole/2), 
                                (wood_left+radius_hole/2, hole_from_edge-radius_hole/2), 
                                (wood_left+radius_hole/2, hole_from_edge+radius_hole/2), 
                                (wood_left-radius_hole/2, hole_from_edge+radius_hole/2), 
                                (wood_left-radius_hole/2, hole_from_edge-radius_hole/2)])
                # plot_line(ring, add_points=False, color='blue', alpha=0.7)
                plot_points(ring, color='gray', alpha=0.7)
                wood_left += dis_between_holes


        st.pyplot(fig)

    def make_ifc_object(self, width, length, height, has_holes = True, hole_from_start = 150, 
        radius_hole = 5, hole_from_edge = 10, dis_between_holes = 300):

        ### Example for loading an IFC file
        # # model = ifcopenshell.open('/Users/jerome/Downloads/AC20-FZK-Haus.ifc')
        # # st.write(model.schema) 
        # # walls = model.by_type('IfcWall')
        # # st.write((len(walls)))


        # # If we plan to store 3D geometry in our IFC model, we have to setup
        # # a "Model" context.
        # model3d = run("context.add_context", model, context_type="Model")
        

        # # Now we setup the subcontexts with each of the geometric "purposes"
        # # we plan to store in our model. "Body" is by far the most important
        # # and common context, as most IFC models are assumed to be viewable
        # # in 3D.
        # body = run("context.add_context", model,
        #     context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model3d)

        # # The 3D Axis subcontext is important if any "axis-based" parametric
        # # geometry is going to be created. For example, a beam, or column
        # # may be drawn using a single 3D axis line, and for this we need an
        # # Axis subcontext.
        # run("context.add_context", model,
        #     context_type="Model", context_identifier="Axis", target_view="GRAPH_VIEW", parent=model3d)

        # # The 3D Box subcontext is useful for clash detection or shape
        # # analysis, or even lazy-loading of large models.
        # run("context.add_context", model,
        #     context_type="Model", context_identifier="Box", target_view="MODEL_VIEW", parent=model3d)



        # Let's create a new project using millimeters with a single furniture element at the origin.
        model = run("project.create_file")
        run("root.create_entity", model, ifc_class="IfcProject")
        run("unit.assign_unit", model)

        # We want our representation to be the 3D body of the element.
        # This representation context is only created once per project.
        # You must reuse the same body context every time you create a new representation.
        model3d = run("context.add_context", model, context_type="Model")
        body = run("context.add_context", model,
            context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model3d)

        # Create our element with an object placement.
        element = run("root.create_entity", model, ifc_class="IfcFurniture")
        run("geometry.edit_object_placement", model, product=element)


        

        # # Rectangles (or squares) are typically used for concrete columns and beams
        # profile = model.create_entity("IfcRectangleProfileDef", ProfileName="600x300", ProfileType="AREA",
        #     XDim=600, YDim=300)

        

        ### Example building profile

        builder = ifcopenshell.util.shape_builder.ShapeBuilder(model)

        length = float(length)
        width = float(width)
        height = float(height)

        outer_curve = builder.polyline([(0.,0.), (length,0.), (length,width), (0.,width), (0.,0.)],
            #arc_points=[1], 
            closed=True)


        inner_curves = []
        if has_holes:

            wood_left = hole_from_start
            while wood_left < length:

                inner_curves.append(builder.circle((float(wood_left), float(hole_from_edge)), radius=float(radius_hole)))
                wood_left += dis_between_holes

        profile = builder.profile(outer_curve, inner_curves=inner_curves, name="Arbitrary")
        # st.write(profile)

        ### end example

        # # Create our element type. Types do not have an object placement.
        element_type = run("root.create_entity", model, ifc_class="IfcFurnitureType")

        # A profile-based representation, 1 meter long
        representation_model = run("geometry.add_profile_representation", model, context=body, profile=profile, depth=1)

        direction = model.createIfcDirection((0., 0., 1.))
        extrusion = model.createIfcExtrudedAreaSolid(SweptArea=profile, ExtrudedDirection=direction, Depth=height)
        body = ifcopenshell.util.representation.get_context(model, "Model", "Body", "MODEL_VIEW")
        representation_model = model.createIfcShapeRepresentation(
            ContextOfItems=body, RepresentationIdentifier="Body", RepresentationType="SweptSolid", Items=[extrusion])


        ### Example extruded object

        # rectangle = model.createIfcRectangleProfileDef(ProfileType="AREA", XDim=500, YDim=250)
        # direction = model.createIfcDirection((0., 0., 1.))

        # extrusion = model.createIfcExtrudedAreaSolid(SweptArea=rectangle, ExtrudedDirection=direction, Depth=500)
        # body = ifcopenshell.util.representation.get_context(model, "Model", "Body", "MODEL_VIEW")
        # representation_model = model.createIfcShapeRepresentation(
        #     ContextOfItems=body, RepresentationIdentifier="Body", RepresentationType="SweptSolid", Items=[extrusion])

        ### End Example



        #### Example round curved object 

        # builder = ifcopenshell.util.shape_builder.ShapeBuilder(model)

        # # Sweep a 10mm radius disk along a polyline with a couple of straight segments and an arc.
        # curve = builder.polyline(
        #     [(0., 0., 0.), (100., 0., 0.), (171., 29., 0.), (200., 100., 0.), (200., 200., 0.)],
        #     arc_points=[2])
        # swept_curve = builder.create_swept_disk_solid(curve, 10)

        # # Create a body representation
        # body = ifcopenshell.util.representation.get_context(model, "Model", "Body", "MODEL_VIEW")
        # representation_model = builder.get_representation(body, swept_curve)


        ### End Example

        # Assign our representation to the element type.
        run("geometry.assign_representation", model, product=element_type, representation=representation_model)

        # Create our element occurrence with an object placement.
        element = run("root.create_entity", model, ifc_class="IfcFurniture")
        run("geometry.edit_object_placement", model, product=element)

        # Assign our furniture occurrence to the type.
        # That's it! The representation will automatically be mapped!
        run("type.assign_type", model, related_object=element, relating_type=element_type)


        # # The shape_builder module depends on mathutils
        # from ifcopenshell.util.shape_builder import V

        # builder = ifcopenshell.util.shape_builder.ShapeBuilder(model)

        # # Parameters to define our table
        # width = 1200
        # depth = 700
        # height = 750
        # leg_size = 50.0
        # thickness = 50.0

        # # Extrude a rectangle profile for the tabletop
        # rectangle = builder.rectangle(size=V(width, depth))
        # tabletop = builder.extrude(builder.profile(rectangle), thickness, V(0, 0, height - thickness))

        # # Create a table leg curve, mirror it along two axes, and extrude.
        # leg_curve = builder.rectangle(size=V(leg_size, leg_size))
        # legs_curves = [leg_curve] + builder.mirror(
        #     leg_curve,
        #     mirror_axes=[V(1, 0), V(0, 1), V(1, 1)],
        #     mirror_point=V(width / 2, depth / 2),
        #     create_copy=True,
        # )
        # legs_profiles = [builder.profile(leg) for leg in legs_curves]
        # legs = [builder.extrude(leg, height - thickness) for leg in legs_profiles]

        # # Shift our table such that the object origin is in the center.
        # items = [tabletop] + legs
        # shift_to_center = V(-width / 2, -depth / 2)
        # builder.translate(items, shift_to_center.to_3d())

        # # Create a body representation
        # body = ifcopenshell.util.representation.get_context(model, "Model", "Body", "MODEL_VIEW")
        # representation = builder.get_representation(context=body, items=items)
        return model
        
