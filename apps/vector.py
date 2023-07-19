import os, sys
import geopandas as gpd
import streamlit as st
import folium
from pathlib import Path

# import inrix_npmrds_processing as inp
sys.path.append('apps')
import inrix_npmrds_functions as inr


def save_uploaded_file(file_content, file_name):
    """
    Save the uploaded file to a temporary directory
    """
    import tempfile
    import os
    import uuid

    _, file_extension = os.path.splitext(file_name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

    with open(file_path, "wb") as file:
        file.write(file_content.getbuffer())

    return file_path


def export_geopackage(gdf, name):
    """
    Save the file in memory to a temporary directory to recall later in defined format (gpkg)
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir, f"{name}.gpkg")
        gdf.to_file(path, driver="GPKG")
        return path.read_bytes()


@st.cache_data(persist=True)
def process_inrix_pipeline(input_data_folder,road_str,chunk_size, npmrds_geodata_path):
    """
    Run the relaibility computation using methods loaded from inrix_npmrds_processing.py file
    Data should include shapefile/zip of INRIX segements as well as probe data from RITIS website
    Arguments include:
        - input_data_folder: temporary location where loaded probe data is saved to 
        - road_str: defined selection for filtering on specific roadways
        - chunk_size: data chunk size for batch processing
        - npmrds_geodata_path:temporary location where loaded shapefile/zip of INRIX segements
    """
    test = inr.read_batch_of_raw_data(input_data_folder,road_str,chunk_size)
    main_data = test['main_data']
    main_tmc_data = test['main_tmc_data']

    main_data = inr.fix_datetime_columns(main_data)
    main_data = inr.define_standard_fhwa_timeslots(main_data)
    all_summaries_concat = inr.filter_group_summarize_fhwa(main_data)

    reliability_summaries_all = inr.calculate_standard_reliability(all_summaries_concat, main_tmc_data)
    summarized_data_with_geoms = inr.add_geometries_to_summaries(reliability_summaries_all, npmrds_geodata_path)

    return summarized_data_with_geoms


def app():
    """
    Main application code.
    Defined number of columns and size for splitting main view 
    """

    st.title("INRIX Reliability Score Generator")

    row1_col1, row1_col2 = st.columns([2, 1])
    width = 880
    height = 600

    with row1_col2:
        """
        
        """

        import leafmap.foliumap as leafmap
        
        basemaps = {
        "Roadmap": "ROADMAP",
        "Terrain": "TERRAIN",
        "Satellite": "SATELLITE",
        "Hybrid": "HYBRID",
        }

        selected_basemap = st.sidebar.radio("Select Basemap", list(basemaps.keys()))

        st.markdown(
            """<style>
        div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 24px;
            # font-weight: bold;
        }
            </style>
            """, unsafe_allow_html=True)

        url = st.text_input(
            "Enter a URL to a vector dataset",

        )

        data = st.file_uploader(
            "Upload a vector dataset", type=["geojson", "kml", "zip", "tab", "gpkg"]
        )

        container = st.container()

        if data or url:
            if data:
                file_path = save_uploaded_file(data, data.name)
                layer_name = os.path.splitext(data.name)[0]
            elif url:
                file_path = url
                layer_name = url.split("/")[-1].split(".")[0]

            with row1_col1:
                if file_path.lower().endswith(".kml"):
                    gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
                    gdf = gpd.read_file(file_path, driver="KML")
                else:
                    gdf = gpd.read_file(file_path)
                lon, lat = leafmap.gdf_centroid(gdf)
                
                # column_names = gdf.select_dtypes(include='number').columns.values.tolist()
                # selected_variable = None

                # with container:
                #     random_color = st.checkbox("Apply Symbology", False)
                #     if random_color:
                #         selected_variable = st.selectbox(
                #                 "Select a column to apply colors", column_names
                #             )
                #         # selected_variable = st.sidebar.selectbox("Select Variable for Legend", column_names)
                #         # min_value = gdf[selected_variable].min()
                #         # max_value = gdf[selected_variable].max()
                # # m = leafmap.Map(center=(40, -100))

                m = leafmap.Map(center=(lat, lon))
                folium.TileLayer("Stamen Terrain", show=False).add_to(m)               
                m.add_basemap(basemaps[selected_basemap])
                m.add_gdf(gdf, layer_name=layer_name)
                m.zoom_to_gdf(gdf)

                # if random_color == True and selected_variable != None:
                #     gdf.explore(
                #     m = m,
                #     column=selected_variable,  # make column
                #     scheme="naturalbreaks",  # use mapclassify's natural breaks scheme
                #     legend=True,  # show legend
                #     k=5,  # use 10 bins
                #     tooltip=False,  # hide tooltip
                #     popup=[selected_variable],  # show popup (on-click)
                #     legend_kwds=dict(colorbar=False),  # do not use colorbar
                #     )
                #     m.add_legend(title=selected_variable, labels=[min_value, max_value])
                m.to_streamlit(width=width, height=height)

        else:
            with row1_col1:
                m = leafmap.Map()
                folium.TileLayer("Stamen Terrain", show=False).add_to(m)
                # folium.TileLayer("Stamen Watercolor", show=False).add_to(m)
                m.add_basemap(basemaps[selected_basemap])
                m.to_streamlit(width=width, height=height)

        probe = st.file_uploader("Upload INRIX dataset", type=["zip"])
        if probe is not None:
            probe_path = save_uploaded_file(probe, probe.name)


        if 'stage' not in st.session_state:
            st.session_state.stage = 0

        def set_stage(stage):
            st.session_state.stage = stage
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Process INRIX", on_click=set_stage, args=(1,)):
                if probe is not None:
                    road_str = ''
                    chunk_size = 100000
                    summarized_data_with_geoms = process_inrix_pipeline(probe_path,road_str,chunk_size,file_path)
                    if st.session_state.stage > 0:
                        with col2:
                            if summarized_data_with_geoms is not None:                                
                                final = gpd.GeoDataFrame(summarized_data_with_geoms, geometry=summarized_data_with_geoms.geometry, crs="EPSG:4326")
                                output_name = Path(probe.name).stem
                                st.download_button("Download GeoPackage", data=export_geopackage(final, name=output_name), file_name=f"{output_name}.gpkg", on_click=set_stage, args=(0,))