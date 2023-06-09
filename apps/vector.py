import os
import geopandas as gpd
import streamlit as st


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


def app():

    st.title("Upload Vector Data")

    row1_col1, row1_col2 = st.columns([2, 1])
    width = 950
    height = 600

    with row1_col2:

        import leafmap.deck as leafmap

        url = st.text_input(
            "Enter a URL to a vector dataset",

        )

        data = st.file_uploader(
            "Upload a vector dataset", type=["geojson", "kml", "zip", "tab"]
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
                column_names = gdf.columns.values.tolist()
                random_column = None

                # m = leafmap.Map(center=(40, -100))
                m = leafmap.Map(center=(lat, lon))
                m.add_gdf(gdf, random_color_column=random_column)
                st.pydeck_chart(m)

        else:
            with row1_col1:
                m = leafmap.Map()
                st.pydeck_chart(m)
