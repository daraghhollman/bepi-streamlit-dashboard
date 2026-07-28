from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import plotly.subplots
import streamlit as st
import xarray as xr


def app():

    map_selection: str = st.selectbox(
        "Source crossings list",
        ("Hollman+ 2026", "Philpott+ 2020"),
    )

    with st.container(border=True, horizontal=True):
        density_selection: int = st.slider("Smoothness", 1, 10, 1)
        something_else: int = st.slider("Something Else", 1, 10, 1)

    maps = load_probability_maps(author=map_selection)
    maps = process_probability_maps(
        maps, author=map_selection, grid_density=density_selection
    )
    plot_probability_maps(maps)


# Math mathjax available for streamlit
with open("load-mathjax.js", "r") as f:
    js = f.read()
    st.components.v1.html(f"<script>{js}</script>", height=0)


@st.cache_resource
def load_probability_maps(
    author: Literal["Hollman+ 2026", "Philpott+ 2020"],
) -> xr.Dataset:

    match author:
        case "Hollman+ 2026":
            probability_map = xr.load_dataset(Path("./data/region_maps_hollman.nc"))

        case "Philpott+ 2020":
            probability_map = xr.load_dataset(Path("./data/region_maps_philpott.nc"))

        case _:
            raise ValueError(f"No valid author: {author}")

    return probability_map


@st.cache_resource()
def process_probability_maps(
    _data: xr.Dataset,  # The underscore prefix means that this var is ignored in hashing
    # We include author to help with caching as it is hashable
    author: Literal["Hollman+ 2026", "Philpott+ 2020"],
    grid_density: float = 1,
) -> xr.Dataset:

    # Interpolate if grid_density is not one
    bin_size = _data.coords["X"][1] - _data.coords["X"][0]

    new_x_coords = np.arange(
        _data.coords["X"][0],
        _data.coords["X"][-1] + bin_size / grid_density,
        bin_size / grid_density,
    )
    new_cyl_coords = np.arange(
        _data.coords["CYL"][0],
        _data.coords["CYL"][-1] + bin_size / grid_density,
        bin_size / grid_density,
    )

    data = _data.interp(coords={"X": new_x_coords, "CYL": new_cyl_coords})

    return data


def plot_probability_maps(data: xr.Dataset) -> None:

    # Selected probability map figure
    regions = ["Solar Wind", "Magnetosheath", "Magnetosphere"]
    fig = plotly.subplots.make_subplots(rows=1, cols=3, subplot_titles=regions)

    for i, region in enumerate(regions):

        region_map = data[f"{region.replace(' ', '_').lower()}_mean"].T

        region_map.values[np.where(region_map.values == 0)] = np.nan

        fig.add_trace(
            go.Heatmap(
                z=region_map.values,
                x=region_map.coords["X"],
                y=region_map.coords["CYL"],
                colorscale="greys_r",
                showscale=(i == len(regions) - 1),
                colorbar=(
                    dict(
                        title=dict(text="Region Probability", side="right"),
                    )
                    if i == len(regions) - 1
                    else None
                ),
            ),
            row=1,
            col=i + 1,
        )

        # Add axis labels for each subplot
        fig.update_xaxes(
            title_text=r"$X_{\rm MSM'} \quad \left[ \text{R}_\text{M} \right]$",
            row=1,
            col=i + 1,
        )
        fig.update_yaxes(
            title_text=r"$\left( Y_{\text{MSM'}}^2 + Z_{\text{MSM'}}^2 \right)^{0.5} \quad \left[ \text{R}_\text{M} \right]$",
            row=1,
            col=1,
        )

        fig.update_layout(template="simple_white")
        fig.update_layout(plot_bgcolor="lightgrey")

        # Force equal aspect
        fig.update_yaxes(
            scaleanchor=f"x{i + 1}",
            scaleratio=1,
            row=1,
            col=i + 1,
        )

        # Remove margins
        fig.update_xaxes(range=(-5, 5), constrain="domain")
        fig.update_yaxes(range=(0, 10), constrain="domain")

    fig.update_layout(width=1700, height=400, autosize=False)


    st.plotly_chart(fig, use_container_width=False)


app()
