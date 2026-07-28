from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xarray as xr
from matplotlib.axes import Axes


def content():

    map_selection: str = st.selectbox(
        "Source crossings list",
        ("Hollman+ 2026", "Philpott+ 2020"),
    )

    with st.container(border=True, horizontal=True):
        density_selection: int = st.slider("Smoothness", 1, 10, 1)
        something_else: int = st.slider("Cool Factor", 1, 9000, 9000)

    maps = load_probability_maps(author=map_selection)
    maps = process_probability_maps(
        maps, author=map_selection, grid_density=density_selection
    )
    plot_probability_maps(maps)


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

    fig, axes = plt.subplots(1, 4, figsize=(6, 2), width_ratios=[1] * 3 + [0.05])

    for i, region in enumerate(regions):

        ax: Axes = axes[i]

        region_map = data[f"{region.replace(' ', '_').lower()}_mean"].T

        region_map.values[np.where(region_map.values == 0)] = np.nan

        mesh = ax.pcolormesh(
            region_map.coords["X"],
            region_map.coords["CYL"],
            region_map.values,
            cmap="grey",
        )

        ax.set_xlabel(r"$X_{\rm MSM'} \quad \left[ \text{R}_\text{M} \right]$")

        if i == 0:
            ax.set_ylabel(
                r"$\left( Y_{\text{MSM'}}^2 + Z_{\text{MSM'}}^2 \right)^{0.5} \quad \left[ \text{R}_\text{M} \right]$"
            )

        else:
            ax.set_yticklabels([])

        if i == 2:
            fig.colorbar(mesh, cax=axes[-1], label="Observation Probability")

        ax.set_facecolor("lightgrey")
        ax.set_aspect("equal")

        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 10)

    st.pyplot(fig)

content()
