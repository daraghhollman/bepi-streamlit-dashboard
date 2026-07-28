import datetime as dt
from pathlib import Path
from typing import Dict, List, Literal

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import planetary_coverage as pc
import spiceypy as spice
import streamlit as st
import xarray as xr
from astropy.table import QTable
from astropy.time import Time
from hermpy.data import rotate_to_aberrated_coordinates
from hermpy.utils import Constants
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, HourLocator

BLACK = "#000000"
RED = "#D55E00"
GREEN = "#009E73"
BLUE = "#0072B2"
YELLOW = "#F0E442"
PINK = "#CC79A7"
LIGHTBLUE = "#56B4E9"
ORANGE = "#E69F00"


def run():

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

    # Set some default times
    if "start_time" not in st.session_state:
        st.session_state.start_time = dt.datetime(2027, 5, 1)
    if "end_time" not in st.session_state:
        st.session_state.end_time = dt.datetime(2027, 5, 2)

    with st.container(border=True, horizontal=True):
        start_time = st.datetime_input(
            "Prediction Start",
            value=st.session_state.start_time,
            key="start_time",
            label_visibility="collapsed",
        )
        end_time = st.datetime_input(
            "Prediction End",
            value=st.session_state.end_time,
            key="end_time",
            label_visibility="collapsed",
        )
        st.button("◀", help="Previous", on_click=shift_range, args=(-1,))
        st.button("▶", help="Next", on_click=shift_range, args=(1,))

    mpo_positions = get_positions(start_time, end_time, "MPO")
    mpo_probabilities = get_probabilities(mpo_positions, maps)

    mio_positions = get_positions(start_time, end_time, "Mio")
    mio_probabilities = get_probabilities(mio_positions, maps)

    plot_probabilities(mpo_probabilities, mio_probabilities)


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
    fig.patch.set_facecolor("none")

    # Update plot theming based on light/dark mode
    text_colour = "black" if st.context.theme.type == "light" else "white"
    matplotlib.rcParams["text.color"] = text_colour
    matplotlib.rcParams["axes.labelcolor"] = text_colour
    matplotlib.rcParams["xtick.color"] = text_colour
    matplotlib.rcParams["ytick.color"] = text_colour
    matplotlib.rcParams["axes.edgecolor"] = text_colour

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


@st.cache_data()
def get_positions(
    start: dt.datetime,
    end: dt.datetime,
    spacecraft: Literal["MPO", "Mio"],
) -> QTable:

    kernels_dir = Path("./data/spice/kernels/")
    metakernel_path = kernels_dir / "mk" / "bc_plan.tm"
    mk = pc.MetaKernel(metakernel_path, kernels=kernels_dir)

    with spice.KernelPool(mk):
        resolution = dt.timedelta(minutes=1)
        times = [
            start + i * resolution for i in range(round((end - start) / resolution))
        ]
        ets = spice.datetime2et(times)

        positions, _ = spice.spkpos(
            spacecraft,
            ets,
            "BC_MSO",
            "NONE",
            "MERCURY",
        )

        positions *= u.km

        # We want the positions in MSM' coordinates, not MSO', and must add
        # 479 km to Z.
        positions[:, 2] += Constants.DIPOLE_OFFSET_RADII

        # Convert to radii
        positions = positions.to(Constants.MERCURY_RADIUS)

        positions_table = QTable(
            positions,
            names=["X MSM", "Y MSM", "Z MSM"],
        )

        positions_table["UTC"] = Time(times)

        positions_table = rotate_to_aberrated_coordinates(positions_table)

        return positions_table[["UTC", "X MSM'", "Y MSM'", "Z MSM'"]]


def get_probabilities(
    positions: QTable, probability_maps: xr.Dataset
) -> Dict[str, List[float]]:

    x_data = positions["X MSM'"].value
    cyl_data = np.sqrt(positions["Y MSM'"] ** 2 + positions["Z MSM'"] ** 2).value

    regions = ["Solar Wind", "Magnetosheath", "Magnetosphere"]

    trajectory_probabilities = {
        region: np.zeros_like(x_data, dtype=float) for region in regions
    }

    bin_size = probability_maps.coords["X"][1] - probability_maps.coords["X"][0]

    x_coords = probability_maps.coords["X"].values
    cyl_coords = probability_maps.coords["CYL"].values

    # Create bin edges (add one extra edge at the end for np.digitize)
    # Resolves floating point precision issues
    x_bins = np.concatenate([x_coords, [x_coords[-1] + bin_size]])
    cyl_bins = np.concatenate([cyl_coords, [cyl_coords[-1] + bin_size]])

    # Digitize the trajectory data into bin indices
    x_indices = np.digitize(x_data, x_bins) - 1
    cyl_indices = np.digitize(cyl_data, cyl_bins) - 1

    # Iterate over trajectory points and assign probabilities
    for i in range(len(x_data)):
        x_index = x_indices[i]
        cyl_index = cyl_indices[i]

        # Ensure the index is within the valid histogram range
        if 0 <= x_index < len(x_bins) - 1 and 0 <= cyl_index < len(cyl_bins) - 1:
            for region in regions:
                trajectory_probabilities[region][i] = probability_maps[
                    f"{region.replace(' ', '_').lower()}_mean"
                ][x_index, cyl_index]

        else:
            for region in regions:
                trajectory_probabilities[region][
                    i
                ] = np.nan  # Assign NaN if out of bounds

    trajectory_probabilities["UTC"] = positions["UTC"].to_datetime()

    return trajectory_probabilities


def plot_probabilities(
    mpo_probabilities: Dict[List[float]], mio_probabilities: Dict[List[float]]
) -> None:

    fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    fig.patch.set_facecolor("none")

    # Update plot theming based on light/dark mode
    text_colour = "black" if st.context.theme.type == "light" else "white"
    matplotlib.rcParams["text.color"] = text_colour
    matplotlib.rcParams["axes.labelcolor"] = text_colour
    matplotlib.rcParams["xtick.color"] = text_colour
    matplotlib.rcParams["ytick.color"] = text_colour
    matplotlib.rcParams["axes.edgecolor"] = text_colour

    matplotlib.rcParams["legend.facecolor"] = "none"
    matplotlib.rcParams["legend.edgecolor"] = "none"

    ax = axes[0]
    ax: Axes

    ax.plot(
        mio_probabilities["UTC"],
        mio_probabilities["Solar Wind"],
        color=YELLOW,
        label="Solar Wind",
    )
    ax.plot(
        mio_probabilities["UTC"],
        mio_probabilities["Magnetosheath"],
        color=ORANGE,
        label="Magnetosheath",
    )
    ax.plot(
        mio_probabilities["UTC"],
        mio_probabilities["Magnetosphere"],
        color=LIGHTBLUE,
        label="Magnetosphere",
    )

    ax.set_ylabel("Mio\nRegion Probabilities")

    ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.25))

    ax = axes[1]
    ax: Axes

    ax.plot(
        mpo_probabilities["UTC"],
        mpo_probabilities["Solar Wind"],
        color=YELLOW,
        label="Solar Wind",
    )
    ax.plot(
        mpo_probabilities["UTC"],
        mpo_probabilities["Magnetosheath"],
        color=ORANGE,
        label="Magnetosheath",
    )
    ax.plot(
        mpo_probabilities["UTC"],
        mpo_probabilities["Magnetosphere"],
        color=LIGHTBLUE,
        label="Magnetosphere",
    )

    ax.set_xlabel("UTC")
    ax.set_ylabel("MPO\nRegion Probabilities")

    for ax in axes:
        ax.patch.set_facecolor("none")
        ax.margins(x=0)

        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.xaxis.set_major_locator(HourLocator(byhour=[0, 6, 12, 18]))

    st.pyplot(fig)


def shift_range(direction):
    span = st.session_state.end_time - st.session_state.start_time
    delta = direction * span
    st.session_state.start_time += delta
    st.session_state.end_time += delta


run()
