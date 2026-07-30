import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

import astropy.units as u
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import planetary_coverage as pc
import spiceypy as spice
import streamlit as st
import xarray as xr
from astropy.table import QTable
from astropy.time import Time
from get_probabilities import get_probability_at_position
from hermpy.data import rotate_to_aberrated_coordinates
from hermpy.plotting import plot_magnetospheric_boundaries
from hermpy.utils import Constants
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.dates import DateFormatter, HourLocator, date2num, num2date
from matplotlib.patches import Circle, Rectangle

BLACK = "#000000"
RED = "#D55E00"
GREEN = "#009E73"
BLUE = "#0072B2"
YELLOW = "#F0E442"
PINK = "#CC79A7"
LIGHTBLUE = "#56B4E9"
ORANGE = "#E69F00"


def run():

    maps = load_probability_maps(author="Hollman+ 2026")
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

    smoothing: int = st.slider("Smooth Factor", 1, 20, 10)

    mio_positions, mpo_positions = get_positions(start_time, end_time)

    mio_probabilities = get_probability_at_position(mio_positions, maps)
    mpo_probabilities = get_probability_at_position(mpo_positions, maps)

    plot_probabilities(
        mpo_positions["UTC"].to_datetime(),
        mpo_probabilities,
        mio_probabilities,
        mpo_positions,
        mio_positions,
        smooth_factor=smoothing,
    )


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


def plot_probability_maps(probability_maps: xr.Dataset) -> None:

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(10, 4))

    fig.patch.set_facecolor("none")

    # Update plot theming based on light/dark mode
    text_colour = "black" if st.context.theme.type == "light" else "white"
    matplotlib.rcParams["text.color"] = text_colour
    matplotlib.rcParams["axes.labelcolor"] = text_colour
    matplotlib.rcParams["xtick.color"] = text_colour
    matplotlib.rcParams["ytick.color"] = text_colour
    matplotlib.rcParams["axes.edgecolor"] = text_colour

    residence_data = probability_maps["Minutes In Bin"] / 60

    # Plot residence time using a log scale to show both low and high coverage areas
    residence_mesh = axes[0].pcolormesh(
        residence_data.coords["X MSM'"],
        residence_data.coords["CYL MSM'"],
        residence_data.values.T,
        cmap="viridis",
        norm="log",
    )

    axes[0].set_facecolor("none")

    # Create a mask showing where the spacecraft has any residence time at all
    residence_mask = residence_data.values.T != 0

    # Configure residence panel
    axes[0].set_title("MESSENGER\nResidence")

    # Add a colorbar below the residence panel
    cbar_bounds = [0, -0.6, 1, 0.1]
    cbar_ax = axes[0].inset_axes(cbar_bounds)
    plt.colorbar(
        residence_mesh,
        cax=cbar_ax,
        location="bottom",
        label="Time Spent [hours]",
    )

    # Y-axis label for the first panel only
    axes[0].set_ylabel(
        r"$\left(Y_{\rm MSM'}^2 + Z_{\rm MSM'}^2 \right)^{0.5}\quad \left[ R_{\rm M} \right]$"
    )

    regions = ["Solar Wind", "Magnetosheath", "Magnetosphere"]

    for i, ax in enumerate(axes[1:]):
        # Select region probability map
        map_data = probability_maps[regions[i]]

        # Hide zeros so unobserved regions don't appear as true 0 probability
        map_data.values[map_data.values == 0] = np.nan

        # Plot region probability (0 to 1)
        mesh = ax.pcolormesh(
            map_data.coords["X MSM'"],
            map_data.coords["CYL MSM'"],
            map_data.values.T,
            vmax=1,
            cmap="magma",
        )

        # Draw the outline of MESSENGER residence coverage
        ax.contour(
            map_data.coords["X MSM'"],
            map_data.coords["CYL MSM'"],
            residence_mask,
            levels=[0.5],
            antialiased=False,
            colors="grey",
            zorder=-1,
        )

        # Shade outside residence coverage in light grey
        ax.contourf(
            map_data.coords["X MSM'"],
            map_data.coords["CYL MSM'"],
            residence_mask,
            levels=[0, 0.5, 1],
            colors=["none", "lightgrey"],
            zorder=-2,
        )

        ax.set_facecolor("none")

        # Title each region panel
        ax.set_title(regions[i])

        # Add a shared region probability colorbar under the middle region panel
        if i == 1:
            cbar_bounds = [-1.2, -0.6, 3.4, 0.1]
            cbar_ax = ax.inset_axes(cbar_bounds)
            plt.colorbar(
                mesh,
                cax=cbar_ax,
                location="bottom",
                label="Relative Region Occurence",
            )

    # ----------------------------
    # Shared axis formatting
    # ----------------------------

    labels = ["a", "b", "c", "d"]

    for i, ax in enumerate(axes):
        # Ensure equal scaling for x/y to preserve geometry
        ax.set_aspect("equal")

        # Set consistent axis limits across all panels
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 8)

        # Shared x-axis label
        ax.set_xlabel(r"$X_{\rm MSM'} \quad \left[ R_{\rm M} \right]$")

        # Draw Mercury boundaries (north/south extent) using alternating circle segments
        mercury_params = {"segments": 30, "linewidth": 1.5}

        draw_alternating_circle(
            (0, Constants.DIPOLE_OFFSET / Constants.MERCURY_RADIUS),
            1,
            ax,
            **mercury_params,
        )
        draw_alternating_circle(
            (0, -(Constants.DIPOLE_OFFSET / Constants.MERCURY_RADIUS)),
            1,
            ax,
            **mercury_params,
        )

        # Add panel label in axes coordinates
        ax.text(0.02, 0.9, f"({labels[i]})", transform=ax.transAxes)

    # ----------------------------
    # Custom legend
    # ----------------------------

    custom_handles = [
        CurvedLegendHandle(angle=180),
        Rectangle((0, 0), 1, 1, facecolor="lightgrey", edgecolor="grey"),
    ]

    fig.legend(
        custom_handles,
        ["Mercury (northern and southern extent)", "MESSENGER Residence Bounds"],
        loc="upper center",
        ncol=1,
        frameon=False,
        handler_map={CurvedLegendHandle: custom_handles[0]},
        bbox_to_anchor=(0.6, 0.95),
    )

    # Adjust spacing and save
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.2)

    st.pyplot(fig)
    plt.close(fig)


@st.cache_data
def get_positions(
    start: dt.datetime,
    end: dt.datetime,
) -> Tuple[QTable, QTable]:

    kernels_dir = Path("./data/spice/kernels/")
    metakernel_path = kernels_dir / "mk" / "bc_plan.tm"
    mk = pc.MetaKernel(metakernel_path, kernels=kernels_dir)

    tables: List[QTable] = []

    with spice.KernelPool(mk):
        resolution = dt.timedelta(minutes=1)
        times = [
            start + i * resolution for i in range(round((end - start) / resolution))
        ]
        ets = spice.datetime2et(times)

        for spacecraft in ["Mio", "MPO"]:
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

            positions_table = positions_table[["UTC", "X MSM'", "Y MSM'", "Z MSM'"]]

            tables.append(positions_table)

        return tuple(tables)


@dataclass
class RegionProbabilities:
    mean: List[float]
    upper: List[float]
    lower: List[float]


@dataclass
class TrajectoryProbabilities:
    regions: List[str]
    probabilities: List[RegionProbabilities]


def plot_probabilities(
    time: List[dt.datetime],
    mpo_probabilities: Tuple[List[List[float]], List[List[float]], List[List[float]]],
    mio_probabilities: Tuple[List[List[float]], List[List[float]], List[List[float]]],
    mpo_positions: QTable,
    mio_positions: QTable,
    smooth_factor: int = 1,
) -> None:

    fig, axes = plt.subplots(2, 2, width_ratios=[3, 1], figsize=(10, 6), sharex="col")
    fig.patch.set_facecolor("none")

    for ax in axes.flatten():
        ax.set_facecolor("none")

    # Update plot theming based on light/dark mode
    text_colour = "black" if st.context.theme.type == "light" else "white"
    matplotlib.rcParams["text.color"] = text_colour
    matplotlib.rcParams["axes.labelcolor"] = text_colour
    matplotlib.rcParams["xtick.color"] = text_colour
    matplotlib.rcParams["ytick.color"] = text_colour
    matplotlib.rcParams["axes.edgecolor"] = text_colour

    matplotlib.rcParams["legend.facecolor"] = "none"
    matplotlib.rcParams["legend.edgecolor"] = "none"

    regions = ["Solar Wind", "Magnetosheath", "Magnetosphere"]
    colours = [YELLOW, ORANGE, LIGHTBLUE]

    def moving_average(data, window_size):
        weights = np.ones(window_size) / window_size

        return np.convolve(data, weights, mode="valid")

    time = num2date(moving_average(date2num(time), smooth_factor))

    for i, region in enumerate(regions):

        for j, probabilities in enumerate([mio_probabilities, mpo_probabilities]):

            mean = moving_average(probabilities[0][i], smooth_factor)
            lower = moving_average(probabilities[1][i], smooth_factor)
            upper = moving_average(probabilities[2][i], smooth_factor)

            axes[j, 0].plot(time, mean, color=colours[i], label=f"P({region})")

            axes[j, 0].fill_between(
                time,
                lower,
                upper,
                color=colours[i],
                alpha=0.3,
            )

    mio_positions["CYL MSM'"] = np.sqrt(
        mio_positions["Y MSM'"] ** 2 + mio_positions["Z MSM'"] ** 2
    )
    mpo_positions["CYL MSM'"] = np.sqrt(
        mpo_positions["Y MSM'"] ** 2 + mpo_positions["Z MSM'"] ** 2
    )

    axes[0, 1].plot(
        mio_positions["X MSM'"], mio_positions["CYL MSM'"], color=text_colour
    )
    axes[1, 1].plot(
        mpo_positions["X MSM'"], mpo_positions["CYL MSM'"], color=text_colour
    )

    axes[1, 1].set_xlabel(r"$X_{\rm MSM'} \quad \left[ R_{\rm M} \right]$")

    axes[0, 0].set_ylabel("Mio\nRegion Probability")
    axes[1, 0].set_ylabel("MPO\nRegion Probability")

    for ax in axes[:, 0]:
        ax: Axes

        ax.margins(x=0)

        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.xaxis.set_major_locator(HourLocator(byhour=[0, 6, 12, 18]))

    for ax in axes[:, 1]:

        plot_magnetospheric_boundaries(ax, color=text_colour)

        ax.set_ylabel(
            r"$\left( Y_{\rm MSM'}^2 + Z_{\rm MSM'}^2 \right)^{0.5} \quad \left[ R_{\rm M} \right]$"
        )

        ax.set_aspect("equal")

        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 8)

        # Add Mercury
        circle = Circle(
            (0, Constants.DIPOLE_OFFSET / Constants.MERCURY_RADIUS),
            1,
            edgecolor="grey",
            facecolor="none",
            linewidth=2,
            zorder=-5,
        )
        ax.add_patch(circle)
        circle = Circle(
            (0, -1 * Constants.DIPOLE_OFFSET / Constants.MERCURY_RADIUS),
            1,
            edgecolor="grey",
            facecolor="none",
            linewidth=2,
            zorder=-5,
        )
        ax.add_patch(circle)

    axes[0, 0].legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.3))

    st.pyplot(fig)
    plt.close(fig)


def shift_range(direction):
    span = st.session_state.end_time - st.session_state.start_time
    delta = direction * span
    st.session_state.start_time += delta
    st.session_state.end_time += delta


def draw_alternating_circle(center, radius, ax, segments=200, linewidth=3, **kwargs):
    """
    Draw a circle on a Matplotlib axis using alternating black/white line segments.

    This is used to visually represent Mercury's boundary with a distinctive
    alternating pattern.

    Parameters
    ----------
    center : tuple[float, float]
        (x, y) center of the circle in data coordinates.
    radius : float
        Circle radius in data units.
    ax : matplotlib.axes.Axes
        Axis to draw onto.
    segments : int, default=200
        Number of segments used to approximate the circle. More segments gives
        a smoother circle.
    linewidth : float, default=3
        Width of the circle segments.
    **kwargs
        Additional keyword arguments passed to `matplotlib.collections.LineCollection`.
    """
    x0, y0 = center

    theta = np.linspace(0, 2 * np.pi, segments)
    x = x0 + radius * np.cos(theta)
    y = y0 + radius * np.sin(theta)

    points = np.column_stack((x, y))
    segments_list = np.stack([points[:-1], points[1:]], axis=1)

    colors = ["black" if i % 2 == 0 else "white" for i in range(len(segments_list))]

    lc = LineCollection(segments_list, colors=colors, linewidths=linewidth, **kwargs)
    ax.add_collection(lc)


class CurvedLegendHandle:
    """
    Custom legend handle that draws a curved arc symbol inside a Matplotlib legend.

    This is used to create a legend entry representing the curved Mercury boundary
    marker drawn in the panels.
    """

    def __init__(self, angle=180):
        """
        Parameters
        ----------
        angle : float, default=180
            Reserved for potential customization of the arc curvature/extent.
            (Currently the arc is drawn from 0 to 180 degrees.)
        """
        self.angle = angle

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        """
        Matplotlib hook to draw the custom legend artist.

        Returns
        -------
        matplotlib.patches.Arc
            The arc artist added to the legend handle box.
        """
        w, h = handlebox.width, handlebox.height
        x, y = handlebox.xdescent, handlebox.ydescent

        arc = mpatches.Arc(
            (x + w / 2, y + h / 2),
            w,
            h * 2,
            angle=0,
            theta1=0,
            theta2=180,
            lw=2,
            ls=(0, (2, 2)),
            color="black",
            transform=handlebox.get_transform(),
        )

        handlebox.add_artist(arc)

        arc = mpatches.Arc(
            (x + w / 2, y + h / 2),
            w,
            h * 2,
            angle=0,
            theta1=0,
            theta2=180,
            lw=2,
            ls=(2, (2, 2)),
            color="white",
            transform=handlebox.get_transform(),
        )

        handlebox.add_artist(arc)
        return arc


run()
