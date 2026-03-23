from __future__ import annotations

"""Shared helpers for the interactive example notebooks.

The notebooks in this repository used to each define their own versions of the
same Plotly traces, slider widgets, and animation scaffolding.  Consolidating
those helpers here keeps the notebooks focused on the mathematics of each
example instead of repeating long blocks of UI boilerplate.
"""

from typing import Callable, Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

TimeFormatter = Callable[[float, float], str]

__all__ = [
    "configure_plotly",
    "line_trace",
    "center_trace",
    "circle_trace",
    "time_label",
    "make_particle_animation",
    "make_two_panel_particle_animation",
    "make_seed_grid_figure",
]


def configure_plotly(renderer: str = "notebook") -> None:
    """Set the default Plotly renderer used by the example notebooks."""
    pio.renderers.default = renderer


def line_trace(
    points: np.ndarray,
    name: str,
    color: str = "rgba(80,80,80,0.55)",
    dash: str = "dot",
    close: bool = False,
    showlegend: bool = True,
    marker_size: int = 4,
    mode: str = "lines",
    opacity: float = 0.8,
) -> go.Scatter:
    pts = np.asarray(points, dtype=float)
    if close:
        pts = np.vstack([pts, pts[0]])
    return go.Scatter(
        x=pts[:, 0],
        y=pts[:, 1],
        mode=mode,
        name=name,
        line=dict(color=color, dash=dash),
        marker=dict(size=marker_size, color=color),
        opacity=opacity,
        showlegend=showlegend,
        hoverinfo="skip",
    )


def center_trace(
    points: np.ndarray,
    name: str,
    color: str = "rgba(50,50,50,0.9)",
    symbol: str = "x",
    size: int = 10,
    showlegend: bool = True,
) -> go.Scatter:
    pts = np.asarray(points, dtype=float)
    return go.Scatter(
        x=pts[:, 0],
        y=pts[:, 1],
        mode="markers",
        name=name,
        marker=dict(
            symbol=symbol, size=size, color=color, line=dict(color=color, width=1)
        ),
        showlegend=showlegend,
    )


def circle_trace(
    center: Sequence[float],
    radius: float = 0.14,
    *,
    name: str = "target region",
    color: str = "rgba(80,80,80,0.45)",
    dash: str = "dash",
    showlegend: bool = True,
) -> go.Scatter:
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    center = np.asarray(center, dtype=float)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line=dict(color=color, dash=dash),
        showlegend=showlegend,
        hoverinfo="skip",
    )


def time_label(t: float, horizon: float) -> str:
    horizon = float(horizon)
    normalized = 0.0 if horizon == 0.0 else float(t) / horizon
    return f"t/T = {normalized:.3f} (t = {t:.2e})"


def _default_hover_text(masses: np.ndarray, *, mass_format: str) -> list[str]:
    return [
        f"particle {i}<br>mass = {format(float(m), mass_format)}"
        for i, m in enumerate(masses)
    ]


def make_particle_animation(
    positions: np.ndarray,
    times: np.ndarray,
    masses: np.ndarray,
    title: str,
    *,
    static_traces: Iterable[go.BaseTraceType] | None = None,
    marker_size: int = 10,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
    particle_name: str = "conditioned particles",
    mass_format: str = ".4f",
    time_formatter: TimeFormatter | None = None,
    slider_label_formatter: Callable[[float, float], str] | None = None,
    currentvalue_prefix: str = "time = ",
    colorbar_title: str = "mass",
    marker_colorscale: str = "Viridis",
    marker_line_width: float = 0.6,
    play_frame_duration: int = 80,
    width: int = 850,
    height: int = 720,
    margin: dict | None = None,
) -> go.Figure:
    static_traces = list(static_traces or [])
    masses = np.asarray(masses, dtype=float)
    times = np.asarray(times, dtype=float)
    horizon = float(times[-1]) if len(times) else 0.0
    time_formatter = time_formatter or (lambda t, h: f"time = {t:.6f}")
    slider_label_formatter = slider_label_formatter or (lambda t, h: f"{t:.6f}")
    hover_text = _default_hover_text(masses, mass_format=mass_format)

    particle_trace = go.Scatter(
        x=positions[0, :, 0],
        y=positions[0, :, 1],
        mode="markers",
        name=particle_name,
        marker=dict(
            size=marker_size,
            color=masses,
            colorscale=marker_colorscale,
            cmin=float(np.min(masses)),
            cmax=float(np.max(masses)),
            showscale=True,
            colorbar=dict(title=colorbar_title),
            line=dict(color="black", width=marker_line_width),
        ),
        text=hover_text,
        hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
    )

    fig = go.Figure(data=[*static_traces, particle_trace])
    particle_trace_index = len(static_traces)

    frames = []
    for k, t in enumerate(times):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=positions[k, :, 0], y=positions[k, :, 1], text=hover_text
                    )
                ],
                traces=[particle_trace_index],
                name=str(k),
                layout=go.Layout(
                    title_text=f"{title}<br><sup>{time_formatter(float(t), horizon)}</sup>"
                ),
            )
        )
    fig.frames = frames

    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(k)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
            label=slider_label_formatter(float(t), horizon),
        )
        for k, t in enumerate(times)
    ]

    layout_kwargs = dict(
        title=f"{title}<br><sup>{time_formatter(float(times[0]), horizon)}</sup>",
        width=width,
        height=height,
        template="simple_white",
        legend=dict(x=0.01, y=1.08, orientation="h"),
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": currentvalue_prefix},
                pad={"t": 20},
                steps=slider_steps,
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.01,
                y=1.16,
                xanchor="left",
                yanchor="top",
                direction="left",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {
                                    "duration": play_frame_duration,
                                    "redraw": True,
                                },
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    ),
                ],
            )
        ],
    )
    if margin is not None:
        layout_kwargs["margin"] = margin
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(range=list(x_range), title="x")
    fig.update_yaxes(range=list(y_range), title="y", scaleanchor="x", scaleratio=1)
    return fig


def make_two_panel_particle_animation(
    *,
    left_positions: np.ndarray,
    right_positions: np.ndarray,
    times: np.ndarray,
    color_values: np.ndarray,
    hover_text: Sequence[str],
    title: str,
    left_static_traces: Iterable[go.BaseTraceType] | None = None,
    right_static_traces: Iterable[go.BaseTraceType] | None = None,
    left_subplot_title: str = "Left",
    right_subplot_title: str = "Right",
    left_name: str = "left particles",
    right_name: str = "right particles",
    marker_size: int = 8,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
    colorscale: str = "HSV",
    cmin: float = 0.0,
    cmax: float = 1.0,
    colorbar_title: str | None = None,
    marker_line_width: float = 0.6,
    width: int = 980,
    height: int = 560,
    time_formatter: Callable[[float], str] | None = None,
) -> go.Figure:
    left_static_traces = list(left_static_traces or [])
    right_static_traces = list(right_static_traces or [])
    time_formatter = time_formatter or (lambda t: f"time = {t:.8f}")

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.08,
        subplot_titles=(left_subplot_title, right_subplot_title),
    )

    for trace in left_static_traces:
        fig.add_trace(trace, row=1, col=1)
    for trace in right_static_traces:
        fig.add_trace(trace, row=1, col=2)

    left_trace = go.Scatter(
        x=left_positions[0, :, 0],
        y=left_positions[0, :, 1],
        mode="markers",
        name=left_name,
        marker=dict(
            size=marker_size,
            color=color_values,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            line=dict(color="black", width=marker_line_width),
        ),
        text=list(hover_text),
        hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
    )
    right_marker = dict(
        size=marker_size,
        color=color_values,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        showscale=colorbar_title is not None,
        line=dict(color="black", width=marker_line_width),
    )
    if colorbar_title is not None:
        right_marker["colorbar"] = dict(title=colorbar_title)

    right_trace = go.Scatter(
        x=right_positions[0, :, 0],
        y=right_positions[0, :, 1],
        mode="markers",
        name=right_name,
        marker=right_marker,
        text=list(hover_text),
        hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
    )

    fig.add_trace(left_trace, row=1, col=1)
    left_trace_index = len(left_static_traces) + len(right_static_traces)
    fig.add_trace(right_trace, row=1, col=2)
    right_trace_index = left_trace_index + 1

    frames = []
    for k, t in enumerate(times):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=left_positions[k, :, 0],
                        y=left_positions[k, :, 1],
                        text=list(hover_text),
                    ),
                    go.Scatter(
                        x=right_positions[k, :, 0],
                        y=right_positions[k, :, 1],
                        text=list(hover_text),
                    ),
                ],
                traces=[left_trace_index, right_trace_index],
                name=str(k),
                layout=go.Layout(
                    title_text=f"{title}<br><sup>{time_formatter(float(t))}</sup>"
                ),
            )
        )
    fig.frames = frames

    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(k)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
            label=f"{float(t):.8f}",
        )
        for k, t in enumerate(times)
    ]

    fig.update_layout(
        title=f"{title}<br><sup>{time_formatter(float(times[0]))}</sup>",
        width=width,
        height=height,
        template="simple_white",
        legend=dict(x=0.01, y=1.08, orientation="h"),
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "time = "},
                pad={"t": 20},
                steps=slider_steps,
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.01,
                y=1.16,
                xanchor="left",
                yanchor="top",
                direction="left",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 60, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    ),
                ],
            )
        ],
    )

    for col in (1, 2):
        xaxis_name = "x" if col == 1 else f"x{col}"
        fig.update_xaxes(range=list(x_range), title="x", row=1, col=col)
        fig.update_yaxes(
            range=list(y_range),
            title="y",
            row=1,
            col=col,
            scaleanchor=xaxis_name,
            scaleratio=1,
        )
    return fig


def make_seed_grid_figure(
    *,
    final_free_by_seed: dict[int, np.ndarray],
    final_conditioned_by_seed: dict[int, np.ndarray],
    color_values: np.ndarray,
    center: Sequence[float],
    target_radius: float,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
    title: str = "Final states across seeds: free (top row) vs conditioned (bottom row)",
) -> go.Figure:
    """Show final particle states for several RNG seeds in a 2xN grid.

    The top row contains free-diffusion end states and the bottom row contains
    the corresponding conditioned end states, using the same particle coloring
    in every panel to preserve particle identity across seeds.
    """
    seeds = list(final_free_by_seed.keys())
    if not seeds:
        raise ValueError("final_free_by_seed must contain at least one seed")
    missing = [seed for seed in seeds if seed not in final_conditioned_by_seed]
    if missing:
        missing_text = ", ".join(str(seed) for seed in missing)
        raise KeyError(f"Missing conditioned final states for seeds: {missing_text}")

    fig = make_subplots(
        rows=2,
        cols=len(seeds),
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
        subplot_titles=[f"free, seed {seed}" for seed in seeds]
        + [f"conditioned, seed {seed}" for seed in seeds],
    )

    ring = circle_trace(center, target_radius, name="target annulus", showlegend=True)

    for col, seed in enumerate(seeds, start=1):
        fig.add_trace(
            go.Scatter(
                x=ring.x,
                y=ring.y,
                mode="lines",
                line=ring.line,
                name="target annulus",
                showlegend=(col == 1),
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=final_free_by_seed[seed][:, 0],
                y=final_free_by_seed[seed][:, 1],
                mode="markers",
                marker=dict(
                    size=5,
                    color=color_values,
                    colorscale="HSV",
                    cmin=0.0,
                    cmax=1.0,
                    showscale=False,
                ),
                name=f"free {seed}",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=ring.x,
                y=ring.y,
                mode="lines",
                line=ring.line,
                name="target annulus",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=final_conditioned_by_seed[seed][:, 0],
                y=final_conditioned_by_seed[seed][:, 1],
                mode="markers",
                marker=dict(
                    size=5,
                    color=color_values,
                    colorscale="HSV",
                    cmin=0.0,
                    cmax=1.0,
                    showscale=False,
                ),
                name=f"conditioned {seed}",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=col,
        )

    fig.update_layout(
        title=title,
        template="simple_white",
        width=max(1160, 260 * len(seeds)),
        height=620,
    )

    n_subplots = 2 * len(seeds)
    for axis_index in range(1, n_subplots + 1):
        xaxis_name = "xaxis" if axis_index == 1 else f"xaxis{axis_index}"
        yaxis_name = "yaxis" if axis_index == 1 else f"yaxis{axis_index}"
        scaleanchor_name = "x" if axis_index == 1 else f"x{axis_index}"
        fig.layout[xaxis_name].update(
            range=list(x_range), showgrid=False, zeroline=False
        )
        fig.layout[yaxis_name].update(
            range=list(y_range),
            showgrid=False,
            zeroline=False,
            scaleanchor=scaleanchor_name,
            scaleratio=1,
        )

    return fig
