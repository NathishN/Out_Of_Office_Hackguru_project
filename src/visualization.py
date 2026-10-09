import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Tuple
from src.routing_model import ProblemInstance, Solution

VEHICLE_COLORS = [
    "#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6",
    "#8B5CF6", "#14B8A6", "#EF4444", "#84CC16", "#06B6D4"
]

def create_route_figure(
    instance: ProblemInstance,
    solution: Solution,
    title: str = "Vehicle Delivery Routes",
    highlight_disruption: bool = False
) -> go.Figure:
    """
    Creates an interactive 2D Plotly map showing depot, customers, vehicle routes, and disrupted edges.
    """
    fig = go.Figure()

    # 1. Plot Disrupted Edges if present
    if highlight_disruption and instance.disrupted_edges:
        for u, v in instance.disrupted_edges:
            node_u = instance.get_node_by_id(u)
            node_v = instance.get_node_by_id(v)
            fig.add_trace(go.Scatter(
                x=[node_u.x, node_v.x],
                y=[node_u.y, node_v.y],
                mode="lines+markers",
                line=dict(color="#EF4444", width=4, dash="dash"),
                marker=dict(size=12, symbol="x", color="#EF4444"),
                name=f"Disrupted Edge ({u} ➔ {v})"
            ))

    # 2. Plot Vehicle Routes
    for r_idx, route in enumerate(solution.routes):
        color = VEHICLE_COLORS[r_idx % len(VEHICLE_COLORS)]
        if len(route.node_ids) <= 2:
            continue

        route_x, route_y, text_labels = [], [], []
        for nid in route.node_ids:
            node = instance.get_node_by_id(nid)
            route_x.append(node.x)
            route_y.append(node.y)
            label = f"Depot" if nid == 0 else f"Cust #{nid} (Dem: {node.demand})"
            text_labels.append(label)

        fig.add_trace(go.Scatter(
            x=route_x,
            y=route_y,
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color),
            hovertext=text_labels,
            hoverinfo="text",
            name=f"Vehicle #{route.vehicle_id + 1} (Cost: {route.total_cost:.1f})"
        ))

    # 3. Plot Customer Nodes
    cust_x = [c.x for c in instance.customers]
    cust_y = [c.y for c in instance.customers]
    cust_hover = [f"<b>Customer #{c.id}</b><br>Demand: {c.demand}<br>Pos: ({c.x:.1f}, {c.y:.1f})" for c in instance.customers]
    cust_text = [str(c.id) for c in instance.customers]

    fig.add_trace(go.Scatter(
        x=cust_x,
        y=cust_y,
        mode="markers+text",
        marker=dict(size=14, color="#3B82F6", line=dict(width=2, color="#1E3A8A")),
        text=cust_text,
        textposition="top center",
        hovertext=cust_hover,
        hoverinfo="text",
        name="Delivery Locations"
    ))

    # 4. Plot Central Depot
    fig.add_trace(go.Scatter(
        x=[instance.depot.x],
        y=[instance.depot.y],
        mode="markers+text",
        marker=dict(size=22, color="#F59E0B", symbol="star", line=dict(width=2, color="#78350F")),
        text=["DEPOT"],
        textposition="bottom center",
        hoverinfo="text",
        hovertext=f"<b>Central Depot</b><br>Pos: ({instance.depot.x:.1f}, {instance.depot.y:.1f})",
        name="Depot (Node 0)"
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#1E293B")),
        xaxis=dict(title="X Coordinate (km / miles)", gridcolor="#E2E8F0", zeroline=False),
        yaxis=dict(title="Y Coordinate (km / miles)", gridcolor="#E2E8F0", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=80),
        height=550
    )

    return fig

def create_benchmark_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """
    Creates comparative bar charts for total cost.
    """
    summary = df.groupby(["Algorithm", "Disrupted"])[["Total Cost", "Runtime (ms)"]].mean().reset_index()
    summary["Status"] = summary["Disrupted"].map({True: "Disrupted", False: "Normal"})

    fig = px.bar(
        summary,
        x="Algorithm",
        y="Total Cost",
        color="Status",
        barmode="group",
        title="Route Cost Comparison Across Algorithms & Disruptions",
        color_discrete_map={"Normal": "#3B82F6", "Disrupted": "#EF4444"},
        text_auto=".1f"
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig

def create_runtime_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """
    Creates comparative bar chart for execution runtime (ms).
    """
    summary = df.groupby(["Algorithm", "Disrupted"])[["Total Cost", "Runtime (ms)"]].mean().reset_index()
    summary["Status"] = summary["Disrupted"].map({True: "Disrupted", False: "Normal"})

    fig = px.bar(
        summary,
        x="Algorithm",
        y="Runtime (ms)",
        color="Status",
        barmode="group",
        title="Execution Runtime (ms) Comparison (Log Scale)",
        color_discrete_map={"Normal": "#10B981", "Disrupted": "#F59E0B"},
        text_auto=".2f",
        log_y=True
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(title="Runtime (ms, log scale)")
    )

    return fig
