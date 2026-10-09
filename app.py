import streamlit as st
import pandas as pd
import numpy as np
import time

from src.data_generator import generate_cvrp_instance
from src.baseline_optimizer import BaselineOptimizer
from src.simulated_annealing import SimulatedAnnealingOptimizer
from src.disruption_simulator import DisruptionSimulator, BLOCKED_COST
from src.visualization import (
    create_route_figure,
    create_benchmark_comparison_chart,
    create_runtime_comparison_chart
)
from src.benchmark import BenchmarkRunner

# Page Configuration
st.set_page_config(
    page_title="RouteResQ AI — Disruption-Aware Fleet Routing",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Aesthetic Design
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E1B4B 0%, #4338CA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .status-badge-container {
        display: inline-block;
        margin-bottom: 1rem;
    }
    .status-badge {
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .badge-disrupted {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-header">RouteResQ AI 🚚</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Disruption-Aware Fleet Routing with Quantum-Inspired Optimization (CVRP)</div>', unsafe_allow_html=True)

# Sidebar Configuration Controls
st.sidebar.header("⚙️ Problem & Fleet Configuration")
num_customers = st.sidebar.slider("Delivery Locations (N)", min_value=5, max_value=30, value=15, step=1)
num_vehicles = st.sidebar.slider("Fleet Vehicles (K)", min_value=2, max_value=8, value=3, step=1)
vehicle_capacity = st.sidebar.slider("Vehicle Capacity (Q)", min_value=20, max_value=150, value=50, step=5)
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# Generate or Retrieve Instance State
if "instance" not in st.session_state or st.sidebar.button("🎲 Generate New Dataset"):
    st.session_state.instance = generate_cvrp_instance(
        num_customers=num_customers,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        seed=int(random_seed)
    )
    st.session_state.disrupted_instance = None
    st.session_state.initial_sol_base = None
    st.session_state.initial_sol_sa = None
    st.session_state.disruption_info = None

instance = st.session_state.instance

# Sidebar SA Controls
st.sidebar.markdown("---")
st.sidebar.header("🧠 Simulated Annealing Settings")
sa_iterations = st.sidebar.slider("Max SA Iterations", 500, 10000, 3000, step=500)
sa_temp = st.sidebar.number_input("Initial Temp (T)", value=1000.0, step=100.0)
sa_cooling = st.sidebar.slider("Cooling Rate (α)", 0.900, 0.999, 0.995, step=0.001, format="%.3f")

# Solve Initial Solutions if not present
if st.session_state.initial_sol_base is None:
    base_solver = BaselineOptimizer(use_two_opt=True)
    st.session_state.initial_sol_base = base_solver.solve(instance)

if st.session_state.initial_sol_sa is None:
    sa_solver = SimulatedAnnealingOptimizer(
        initial_temp=sa_temp,
        cooling_rate=sa_cooling,
        max_iterations=sa_iterations,
        seed=int(random_seed)
    )
    st.session_state.initial_sol_sa = sa_solver.solve(instance)

# Operational Status Badge
is_disrupted = st.session_state.disrupted_instance is not None
if is_disrupted:
    badge_html = f'<div class="status-badge-container"><span class="status-badge badge-disrupted">🔴 Disruption Active: {st.session_state.disruption_info}</span></div>'
else:
    badge_html = '<div class="status-badge-container"><span class="status-badge badge-normal">🟢 Operational: Normal Network Conditions</span></div>'
st.markdown(badge_html, unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Overview",
    "🗺️ Initial Routes",
    "⚠️ Disruption Simulator",
    "🔄 Dynamic Replanning",
    "📊 Benchmark & Comparison",
    "🔬 Experiment Details"
])

# Helper function to generate route breakdown table
def get_routes_table(solution, inst):
    rows = []
    for r in solution.routes:
        if len(r.node_ids) <= 2:
            continue
        path_str = " ➔ ".join(str(n) if n != 0 else "Depot" for n in r.node_ids)
        utilization = (r.total_demand / inst.vehicle_capacity) * 100.0 if inst.vehicle_capacity > 0 else 0.0
        rows.append({
            "Vehicle ID": f"Vehicle #{r.vehicle_id + 1}",
            "Stops Count": len(r.node_ids) - 2,
            "Path Sequence": path_str,
            "Load (units)": f"{r.total_demand} / {inst.vehicle_capacity}",
            "Capacity Util.": f"{utilization:.1f}%",
            "Route Cost (km)": f"{r.total_cost:.2f}"
        })
    return pd.DataFrame(rows)

# TAB 1: OVERVIEW
with tab1:
    st.subheader("System Status & Network Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Depot Node", "1 Central (Node 0)")
    with col2:
        st.metric("Delivery Locations", f"{instance.num_customers} Nodes")
    with col3:
        total_fleet_cap = instance.num_vehicles * instance.vehicle_capacity
        st.metric("Fleet Capacity", f"{instance.num_vehicles} x {instance.vehicle_capacity} ({total_fleet_cap} total)")
    with col4:
        total_demand = sum(c.demand for c in instance.customers)
        util = (total_demand / total_fleet_cap) * 100.0 if total_fleet_cap > 0 else 0.0
        st.metric("Total Load Demand", f"{total_demand} units ({util:.1f}% fleet load)")

    st.markdown("---")
    st.markdown("""
    ### 🎯 Hackathon Scope & Operational Challenge
    Logistics companies plan delivery vehicle routes in advance to minimize travel time, fuel usage, and operational costs. However, operations are frequently disrupted by:
    * **Road Closures** (accidents, construction, severe weather)
    * **Traffic Congestion** (unexpected delays and travel-cost spikes)
    * **Vehicle Breakdowns** (loss of active fleet capacity)

    **RouteResQ AI** models the **Capacitated Vehicle Routing Problem (CVRP)** and evaluates two classical optimization approaches:
    1. **Traditional Baseline**: Nearest Neighbor (NN) greedy heuristic enhanced with $2$-opt local search improvement.
    2. **Quantum-Inspired Optimization**: Simulated Annealing (SA) using Metropolis thermal state transitions to escape local minima without actual quantum hardware.
    """)

    st.markdown("""
    #### 📐 Mathematical Formulation of CVRP
    $$\\min \\sum_{k=1}^K \\sum_{i=0}^N \\sum_{j=0}^N c_{ij} x_{ijk}$$
    **Subject to:**
    * $\\sum_{j=1}^N x_{0jk} = 1 \\quad \\forall k$ *(Every vehicle departs from depot)*
    * $\\sum_{k=1}^K \\sum_{j=0, j \\ne i}^N x_{ijk} = 1 \\quad \\forall i \\in \\{1, \\dots, N\\}$ *(Every customer visited exactly once)*
    * $\\sum_{i=1}^N d_i \\sum_{j=0}^N x_{ijk} \\le Q \\quad \\forall k$ *(Vehicle capacity constraint)*
    * $x_{ijk} = 0 \\quad \\forall (i, j) \\in \\mathcal{E}_{\\text{disrupted}}$ *(Unavailable road segment exclusion)*
    """)
    
    st.info("💡 Note: Synthetic Euclidean coordinates are generated for local benchmark reproducibility. Real-world commercial deployments require actual road-network graph matrices (e.g., OpenStreetMap / OSRM).")

# TAB 2: INITIAL ROUTE PLAN
with tab2:
    st.subheader("Pre-Disruption Optimized Routes")
    
    sol_base = st.session_state.initial_sol_base
    sol_sa = st.session_state.initial_sol_sa

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Total Cost", f"{sol_base.total_cost:.2f} km")
    c2.metric("Baseline Runtime", f"{sol_base.execution_time_ms:.1f} ms")
    cost_diff = sol_sa.total_cost - sol_base.total_cost
    c3.metric("SA Total Cost", f"{sol_sa.total_cost:.2f} km", delta=f"{cost_diff:+.2f} km", delta_color="inverse")
    c4.metric("SA Runtime", f"{sol_sa.execution_time_ms:.1f} ms")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Baseline Route Plan (Nearest Neighbor + 2-opt)")
        fig_base = create_route_figure(instance, sol_base, title="Baseline Initial Routes")
        st.plotly_chart(fig_base, use_container_width=True)
        st.markdown("**Vehicle Route Breakdown:**")
        st.dataframe(get_routes_table(sol_base, instance), use_container_width=True)

    with col_right:
        st.markdown("#### Quantum-Inspired Route Plan (Simulated Annealing)")
        fig_sa = create_route_figure(instance, sol_sa, title="Simulated Annealing Initial Routes")
        st.plotly_chart(fig_sa, use_container_width=True)
        st.markdown("**Vehicle Route Breakdown:**")
        st.dataframe(get_routes_table(sol_sa, instance), use_container_width=True)

# TAB 3: DISRUPTION SIMULATOR
with tab3:
    st.subheader("Inject Operational Disruption")
    
    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        disruption_type = st.selectbox(
            "Disruption Scenario",
            ["Road Closure (Edge Blockage)", "Traffic Congestion Penalty", "Vehicle Breakdown"]
        )
    
    with d_col2:
        all_node_ids = [0] + [c.id for c in instance.customers]
        u_node = st.selectbox("Origin Node (u)", all_node_ids, index=0)
        default_v = 1 if len(all_node_ids) > 1 else 0
        v_node = st.selectbox("Destination Node (v)", all_node_ids, index=default_v)

    with d_col3:
        penalty_factor = st.slider("Traffic Congestion Multiplier", 2.0, 10.0, 5.0) if disruption_type == "Traffic Congestion Penalty" else 1.0
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            trigger_btn = st.button("🚨 Apply Disruption", type="primary", use_container_width=True)
        with btn_col2:
            reset_btn = st.button("🔄 Reset Network", use_container_width=True)

    if reset_btn:
        st.session_state.disrupted_instance = None
        st.session_state.disruption_info = None
        st.success("Network reset to normal state.")
        st.rerun()

    if trigger_btn:
        if disruption_type == "Road Closure (Edge Blockage)":
            if u_node == v_node:
                st.error("Origin and Destination nodes must be distinct.")
            else:
                st.session_state.disrupted_instance = DisruptionSimulator.apply_edge_closure(instance, u_node, v_node)
                st.session_state.disruption_info = f"Road Closure between Node {u_node} and Node {v_node}"
                st.warning(f"⚠️ Road Closure applied to edge ({u_node} ➔ {v_node}). Edge cost set to {BLOCKED_COST:,.0f} (blocked).")
        elif disruption_type == "Traffic Congestion Penalty":
            if u_node == v_node:
                st.error("Origin and Destination nodes must be distinct.")
            else:
                st.session_state.disrupted_instance = DisruptionSimulator.apply_traffic_penalty(instance, u_node, v_node, penalty_factor)
                st.session_state.disruption_info = f"Traffic Penalty ({penalty_factor}x) on edge ({u_node} ➔ {v_node})"
                st.warning(f"⚠️ Congestion penalty ({penalty_factor}x) applied to edge ({u_node} ➔ {v_node}).")
        elif disruption_type == "Vehicle Breakdown":
            st.session_state.disrupted_instance = DisruptionSimulator.apply_vehicle_breakdown(instance, 0)
            st.session_state.disruption_info = f"Vehicle #1 Broken Down (Fleet: {instance.num_vehicles} ➔ {instance.num_vehicles - 1})"
            st.warning(f"⚠️ Vehicle #1 broken down! Active fleet reduced to {st.session_state.disrupted_instance.num_vehicles} vehicles.")

    st.markdown("---")
    st.markdown("### 🌐 Changed Travel-Cost Network Representation")
    
    if st.session_state.disrupted_instance is not None:
        d_inst = st.session_state.disrupted_instance
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown(f"**Disruption Summary:** `{st.session_state.disruption_info}`")
            if d_inst.disrupted_edges:
                u, v = d_inst.disrupted_edges[-1]
                orig_cost = instance.cost_matrix[u, v]
                new_cost = d_inst.cost_matrix[u, v]
                cost_label = "BLOCKED (1,000,000)" if new_cost >= 1e5 else f"{new_cost:.2f} km"
                st.info(f"**Edge ({u} ➔ {v}) Cost Change:** {orig_cost:.2f} km ➔ **{cost_label}**")
                
                # Check which initial routes were affected
                base_affected = DisruptionSimulator.check_affected_routes(st.session_state.initial_sol_base, [(u, v)])
                sa_affected = DisruptionSimulator.check_affected_routes(st.session_state.initial_sol_sa, [(u, v)])
                st.write(f"- **Initial Baseline Routes Affected:** Vehicle(s) {[i + 1 for i in base_affected] or 'None'}")
                st.write(f"- **Initial SA Routes Affected:** Vehicle(s) {[i + 1 for i in sa_affected] or 'None'}")
            elif disruption_type == "Vehicle Breakdown":
                st.info(f"Active Vehicles: {d_inst.num_vehicles} (Remaining fleet capacity: {d_inst.num_vehicles * d_inst.vehicle_capacity})")

        with c_right:
            # Show figure highlighting the disrupted edge
            fig_disrupt = create_route_figure(d_inst, st.session_state.initial_sol_base, title="Network with Disrupted Segment", highlight_disruption=True)
            st.plotly_chart(fig_disrupt, use_container_width=True)
    else:
        st.info("No disruption active. Select parameters and click 'Apply Disruption' to simulate.")

# TAB 4: DYNAMIC REPLANNING
with tab4:
    st.subheader("Post-Disruption Route Recalculation")

    disrupted_inst = st.session_state.disrupted_instance
    if disrupted_inst is None:
        st.info("👈 Please inject a disruption in the 'Disruption Simulator' tab first.")
    else:
        # Solve replanning
        base_solver = BaselineOptimizer(use_two_opt=True)
        replan_base = base_solver.solve(disrupted_inst)

        sa_solver = SimulatedAnnealingOptimizer(
            initial_temp=sa_temp,
            cooling_rate=sa_cooling,
            max_iterations=sa_iterations,
            seed=int(random_seed)
        )
        replan_sa = sa_solver.solve(disrupted_inst)

        # Metrics display
        r1, r2, r3, r4 = st.columns(4)
        cost_diff_base = replan_base.total_cost - st.session_state.initial_sol_base.total_cost
        cost_diff_sa = replan_sa.total_cost - st.session_state.initial_sol_sa.total_cost

        r1.metric("Baseline Replanned Cost", f"{replan_base.total_cost:.2f} km", delta=f"{cost_diff_base:+.2f} km", delta_color="inverse")
        r2.metric("Baseline Replanned Time", f"{replan_base.execution_time_ms:.1f} ms")
        r3.metric("SA Replanned Cost", f"{replan_sa.total_cost:.2f} km", delta=f"{cost_diff_sa:+.2f} km", delta_color="inverse")
        r4.metric("SA Replanned Time", f"{replan_sa.execution_time_ms:.1f} ms")

        col_left_replan, col_right_replan = st.columns(2)
        with col_left_replan:
            st.markdown("#### Replanned Baseline Routes")
            fig_replan_base = create_route_figure(disrupted_inst, replan_base, title="Replanned Baseline", highlight_disruption=True)
            st.plotly_chart(fig_replan_base, use_container_width=True)
            st.dataframe(get_routes_table(replan_base, disrupted_inst), use_container_width=True)

        with col_right_replan:
            st.markdown("#### Replanned Simulated Annealing Routes")
            fig_replan_sa = create_route_figure(disrupted_inst, replan_sa, title="Replanned Quantum-Inspired SA", highlight_disruption=True)
            st.plotly_chart(fig_replan_sa, use_container_width=True)
            st.dataframe(get_routes_table(replan_sa, disrupted_inst), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📝 Reason for Route Change & Fleet Rebalancing")
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("""
            **Baseline Detour Logic (Nearest Neighbor + 2-opt):**
            * When an edge is closed or penalized, its entry in the cost matrix is updated.
            * The greedy constructor immediately avoids the high-cost edge for candidate selection.
            * Local search 2-opt swaps evaluate alternative node sequences to find a lower-cost unblocked detour.
            """)
        with exp_col2:
            st.markdown("""
            **Quantum-Inspired SA Rebalancing Logic:**
            * The objective function penalizes traversal of the disrupted edge.
            * High energy states are penalized and rejected by the Metropolis criterion at cooling temperatures.
            * Customer nodes previously on the congested route are migrated and swapped into neighboring vehicle routes.
            """)

        if disrupted_inst.num_vehicles < instance.num_vehicles:
            st.warning(f"🚗 **Vehicle Breakdown Reassignment Details:** All customers originally served by the broken-down vehicle were automatically redistributed across the remaining {disrupted_inst.num_vehicles} active vehicles while respecting capacity limits.")

# TAB 5: BENCHMARK & COMPARISON
with tab5:
    st.subheader("Statistical Benchmark & Algorithm Comparison")
    
    b_col1, b_col2 = st.columns([1, 2])
    with b_col1:
        num_runs = st.slider("Monte Carlo Runs", 3, 20, 5, step=1)
        compare_modes = st.checkbox("Evaluate Disruptions in Benchmark (Compare Before vs After)", value=True)
        run_btn = st.button("🚀 Run Monte Carlo Experiments", type="primary")

    if run_btn:
        with st.spinner("Executing statistical benchmark across multiple random seeds..."):
            runner = BenchmarkRunner(
                num_runs=num_runs,
                num_customers=num_customers,
                num_vehicles=num_vehicles,
                capacity=vehicle_capacity
            )
            df_results, summary_metrics = runner.run_benchmark(compare_modes=compare_modes)
            st.session_state.benchmark_df = df_results
            st.session_state.benchmark_summary = summary_metrics

    if "benchmark_df" in st.session_state:
        df = st.session_state.benchmark_df
        summary = st.session_state.benchmark_summary
        
        st.markdown("#### Benchmark Statistical Summary")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Baseline Mean Cost", f"{summary['Baseline Mean Cost']:.2f} km", help=f"Best: {summary['Baseline Best Cost']:.1f} km, Worst: {summary['Baseline Worst Cost']:.1f} km")
        b2.metric("SA Mean Cost", f"{summary['SA Mean Cost']:.2f} km", help=f"Best: {summary['SA Best Cost']:.1f} km, Worst: {summary['SA Worst Cost']:.1f} km")
        b3.metric("Baseline Feasibility", f"{summary['Baseline Feasibility Rate']:.0f}%")
        b4.metric("SA Feasibility", f"{summary['SA Feasibility Rate']:.0f}%")

        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            chart_cost = create_benchmark_comparison_chart(df)
            st.plotly_chart(chart_cost, use_container_width=True)
        with ch_col2:
            chart_runtime = create_runtime_comparison_chart(df)
            st.plotly_chart(chart_runtime, use_container_width=True)

        st.markdown("#### Complete Experimental Run Log")
        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Benchmark Results (CSV)",
            data=csv_data,
            file_name="routeresq_benchmark_results.csv",
            mime="text/csv"
        )

# TAB 6: EXPERIMENT DETAILS
with tab6:
    st.subheader("Experimental Parameters & Reproducibility")
    
    param_col1, param_col2 = st.columns(2)
    with param_col1:
        st.markdown("#### ⚙️ Current Instance Configuration")
        raw_values = [
            instance.num_customers,
            instance.num_vehicles,
            instance.vehicle_capacity,
            int(random_seed),
            sum(c.demand for c in instance.customers),
            "100.0 x 100.0",
            sa_iterations,
            sa_temp,
            sa_cooling,
            100.0,
            500.0
        ]
        param_data = {
            "Parameter": [
                "Delivery Customers (N)",
                "Fleet Vehicles (K)",
                "Vehicle Capacity (Q)",
                "Random Seed",
                "Total Demand (units)",
                "Grid Size",
                "Simulated Annealing Iterations",
                "Initial Temperature",
                "Cooling Rate",
                "Capacity Penalty Multiplier",
                "Unassigned Penalty Multiplier"
            ],
            "Value": [str(v) for v in raw_values]
        }
        st.table(pd.DataFrame(param_data))

    with param_col2:
        st.markdown("""
        #### 🔬 Responsible Scientific & AI Claims
        * **Classical Algorithm Notice**: Simulated Annealing is a **classical metaheuristic** that uses stochastic thermal cooling analogous to quantum tunneling. It does not run on quantum hardware.
        * **No Fabricated Quantum Advantage**: Claims of quantum supremacy or guaranteed global optimality are avoided. Performance trade-offs are strictly empirically measured.
        * **Euclidean Data Model**: Cost values reflect planar Euclidean distances ($L_2$ norm) for algorithmic benchmarking, rather than physical road GIS routing.
        """)
        
        st.markdown("""
        #### 🏆 Hackathon Demonstration Checklist
        - [x] **Reproducible Generation**: Configurable seeds and demand generation.
        - [x] **Traditional Baseline**: Nearest Neighbor heuristic + 2-opt search.
        - [x] **Quantum-Inspired Optimizer**: Simulated Annealing with soft penalty evaluation.
        - [x] **Disruption Simulation**: Road closure, traffic congestion multiplier, vehicle breakdown.
        - [x] **Dynamic Replanning**: Route recalculation without traversing unavailable edges.
        - [x] **Benchmarking**: Statistical Monte Carlo runs with exportable CSV.
        """)
