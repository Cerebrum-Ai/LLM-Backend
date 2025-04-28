#!/usr/bin/env python3
"""
Typing Analysis Dashboard for Cerebrum AI

This script provides a dashboard for analyzing typing patterns and visualizing
the results from the ML service.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ML_SERVICE_URL = "http://localhost:9000"

def load_typing_data():
    """Load all typing data files from the data directory"""
    data_files = []
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            if file.endswith('.json'):
                data_files.append(os.path.join(DATA_DIR, file))
    return data_files

# Initialize the Dash app
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.DARKLY],
                title="Cerebrum AI - Typing Analysis Dashboard")

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Cerebrum AI - Typing Pattern Analysis Dashboard", 
                    className="text-center text-primary my-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Selection"),
                dbc.CardBody([
                    html.P("Select data files or submit live data for analysis"),
                    dbc.Button("Refresh Data Files", id="refresh-btn", color="primary", className="me-2"),
                    dcc.Dropdown(id="file-selector", multi=True, placeholder="Select data files..."),
                    html.Hr(),
                    dbc.Button("Analyze Selected Files", id="analyze-btn", color="success", className="mt-2")
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("ML Analysis Results"),
                dbc.CardBody([
                    html.Div(id="analysis-results")
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Key Duration Analysis", children=[
                    dcc.Graph(id="duration-time-graph", style={"height": "300px"}),
                    dcc.Graph(id="duration-hist-graph", style={"height": "300px"})
                ]),
                dbc.Tab(label="Interval Analysis", children=[
                    dcc.Graph(id="interval-time-graph", style={"height": "300px"}),
                    dcc.Graph(id="interval-hist-graph", style={"height": "300px"})
                ]),
                dbc.Tab(label="Feature Comparison", children=[
                    dcc.Graph(id="feature-radar-graph", style={"height": "400px"}),
                    dcc.Graph(id="feature-bar-graph", style={"height": "200px"})
                ])
            ])
        ], width=8)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Live Data Collection"),
                dbc.CardBody([
                    dbc.Input(id="subject-name", placeholder="Subject Name", type="text", className="mb-2"),
                    dbc.Button("Start Collecting", id="start-collecting-btn", color="warning", className="me-2"),
                    dbc.Button("Stop & Save", id="stop-collecting-btn", color="danger", disabled=True),
                    html.Div(id="collection-status", className="mt-2")
                ])
            ], className="my-4")
        ], width=12)
    ]),
    
    dcc.Store(id="typing-data-store"),
    dcc.Store(id="analysis-store"),
    dcc.Interval(id="collection-interval", interval=500, disabled=True)
    
], fluid=True)

# Callback to refresh data files
@callback(
    Output("file-selector", "options"),
    Input("refresh-btn", "n_clicks")
)
def refresh_data_files(n_clicks):
    if n_clicks is None:
        return []
    
    data_files = glob.glob("typing_data_*.json")
    data_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    options = []
    for file in data_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                keystrokes = len(data.get("keystrokes", []))
                timestamp = os.path.getmtime(file)
                dt = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                
                label = f"{file} ({dt}, {keystrokes} keystrokes)"
                options.append({"label": label, "value": file})
        except:
            options.append({"label": file, "value": file})
    
    return options

# Callback to analyze selected files
@callback(
    Output("typing-data-store", "data"),
    Output("analysis-store", "data"),
    Input("analyze-btn", "n_clicks"),
    State("file-selector", "value")
)
def analyze_files(n_clicks, selected_files):
    if n_clicks is None or not selected_files:
        return None, None
    
    all_data = {}
    analysis_results = {}
    
    for file in selected_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                
                # Process keystrokes data
                keystrokes = data.get("keystrokes", [])
                
                if not keystrokes:
                    continue
                
                # Extract durations and intervals
                durations = []
                intervals = []
                first_time = keystrokes[0]["timeDown"]
                
                for i, keystroke in enumerate(keystrokes):
                    if "timeDown" in keystroke and "timeUp" in keystroke:
                        duration = keystroke["timeUp"] - keystroke["timeDown"]
                        durations.append({
                            "file": file,
                            "time": keystroke["timeDown"] - first_time,
                            "duration": duration
                        })
                    
                    if i > 0 and "timeDown" in keystroke and "timeDown" in keystrokes[i-1]:
                        interval = keystroke["timeDown"] - keystrokes[i-1]["timeDown"]
                        if 0 < interval < 2000:  # Filter out pauses > 2 seconds
                            intervals.append({
                                "file": file,
                                "time": keystroke["timeDown"] - first_time,
                                "interval": interval
                            })
                
                all_data[file] = {
                    "durations": durations,
                    "intervals": intervals,
                    "raw_data": data
                }
                
                # Send to ML service for analysis
                try:
                    response = requests.post(
                        "http://localhost:9000/ml/process",
                        json={
                            "url": data,
                            "data_type": "typing",
                            "model": "pattern"
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        analysis_results[file] = response.json()
                except Exception as e:
                    analysis_results[file] = {"error": str(e)}
                    
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    return all_data, analysis_results

# Callback to update duration time graph
@callback(
    Output("duration-time-graph", "figure"),
    Input("typing-data-store", "data")
)
def update_duration_time_graph(data):
    if not data:
        return go.Figure()
    
    fig = go.Figure()
    
    for file, file_data in data.items():
        durations = file_data["durations"]
        if durations:
            df = pd.DataFrame(durations)
            
            # Add a trend line
            x = df["time"]
            y = df["duration"]
            trend_line = np.poly1d(np.polyfit(x, y, 1))
            
            file_name = os.path.basename(file)
            fig.add_trace(go.Scatter(
                x=df["time"],
                y=df["duration"],
                mode="markers",
                name=file_name,
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=df["time"],
                y=trend_line(df["time"]),
                mode="lines",
                name=f"{file_name} trend",
                line=dict(dash="dash")
            ))
    
    fig.update_layout(
        title="Key Press Duration Over Time",
        xaxis_title="Time (ms)",
        yaxis_title="Duration (ms)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Callback to update duration histogram
@callback(
    Output("duration-hist-graph", "figure"),
    Input("typing-data-store", "data")
)
def update_duration_hist_graph(data):
    if not data:
        return go.Figure()
    
    fig = go.Figure()
    
    for file, file_data in data.items():
        durations = file_data["durations"]
        if durations:
            df = pd.DataFrame(durations)
            file_name = os.path.basename(file)
            
            fig.add_trace(go.Histogram(
                x=df["duration"],
                name=file_name,
                opacity=0.7,
                nbinsx=20
            ))
    
    fig.update_layout(
        title="Key Press Duration Distribution",
        xaxis_title="Duration (ms)",
        yaxis_title="Count",
        template="plotly_dark",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Callback to update interval time graph
@callback(
    Output("interval-time-graph", "figure"),
    Input("typing-data-store", "data")
)
def update_interval_time_graph(data):
    if not data:
        return go.Figure()
    
    fig = go.Figure()
    
    for file, file_data in data.items():
        intervals = file_data["intervals"]
        if intervals:
            df = pd.DataFrame(intervals)
            
            # Add a trend line
            x = df["time"]
            y = df["interval"]
            trend_line = np.poly1d(np.polyfit(x, y, 1))
            
            file_name = os.path.basename(file)
            fig.add_trace(go.Scatter(
                x=df["time"],
                y=df["interval"],
                mode="markers",
                name=file_name,
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=df["time"],
                y=trend_line(df["time"]),
                mode="lines",
                name=f"{file_name} trend",
                line=dict(dash="dash")
            ))
    
    fig.update_layout(
        title="Time Between Keys",
        xaxis_title="Time (ms)",
        yaxis_title="Interval (ms)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Callback to update interval histogram
@callback(
    Output("interval-hist-graph", "figure"),
    Input("typing-data-store", "data")
)
def update_interval_hist_graph(data):
    if not data:
        return go.Figure()
    
    fig = go.Figure()
    
    for file, file_data in data.items():
        intervals = file_data["intervals"]
        if intervals:
            df = pd.DataFrame(intervals)
            file_name = os.path.basename(file)
            
            fig.add_trace(go.Histogram(
                x=df["interval"],
                name=file_name,
                opacity=0.7,
                nbinsx=20
            ))
    
    fig.update_layout(
        title="Time Between Keys Distribution",
        xaxis_title="Interval (ms)",
        yaxis_title="Count",
        template="plotly_dark",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Callback to update feature radar graph
@callback(
    Output("feature-radar-graph", "figure"),
    Output("feature-bar-graph", "figure"),
    Output("analysis-results", "children"),
    Input("analysis-store", "data")
)
def update_feature_graphs(data):
    if not data:
        return go.Figure(), go.Figure(), html.P("No analysis results yet")
    
    # Create radar chart for features
    feature_fig = go.Figure()
    
    # Create bar chart for probabilities
    prob_fig = go.Figure()
    
    # Create analysis results text
    results_elements = []
    
    for file, analysis in data.items():
        file_name = os.path.basename(file)
        
        if "error" in analysis:
            results_elements.append(html.Div([
                html.H5(file_name, className="text-warning"),
                html.P(f"Error: {analysis['error']}", className="text-danger")
            ]))
            continue
        
        # Extract features if available
        features = analysis.get("features", {})
        if features:
            feature_names = list(features.keys())
            feature_values = list(features.values())
            
            feature_fig.add_trace(go.Scatterpolar(
                r=feature_values,
                theta=feature_names,
                fill='toself',
                name=file_name
            ))
        
        # Extract probabilities if available
        probabilities = analysis.get("probabilities", {})
        if probabilities:
            conditions = list(probabilities.keys())
            probs = list(probabilities.values())
            
            prob_fig.add_trace(go.Bar(
                x=conditions,
                y=probs,
                name=file_name
            ))
        
        # Create results text
        detected = analysis.get("detected_condition", "unknown")
        
        results_elements.append(html.Div([
            html.H5(file_name, className="text-info"),
            html.P([
                html.Strong("Detected condition: "),
                html.Span(detected, className="text-warning")
            ]),
            html.Hr()
        ]))
    
    feature_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Typing Feature Comparison",
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    prob_fig.update_layout(
        title="Condition Probabilities",
        xaxis_title="Condition",
        yaxis_title="Probability",
        template="plotly_dark",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return feature_fig, prob_fig, results_elements

if __name__ == "__main__":
    print("Starting Cerebrum AI Typing Analysis Dashboard...")
    print("Dashboard available at: http://127.0.0.1:8050/")
    app.run_server(debug=True) 