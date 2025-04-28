#!/usr/bin/env python3
"""
Typing Analysis Dashboard for Cerebrum AI

This script provides a dashboard for analyzing typing patterns and visualizing
the results from the ML service.
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import time
import glob
import requests
from datetime import datetime
from pynput import keyboard
import threading
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ML_SERVICE_URL = "http://localhost:9000"

# Global variables for keyboard monitoring
is_collecting = False
current_keystrokes = []
keystroke_lock = threading.Lock()
keyboard_listener = None

def on_press(key):
    """Handle key press events"""
    global current_keystrokes, is_collecting
    
    if not is_collecting:
        return
        
    try:
        key_char = key.char if hasattr(key, 'char') else str(key)
    except AttributeError:
        key_char = str(key)
        
    with keystroke_lock:
        current_keystrokes.append({
            "key": key_char,
            "timeDown": int(time.time() * 1000),
            "timeUp": None,
            "pressure": 1.0  # Placeholder for pressure
        })

def on_release(key):
    """Handle key release events"""
    global current_keystrokes, is_collecting
    
    if not is_collecting:
        return
        
    try:
        key_char = key.char if hasattr(key, 'char') else str(key)
    except AttributeError:
        key_char = str(key)
        
    with keystroke_lock:
        # Find the matching keydown event
        for stroke in reversed(current_keystrokes):
            if stroke["key"] == key_char and stroke["timeUp"] is None:
                stroke["timeUp"] = int(time.time() * 1000)
                break

def start_keyboard_monitoring():
    """Start monitoring keyboard events"""
    global keyboard_listener
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()

def stop_keyboard_monitoring():
    """Stop monitoring keyboard events"""
    global keyboard_listener
    if keyboard_listener:
        keyboard_listener.stop()
        keyboard_listener = None

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

# Add new callbacks for live data collection
@callback(
    [Output("stop-collecting-btn", "disabled"),
     Output("start-collecting-btn", "disabled"),
     Output("collection-status", "children")],
    [Input("start-collecting-btn", "n_clicks"),
     Input("stop-collecting-btn", "n_clicks"),
     Input("collection-interval", "n_intervals")],
    [State("subject-name", "value")]
)
def handle_data_collection(start_clicks, stop_clicks, n_intervals, subject_name):
    global is_collecting, current_keystrokes
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, False, ""
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "start-collecting-btn" and start_clicks:
        if not subject_name:
            return True, False, html.P("Please enter a subject name", className="text-danger")
        
        is_collecting = True
        with keystroke_lock:
            current_keystrokes = []
        start_keyboard_monitoring()
        return False, True, html.P("Recording keystrokes...", className="text-success")
    
    elif button_id == "stop-collecting-btn" and stop_clicks:
        is_collecting = False
        stop_keyboard_monitoring()
        
        # Save the collected data
        if current_keystrokes:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"typing_data_{subject_name}_{timestamp}.json"
            
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)
                
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, 'w') as f:
                json.dump({
                    "subject": subject_name,
                    "timestamp": timestamp,
                    "keystrokes": current_keystrokes
                }, f, indent=2)
            
            return True, False, html.P(f"Data saved to {filename}", className="text-success")
        
        return True, False, html.P("No data collected", className="text-warning")
    
    elif button_id == "collection-interval" and is_collecting:
        return False, True, html.P(f"Recording... ({len(current_keystrokes)} keystrokes)", className="text-success")
    
    return True, False, ""

@callback(
    Output("collection-interval", "disabled"),
    Input("start-collecting-btn", "n_clicks"),
    Input("stop-collecting-btn", "n_clicks")
)
def toggle_interval(start_clicks, stop_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return button_id != "start-collecting-btn"

if __name__ == '__main__':
    print("Starting Cerebrum AI Typing Analysis Dashboard...")
    print("Dashboard available at: http://127.0.0.1:8050/")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}")
    
    app.run(debug=True) 