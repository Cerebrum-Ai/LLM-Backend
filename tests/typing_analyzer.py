import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from pynput import keyboard
import threading
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from transformers import AutoModel, AutoTokenizer
from models.typing.pattern.typing_processor import KeystrokeProcessor  # Import the correct class

class TypingDataCollector:
    def __init__(self):
        self.keystrokes = []
        self.start_time = None
        self.listener = None
        self.is_recording = False

    def on_press(self, key):
        if not self.is_recording:
            return
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)
        
        self.keystrokes.append({
            'key': key_char,
            'timeDown': int(time.time() * 1000),
            'timeUp': None,
            'pressure': 1.0
        })

    def on_release(self, key):
        if not self.is_recording:
            return
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)
        
        # Find the last keystroke for this key that hasn't been released
        for stroke in reversed(self.keystrokes):
            if stroke['key'] == key_char and stroke['timeUp'] is None:
                stroke['timeUp'] = int(time.time() * 1000)
                break

    def start_recording(self):
        self.keystrokes = []
        self.start_time = time.time()
        self.is_recording = True
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def stop_recording(self):
        self.is_recording = False
        if self.listener:
            self.listener.stop()
        return {
            'subject': 'User',
            'timestamp': time.strftime("%Y%m%d_%H%M%S"),
            'keystrokes': self.keystrokes
        }

def load_typing_data(file_path):
    """Load typing data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def process_typing_data(data):
    """Process typing data into a pandas DataFrame."""
    keystrokes = data['keystrokes']
    df = pd.DataFrame(keystrokes)
    
    # Calculate duration for each keystroke
    df['duration'] = df['timeUp'] - df['timeDown']
    
    # Calculate time between keystrokes
    df['time_since_last'] = df['timeDown'].diff()
    
    # Add cumulative time
    df['cumulative_time'] = df['timeDown'] - df['timeDown'].iloc[0]
    
    return df

def get_data_files():
    """Get list of JSON files from the data directory."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return [f for f in os.listdir(data_dir) if f.endswith('.json')]

# Load the typing model
def load_typing_model():
    model_path = os.path.join('models', 'typing', 'pattern', 'typing_pattern_model.pkl')
    scaler_path = os.path.join('models', 'typing', 'pattern', 'typing_scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print("Model and scaler loaded successfully")
            return model, scaler
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
    else:
        print(f"Model files not found at {model_path} or {scaler_path}")
        return None, None

# Initialize model and scaler
model, scaler = load_typing_model()

# Initialize the keystroke processor
keystroke_processor = KeystrokeProcessor.get_instance()

def extract_typing_features(data):
    """Extract features from typing data for disease prediction."""
    try:
        # Use the keystroke processor to analyze the data
        result = keystroke_processor.analyze_typing(data)
        return result
    except Exception as e:
        print(f"Error processing typing data: {e}")
        return None

def predict_disease(data):
    """Predict disease based on typing patterns."""
    try:
        # Get analysis from the keystroke processor
        result = extract_typing_features(data)
        if result is None:
            return "Error processing typing data", 0.0
            
        # Map the condition to a display message
        condition = result.get('condition', 'unknown')
        confidence = result.get('confidence', 0.0)
        
        condition_map = {
            'normal': "No significant indicators detected",
            'parkinsons': "Potential indicators of Parkinson's disease",
            'essential_tremor': "Potential indicators of Essential Tremor",
            'carpal_tunnel': "Potential indicators of Carpal Tunnel Syndrome",
            'multiple_sclerosis': "Potential indicators of Multiple Sclerosis",
            'unknown': "Unable to determine condition"
        }
        
        return condition_map.get(condition, "Analysis inconclusive"), confidence
    except Exception as e:
        print(f"Error making prediction: {e}")
        return "Error making prediction", 0.0

def create_typing_visualization():
    """Create interactive visualization of typing patterns."""
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Initialize the data collector
    data_collector = TypingDataCollector()
    
    app.layout = dbc.Container([
        # Header with title and description
        dbc.Row([
            dbc.Col([
                html.H1("Typing Analysis Dashboard", className="text-center mb-4"),
                html.P("Analyze your typing patterns, speed, and accuracy with detailed visualizations.", 
                      className="text-center text-muted mb-4"),
                html.Hr(className="my-4")
            ])
        ]),
        
        # Data Selection Card
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Data Selection", className="mb-0"),
                        html.Small("Choose how to analyze typing data", className="text-muted")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            # Existing Files Selection
                            dbc.Col([
                                html.H5("Select from existing files:", className="mb-3"),
                                dcc.Dropdown(
                                    id='data-file-dropdown',
                                    options=[{'label': f, 'value': f} for f in get_data_files()],
                                    placeholder="Select a typing data file",
                                    className="mb-3"
                                )
                            ], width=4),
                            
                            # File Upload
                            dbc.Col([
                                html.H5("Upload a new file:", className="mb-3"),
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div([
                                        html.I(className="bi bi-cloud-upload me-2"),
                                        'Drag and Drop or ',
                                        html.A('Select Files', className="text-primary")
                                    ]),
                                    className="upload-area",
                                    multiple=False
                                )
                            ], width=4),
                            
                            # Live Recording
                            dbc.Col([
                                html.H5("Record live typing:", className="mb-3"),
                                dbc.ButtonGroup([
                                    dbc.Button([
                                        html.I(className="bi bi-record-circle me-2"),
                                        "Start Recording"
                                    ], id="start-recording", color="danger", className="me-2"),
                                    dbc.Button([
                                        html.I(className="bi bi-stop-circle me-2"),
                                        "Stop Recording"
                                    ], id="stop-recording", color="secondary")
                                ], className="mb-2"),
                                html.Div(id="recording-status", className="mt-2")
                            ], width=4)
                        ])
                    ])
                ], className="mb-4 shadow-sm")
            ])
        ]),
        
        # Statistics Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Total Keystrokes", className="card-title"),
                        html.H3(id="total-keystrokes", className="card-text text-primary")
                    ])
                ], className="mb-4 shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Average Duration", className="card-title"),
                        html.H3(id="avg-duration", className="card-text text-success")
                    ])
                ], className="mb-4 shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Typing Speed", className="card-title"),
                        html.H3(id="typing-speed", className="card-text text-info")
                    ])
                ], className="mb-4 shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Accuracy", className="card-title"),
                        html.H3(id="accuracy", className="card-text text-warning")
                    ])
                ], className="mb-4 shadow-sm")
            ], width=3)
        ]),
        
        # Disease Prediction Card
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Disease Prediction", className="mb-0"),
                        html.Small("Analysis based on typing patterns", className="text-muted")
                    ]),
                    dbc.CardBody([
                        html.Div(id="prediction-result", className="text-center mb-3"),
                        dbc.Progress(id="prediction-confidence", className="mb-3"),
                        html.Div(id="prediction-details", className="text-muted")
                    ])
                ], className="mb-4 shadow-sm")
            ])
        ]),
        
        # Visualization Row 1
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Keystroke Duration Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id='duration-plot')
                    ])
                ], className="mb-4 shadow-sm")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Typing Pattern Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id='timing-plot')
                    ])
                ], className="mb-4 shadow-sm")
            ], width=6)
        ]),
        
        # Visualization Row 2
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Key Pressure Distribution"),
                    dbc.CardBody([
                        dcc.Graph(id='pressure-plot')
                    ])
                ], className="mb-4 shadow-sm")
            ], width=12)
        ]),
        
        # Store for data
        dcc.Store(id='stored-data')
    ], fluid=True, className="py-4")
    
    # Custom CSS
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/font/bootstrap-icons.css">
            <style>
                .upload-area {
                    width: 100%;
                    height: 100px;
                    border: 2px dashed #dee2e6;
                    border-radius: 5px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }
                .upload-area:hover {
                    border-color: #0d6efd;
                    background-color: #f8f9fa;
                }
                .card {
                    transition: transform 0.2s;
                }
                .card:hover {
                    transform: translateY(-5px);
                }
                .text-primary { color: #0d6efd !important; }
                .text-success { color: #198754 !important; }
                .text-info { color: #0dcaf0 !important; }
                .text-warning { color: #ffc107 !important; }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    def parse_uploaded_file(contents):
        """Parse uploaded file contents."""
        if contents is None:
            return None
            
        content_type, content_string = contents.split(',')
        decoded = json.loads(content_string)
        return decoded
    
    @app.callback(
        [Output('stored-data', 'data'),
         Output('recording-status', 'children'),
         Output('total-keystrokes', 'children'),
         Output('avg-duration', 'children'),
         Output('typing-speed', 'children'),
         Output('accuracy', 'children')],
        [Input('upload-data', 'contents'),
         Input('data-file-dropdown', 'value'),
         Input('start-recording', 'n_clicks'),
         Input('stop-recording', 'n_clicks')],
        [State('upload-data', 'filename')]
    )
    def update_stored_data(contents, selected_file, start_clicks, stop_clicks, filename):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'upload-data' and contents:
            data = parse_uploaded_file(contents)
            stats = calculate_statistics(data)
            return data, "", *stats
        elif trigger_id == 'data-file-dropdown' and selected_file:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
            data = load_typing_data(os.path.join(data_dir, selected_file))
            stats = calculate_statistics(data)
            return data, "", *stats
        elif trigger_id == 'start-recording':
            data_collector.start_recording()
            return None, html.Div([
                html.I(className="bi bi-record-circle-fill text-danger me-2"),
                "Recording in progress..."
            ], className="text-danger"), "0", "0ms", "0 WPM", "0%"
        elif trigger_id == 'stop-recording':
            data = data_collector.stop_recording()
            stats = calculate_statistics(data)
            return data, html.Div([
                html.I(className="bi bi-check-circle-fill text-success me-2"),
                "Recording stopped"
            ], className="text-success"), *stats
            
        raise PreventUpdate
    
    def calculate_statistics(data):
        if not data:
            return "0", "0ms", "0 WPM", "0%"
            
        df = process_typing_data(data)
        
        # Total keystrokes
        total_keys = len(df)
        
        # Average duration
        avg_duration = f"{df['duration'].mean():.0f}ms"
        
        # Typing speed (WPM)
        total_time = (df['timeUp'].max() - df['timeDown'].min()) / 1000  # in seconds
        words = len(''.join(df['key'].tolist()).split())  # rough word count
        wpm = (words / total_time) * 60 if total_time > 0 else 0
        typing_speed = f"{wpm:.1f} WPM"
        
        # Accuracy (placeholder - you might want to implement a more sophisticated accuracy calculation)
        accuracy = "95%"  # This is a placeholder
        
        return str(total_keys), avg_duration, typing_speed, accuracy
    
    @app.callback(
        [Output('duration-plot', 'figure'),
         Output('timing-plot', 'figure'),
         Output('pressure-plot', 'figure')],
        [Input('stored-data', 'data')]
    )
    def update_plots(data):
        if not data:
            return {}, {}, {}
            
        df = process_typing_data(data)
        
        # Duration plot with improved styling
        duration_fig = px.bar(df, x='key', y='duration', 
                            title='Keystroke Duration by Key',
                            labels={'duration': 'Duration (ms)', 'key': 'Key'},
                            color='duration',
                            color_continuous_scale='Viridis')
        duration_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        # Timing plot with improved styling
        timing_fig = px.scatter(df, x='cumulative_time', y='time_since_last',
                              title='Time Between Keystrokes',
                              labels={'cumulative_time': 'Time (ms)', 'time_since_last': 'Time Since Last Key (ms)'},
                              color='time_since_last',
                              color_continuous_scale='Plasma')
        timing_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Pressure plot with improved styling
        pressure_fig = px.bar(df, x='key', y='pressure',
                            title='Key Pressure Distribution',
                            labels={'pressure': 'Pressure', 'key': 'Key'},
                            color='pressure',
                            color_continuous_scale='Inferno')
        pressure_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        return duration_fig, timing_fig, pressure_fig
    
    @app.callback(
        [Output('prediction-result', 'children'),
         Output('prediction-confidence', 'value'),
         Output('prediction-confidence', 'label'),
         Output('prediction-details', 'children')],
        [Input('stored-data', 'data')]
    )
    def update_prediction(data):
        if not data:
            return "No data available", 0, "0%", "Please select or record typing data"
            
        prediction, probability = predict_disease(data)
        confidence = int(probability * 100)
        
        # Determine color based on prediction
        color = "danger" if "Potential" in prediction else "success"
        
        return [
            html.H3(prediction, className=f"text-{color}"),
            confidence,
            f"{confidence}%",
            html.P([
                "This prediction is based on analysis of:",
                html.Ul([
                    html.Li("Typing speed and rhythm"),
                    html.Li("Key press duration"),
                    html.Li("Error patterns"),
                    html.Li("Typing consistency")
                ])
            ])
        ]
    
    return app

if __name__ == "__main__":
    # Create the visualization app
    app = create_typing_visualization()
    app.run(debug=True) 