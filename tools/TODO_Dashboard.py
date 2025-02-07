# Problems with this dashboard:
# metric and loss plots don't fit the grid
# dashboard doesn't delete plots from old epochs


import os
from pathlib import Path
import base64
from dash import Dash, dcc, html, Output, Input
import dash

# Directory to monitor
MONITOR_DIR = "path_to_your_directory"

# Initialize the Dash app
app = Dash(__name__)

def get_images_from_directory(directory):
    """Retrieve all images from the directory and its subdirectories."""
    supported_formats = (".png", ".jpg", ".jpeg", ".gif", ".svg")
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                file_path = os.path.join(root, file)
                image_files.append(file_path)
    return image_files

def create_layout():
    """Generate the app layout dynamically based on images."""
    return html.Div([
        html.H1("Dynamic Image Dashboard"),
        dcc.Interval(id="interval-component", interval=5000, n_intervals=0),  # Refresh every 5 seconds
        html.Div(id="image-container", style={'display': 'flex', 'flex-wrap': 'wrap'})
    ])

def create_image_elements(images):
    """Generate HTML elements for each image."""
    image_elements = []
    for image_path in images:
        # Encode image for Dash display
        encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
        image_elements.append(html.Img(src=f"data:image/png;base64,{encoded_image}", style={'width': '300px', 'margin': '10px'}))
    return image_elements

# App layout
app.layout = create_layout()

@app.callback(
    Output("image-container", "children"),
    Input("interval-component", "n_intervals")
)
def update_images(n_intervals):
    """Update the images in the dashboard."""
    images = get_images_from_directory(MONITOR_DIR)
    return create_image_elements(images)

if __name__ == "__main__":
    # Run the Dash app
    app.run_server(debug=True)
