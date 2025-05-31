import base64
import io
import json
import os
from datetime import datetime
import cv2

import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_canvas
import numpy as np
from PIL import Image
import torch

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Image Segmentation Labeling Tool'),
    
    # Upload component
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select an Image')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    
    # Canvas for drawing
    html.Div(id='canvas-container', children=[
        dash_canvas.DashCanvas(
            id='canvas',
            tool='line',
            lineWidth=5,
            lineColor='red',
            width=800,
            height=600,
            hide_buttons=['zoom', 'pan'],
            goButtonTitle='Save Mask'
        )
    ], style={'display': 'none'}),
    
    # Save controls
    html.Div([
        html.Button('Save to Dataset', id='save-button', n_clicks=0),
        html.Div(id='save-status')
    ]),
    
    # Store the original image data
    dcc.Store(id='original-image-store'),
])

def save_to_dataset(image_array, mask_array, filename):
    """Save the image and mask to a PyTorch-compatible dataset format"""
    # Create dataset directory if it doesn't exist
    os.makedirs('dataset/images', exist_ok=True)
    os.makedirs('dataset/masks', exist_ok=True)
    
    # Save image
    image = Image.fromarray(image_array)
    image.save(f'dataset/images/{filename}.png')
    
    # Save mask
    mask = Image.fromarray(mask_array)
    mask.save(f'dataset/masks/{filename}_mask.png')
    
    return True

@app.callback(
    [Output('canvas', 'image_content'),
     Output('canvas-container', 'style'),
     Output('original-image-store', 'data')],
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def update_canvas(contents, filename):
    if contents is None:
        raise PreventUpdate
    
    # Decode the image
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded))
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to fit canvas while maintaining aspect ratio
    max_size = (800, 600)
    img.thumbnail(max_size, Image.LANCZOS)
    
    # Convert back to base64 for display
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Store the original image array
    img_array = np.array(img)
    
    return (
        f'data:image/png;base64,{img_str}',
        {'display': 'block'},
        json.dumps({
            'array': img_array.tolist(),
            'filename': filename
        })
    )

@app.callback(
    Output('save-status', 'children'),
    Input('save-button', 'n_clicks'),
    [State('canvas', 'json_data'),
     State('original-image-store', 'data')],
    prevent_initial_call=True
)
def save_annotation(n_clicks, json_data, original_image_data):
    if not n_clicks or not json_data or not original_image_data:
        raise PreventUpdate
    
    # Load the original image data
    image_data = json.loads(original_image_data)
    image_array = np.array(image_data['array'])
    filename = os.path.splitext(image_data['filename'])[0]
    
    # Convert the canvas data to a binary mask
    mask_data = json.loads(json_data)
    mask = np.array(mask_data['objects'][0]['path'])
    h, w = image_array.shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create binary mask from the drawing
    for i in range(len(mask)-1):
        pt1 = tuple(map(int, mask[i]))
        pt2 = tuple(map(int, mask[i+1]))
        binary_mask = cv2.line(binary_mask, pt1, pt2, 255, 5)
    
    # Save to dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_filename = f'{filename}_{timestamp}'
    
    if save_to_dataset(image_array, binary_mask, save_filename):
        return html.Div('Successfully saved to dataset!', style={'color': 'green'})
    else:
        return html.Div('Error saving to dataset', style={'color': 'red'})

if __name__ == '__main__':
    app.run_server(debug=True)