#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 10:15:40 2025
@author: giordano (modified)
"""
# app.py
# ------
import dash
from dash import dcc, html, Input, Output, callback, dash_table, State
import plotly.express as px
import pandas as pd
import numpy as np
import colorsys
import json

UMAP_PARQUET = "translated_job_titles.parquet"

# -------- Custom color palette generation for many categories ----------
def generate_isco_grouped_colors(isco_categories):
    """
    Generate colors where similar ISCO codes get similar colors.
    ISCO-2D codes are two-digit codes where the first digit represents the major group.
    This function assigns similar hues to codes in the same major group.
    """
    # Convert string categories to integers if needed and sort
    try:
        # Try to convert to integers for proper numerical sorting
        isco_integers = [int(code) for code in isco_categories]
        isco_map = {str(code): code for code in isco_integers}
        sorted_codes = sorted(isco_integers)
        # Convert back to string format for the result
        sorted_isco = [str(code) for code in sorted_codes]
    except ValueError:
        # If conversion fails, use string sorting
        sorted_isco = sorted(isco_categories)
        isco_map = {code: i for i, code in enumerate(sorted_isco)}
    
    # Get the major groups (first digit of each code)
    # ISCO has 9 major groups (1-9)
    major_groups = set()
    for code in sorted_isco:
        if len(code) >= 1:
            major_groups.add(code[0])
    
    major_groups = sorted(list(major_groups))
    n_major_groups = len(major_groups)
    
    # Create a mapping from major group to hue range
    major_group_hue = {}
    for i, group in enumerate(major_groups):
        # Distribute hues around the color wheel
        # Use 0.8 of the wheel to avoid red appearing at both ends
        major_group_hue[group] = i * 0.8 / n_major_groups
    
    # Generate colors for each ISCO code
    colors = {}
    for code in sorted_isco:
        if len(code) >= 1:
            major_group = code[0]
            # Base hue for this major group
            base_hue = major_group_hue[major_group]
            
            # Add a small variation within the major group
            # Codes within the same major group will have slightly different hues
            minor_variation = 0
            if len(code) >= 2:
                try:
                    # Use second digit for minor variation within the major group's hue range
                    minor_digit = int(code[1])
                    # Scale to a small fraction of the total hue space
                    hue_range_per_group = 0.8 / n_major_groups
                    minor_variation = minor_digit * hue_range_per_group / 12  # Divide by ~10 for small variation
                except ValueError:
                    pass
            
            # Final hue combines major group position and minor variation
            hue = (base_hue + minor_variation) % 1.0
            
            # Create lighter colors for even-numbered groups and darker for odd
            # This adds another dimension of visual differentiation
            is_odd_group = int(major_group) % 2 == 1
            saturation = 0.85 if is_odd_group else 0.75
            value = 0.9 if is_odd_group else 0.95
            
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            colors[code] = hex_color
    
    return colors

# -------- load pre-computed UMAP data ----------
df = pd.read_parquet(UMAP_PARQUET)

# Get the unique ISCO-2D categories
unique_categories = df['isco2d'].unique()
n_categories = len(unique_categories)
print(f"Number of unique ISCO-2D categories: {n_categories}")

# Generate colors based on ISCO grouping
color_map = generate_isco_grouped_colors(unique_categories)

# Create a sorted list of categories for legend ordering
sorted_categories = sorted([str(cat) for cat in unique_categories], 
                          key=lambda x: int(x) if x.isdigit() else float('inf'))

# Create the scatter plot with our custom color palette
fig = px.scatter(
    df,
    x="x", y="y",
    color="isco2d",
    hover_data=["title_en", "isco2d"],
    title="ISCO-2D landscape of Russian vacancies",
    color_discrete_map=color_map,
    category_orders={"isco2d": sorted_categories}  # Order categories numerically
)

# Improve the marker appearance
fig.update_traces(
    marker=dict(
        size=6, 
        opacity=0.85,
        line=dict(width=0.5, color='DarkSlateGrey')
    ), 
    selector=dict(mode='markers')
)

# Improve layout for better readability
fig.update_layout(
    legend_title_text='Job Category (ISCO-2D)',
    legend=dict(
        itemsizing='constant',  # Make legend items same size
        title_font=dict(size=14),
        font=dict(size=12),
        traceorder='normal',  # Use the category order we defined
        itemwidth=30,
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='lightgrey',
        borderwidth=1
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        showgrid=True,
        gridcolor='lightgrey',
        title='UMAP Dimension 1',
        zeroline=False
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgrey',
        title='UMAP Dimension 2',
        zeroline=False
    ),
    margin=dict(r=150)  # Add more room for the legend
)

# -------- Dash layout ----------
app = dash.Dash(__name__)

# -------- Dash layout ----------
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Store for misclassified jobs
app.layout = html.Div([
    # Hidden div to store list of misclassified jobs
    dcc.Store(id='misclassified-jobs-store', data=[]),
    
    html.H2("Vacancy landscape (UMAP 2-D)", style={'textAlign': 'center'}),
    html.Div([
        # Add a description
        html.Div([
            html.P([
                "This visualization shows the landscape of job categories based on ISCO-2D codes. ",
                "Colors are grouped by major job categories (first digit of ISCO code), ",
                "with similar jobs having similar colors."
            ], style={'marginBottom': '10px'}),
            html.P([
                "Click on any point to see details below, including the specific ISCO-2D submajor group. ",
                "The legend on the right is sorted numerically by ISCO-2D code."
            ])
        ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        # Main visualization
        dcc.Graph(id="scatter", figure=fig, style={"height": "70vh"}),
    ], style={'width': '100%', 'display': 'inline-block'}),
    html.Hr(),
    
    # Information panel for clicked points
    html.Div([
        html.H4("Selected Job Details", style={'marginBottom': '10px'}),
        html.Div(id='click-info', style={
            "fontSize": 16, 
            "whiteSpace": "pre-line",
            "padding": "15px",
            "backgroundColor": "#f8f9fa",
            "borderRadius": "5px",
            "border": "1px solid #e9ecef"
        }),
        
        # Add button to mark as misclassified
        html.Div([
            html.Button(
                "Add to Misclassified List", 
                id="add-misclassified-btn",
                n_clicks=0,
                style={
                    "backgroundColor": "#dc3545",
                    "color": "white",
                    "border": "none",
                    "padding": "10px 15px",
                    "borderRadius": "5px",
                    "marginTop": "15px",
                    "cursor": "pointer"
                }
            ),
            html.Div(id="misclassified-feedback", style={"marginTop": "10px"})
        ]),
    ], style={'marginTop': '20px'}),
    
    html.Hr(),
    
    # Misclassified jobs table
    html.Div([
        html.H4("Potential Classification Errors", style={'marginBottom': '10px'}),
        html.Div(id="misclassified-count", style={"marginBottom": "10px"}),
        html.Div(id="misclassified-table-container"),
        html.Div([
            html.Button(
                "Export Misclassified Jobs", 
                id="export-btn",
                n_clicks=0,
                style={
                    "backgroundColor": "#007bff",
                    "color": "white",
                    "border": "none",
                    "padding": "10px 15px",
                    "borderRadius": "5px",
                    "marginTop": "15px",
                    "marginRight": "10px",
                    "cursor": "pointer"
                }
            ),
            html.Button(
                "Clear All", 
                id="clear-btn",
                n_clicks=0,
                style={
                    "backgroundColor": "#6c757d",
                    "color": "white",
                    "border": "none",
                    "padding": "10px 15px",
                    "borderRadius": "5px",
                    "marginTop": "15px",
                    "cursor": "pointer"
                }
            ),
            dcc.Download(id="download-misclassified"),
        ], style={"display": "flex", "alignItems": "center"})
    ], style={'marginTop': '30px', 'marginBottom': '50px'})
])

# Callback for misclassified button
@callback(
    [
        Output('misclassified-jobs-store', 'data'),
        Output('misclassified-feedback', 'children')
    ],
    [
        Input('add-misclassified-btn', 'n_clicks')
    ],
    [
        State('misclassified-jobs-store', 'data')
    ]
)
def add_to_misclassified(n_clicks, current_data):
    if n_clicks == 0 or len(last_clicked_data) == 0:
        return current_data, ""
    
    # Add a timestamp to the data
    import datetime
    misclassified_entry = last_clicked_data.copy()
    misclassified_entry['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    misclassified_entry['id'] = str(len(current_data) + 1)  # Add an ID for easy reference
    
    # Check if this entry already exists to avoid duplicates
    # We'll check based on title_en and isco2d for uniqueness
    for entry in current_data:
        if (entry.get('title_en') == misclassified_entry.get('title_en') and 
            entry.get('isco2d') == misclassified_entry.get('isco2d')):
            return current_data, html.Div(
                "This job is already in the misclassified list!",
                style={"color": "#dc3545", "fontWeight": "bold"}
            )
    
    # Add to the list
    updated_data = current_data + [misclassified_entry]
    
    return updated_data, html.Div(
        "Added to misclassified jobs list!",
        style={"color": "#28a745", "fontWeight": "bold"}
    )

# Callback to display misclassified jobs in a table
@callback(
    [
        Output('misclassified-table-container', 'children'),
        Output('misclassified-count', 'children')
    ],
    Input('misclassified-jobs-store', 'data')
)
def update_misclassified_table(data):
    if not data or len(data) == 0:
        return html.Div("No misclassified jobs added yet."), "Misclassified Jobs: 0"
    
    # Create dataframe from the stored data
    table_df = pd.DataFrame(data)
    
    # Select and reorder columns for display
    display_columns = ['id', 'isco2d', 'submajor_group_name', 'title', 'title_en', 'timestamp']
    
    # Make sure all expected columns exist
    for col in display_columns:
        if col not in table_df.columns:
            table_df[col] = ""  # Add empty column if missing
    
    # Select just the columns we want to display
    table_df = table_df[display_columns]
    
    # Rename columns for display
    column_rename = {
        'id': 'ID',
        'isco2d': 'ISCO-2D',
        'submajor_group_name': 'Job Category',
        'title': 'Original Title',
        'title_en': 'English Title',
        'timestamp': 'Added On'
    }
    
    # Create the table with properly named columns
    table = dash_table.DataTable(
        id='misclassified-table',
        columns=[{"name": column_rename.get(col, col), "id": col} for col in display_columns],
        data=table_df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'whiteSpace': 'normal',
            'height': 'auto',
            'minWidth': '100px', 'width': '150px', 'maxWidth': '200px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold',
            'border': '1px solid #ddd'
        },
        style_data={
            'border': '1px solid #ddd'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f9f9f9'
            }
        ],
        sort_action='native',
        filter_action='native',
        page_size=10,
        tooltip_data=[
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in table_df.to_dict('records')
        ],
        tooltip_duration=None,
    )
    
    count_message = f"Misclassified Jobs: {len(data)}"
    
    # Debug output to console
    print("Table data:", table_df.to_dict('records'))
    
    return table, count_message

# Callback for exporting data
@callback(
    Output('download-misclassified', 'data'),
    Input('export-btn', 'n_clicks'),
    State('misclassified-jobs-store', 'data'),
    prevent_initial_call=True
)
def export_misclassified(n_clicks, data):
    if not data or len(data) == 0:
        return None
    
    # Convert to dataframe
    export_df = pd.DataFrame(data)
    
    # Create a CSV string
    return dcc.send_data_frame(export_df.to_csv, 
                              "misclassified_jobs.csv", 
                              index=False)

# Callback for clearing the list
@callback(
    [
        Output('misclassified-jobs-store', 'data', allow_duplicate=True),
        Output('misclassified-feedback', 'children', allow_duplicate=True)
    ],
    Input('clear-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_misclassified(n_clicks):
    return [], html.Div(
        "Misclassified jobs list cleared!",
        style={"color": "#6c757d", "fontWeight": "bold"}
    )
# Store the last clicked point data
last_clicked_data = {}

@callback(
    Output('click-info', 'children'),
    Input('scatter', 'clickData')
)
def display_click(clickData):
    global last_clicked_data
    
    if not clickData:
        return "Click on a dot to see details here."
    
    # Get the clicked point data
    pt = clickData["points"][0]
    
    # Instead of using pointIndex, use the data from the point itself
    isco2d = pt["customdata"][1]  # We set hover_data=["title_en", "isco2d"], so isco2d is at index 1
    title_en = pt["customdata"][0]  # title_en is at index 0
    x_coord = pt["x"]
    y_coord = pt["y"]
    
    # Debug info to console
    print(f"Clicked point data: {pt}")
    
    # Find the matching row in the dataframe for additional info
    # We need to do this to get the original title, which isn't in the hover data
    matching_rows = df[(df["isco2d"] == isco2d) & 
                      (df["title_en"] == title_en) & 
                      (df["x"].round(3) == round(x_coord, 3)) & 
                      (df["y"].round(3) == round(y_coord, 3))]
    
    # Store this data for later use in the "Add to Misclassified" button
    last_clicked_data = {
        'isco2d': isco2d,
        'title_en': title_en,
        'x': x_coord,
        'y': y_coord
    }
    
    if len(matching_rows) > 0:
        # Use the first matching row
        row = matching_rows.iloc[0]
        last_clicked_data['title'] = row['title']  # Also store the original title
    else:
        # If no exact match found, create a partial data structure
        # This ensures we still show something even if there's a data mismatch
        row = pd.Series({
            'isco2d': isco2d,
            'title_en': title_en,
            'title': "Original title not available",
            'x': x_coord,
            'y': y_coord
        })
        last_clicked_data['title'] = "Original title not available"
    
    # Get the color for this category
    cat_color = color_map.get(isco2d, '#333333')
    
    # ISCO-2D submajor groups (2-digit level) - same as before
    isco_submajor_groups = {
        "11": "Chief executives, senior officials and legislators",
        "12": "Administrative and commercial managers",
        "13": "Production and specialized services managers",
        "14": "Hospitality, retail and other services managers",
        "21": "Science and engineering professionals",
        "22": "Health professionals",
        "23": "Teaching professionals",
        "24": "Business and administration professionals",
        "25": "Information and communications technology professionals",
        "26": "Legal, social and cultural professionals",
        "31": "Science and engineering associate professionals",
        "32": "Health associate professionals",
        "33": "Business and administration associate professionals",
        "34": "Legal, social, cultural and related associate professionals",
        "35": "Information and communications technicians",
        "41": "General and keyboard clerks",
        "42": "Customer services clerks",
        "43": "Numerical and material recording clerks",
        "44": "Other clerical support workers",
        "51": "Personal service workers",
        "52": "Sales workers",
        "53": "Personal care workers",
        "54": "Protective services workers",
        "61": "Market-oriented skilled agricultural workers",
        "62": "Market-oriented skilled forestry, fishery and hunting workers",
        "63": "Subsistence farmers, fishers, hunters and gatherers",
        "71": "Building and related trades workers, excluding electricians",
        "72": "Metal, machinery and related trades workers",
        "73": "Handicraft and printing workers",
        "74": "Electrical and electronic trades workers",
        "75": "Food processing, wood working, garment and other craft and related trades workers",
        "81": "Stationary plant and machine operators",
        "82": "Assemblers",
        "83": "Drivers and mobile plant operators",
        "91": "Cleaners and helpers",
        "92": "Agricultural, forestry and fishery labourers",
        "93": "Labourers in mining, construction, manufacturing and transport",
        "94": "Food preparation assistants",
        "95": "Street and related sales and service workers",
        "96": "Refuse workers and other elementary workers",
        "101": "Commissioned armed forces officers",
        "102": "Non-commissioned armed forces officers",
        "103": "Armed forces occupations, other ranks"
    }
    
    # Store the submajor group name for later use
    submajor_group = isco2d
    submajor_group_name = isco_submajor_groups.get(submajor_group, "")
    last_clicked_data['submajor_group_name'] = submajor_group_name
    
    # If we can't find the exact submajor, fall back to the major group
    if not submajor_group_name and len(submajor_group) >= 1:
        major_group = submajor_group[0] if submajor_group[0] != '1' else ('10' if len(submajor_group) >= 2 and submajor_group[1] == '0' else '1')
        isco_major_groups = {
            "1": "Managers",
            "2": "Professionals",
            "3": "Technicians and Associate Professionals",
            "4": "Clerical Support Workers",
            "5": "Service and Sales Workers",
            "6": "Skilled Agricultural, Forestry and Fishery Workers",
            "7": "Craft and Related Trades Workers",
            "8": "Plant and Machine Operators and Assemblers",
            "9": "Elementary Occupations",
            "10": "Armed Forces Occupations"
        }
        submajor_group_name = f"(In {isco_major_groups.get(major_group, 'Unknown Group')})"
        last_clicked_data['submajor_group_name'] = submajor_group_name
    
    print(f"Last clicked data: {last_clicked_data}")  # Debug info
    
    return html.Div([
        # ISCO code with color coding and submajor group name - same as before
        html.Div([
            html.Span("ISCO-2D: ", style={"fontWeight": "bold"}),
            html.Span(f"{isco2d}", style={"fontWeight": "bold"}),
            html.Div(submajor_group_name, style={
                "fontSize": "14px", 
                "marginTop": "3px",
                "fontStyle": "italic"
            })
        ], style={
            "borderLeft": f"5px solid {cat_color}", 
            "paddingLeft": "10px",
            "paddingTop": "5px",
            "paddingBottom": "5px",
            "backgroundColor": f"{cat_color}22"  # Add very light background of same color
        }),
        
        # Job title info - same as before
        html.Div([
            html.Div([
                html.Span("Original Title: ", style={"fontWeight": "bold"}),
                html.Span(f"{row['title']}")
            ], style={"marginTop": "10px"}),
            
            html.Div([
                html.Span("English Title: ", style={"fontWeight": "bold"}),
                html.Span(f"{row['title_en']}")
            ], style={"marginTop": "5px"}),
            
            html.Div([
                html.Span("UMAP Coordinates: ", style={"fontWeight": "bold"}),
                html.Span(f"({x_coord:.3f}, {y_coord:.3f})")
            ], style={"marginTop": "5px"})
        ])
    ])

server = app.server  # crucial for Gunicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)
