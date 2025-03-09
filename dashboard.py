import pandas as pd
import dash
from dash import dash_table, html, dcc, Input, Output
import plotly.express as px

player_point = pd.read_csv("predictions.csv")

team_options = [{'label': team, 'value': team} for team in sorted(player_point['TEAM'].unique())]
team_options.insert(0, {'label': 'All Teams', 'value': 'all'})


# Dash application
app = dash.Dash(__name__)

# app layout
app.layout = html.Div(children=[html.H1('Current NBA Player Stat Predictions', 
                                        style={'textAlign': 'center', 'font-size': 40}),
                                html.Div([
                                    dcc.Input(
                                        id='player-search',
                                        type='text',
                                        placeholder='Search by player name...',
                                        style={'mmarginRight': '20px', 'width': '300px'}
                                    ),
                                    dcc.Dropdown(
                                        id='team-filter',
                                        options=team_options,
                                        value='all',
                                        clearable=False,
                                        style={'width': '200px'}
                                    )
                                    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
                                dash_table.DataTable(
                                    id='datatable-interactivity',
                                    columns=[{"name": i, "id": i} for i in player_point.columns],
                                    data=player_point.to_dict('records'),
                                    #filter_action="native", # built-in filtering
                                    sort_action="native", # built-in sorting
                                    sort_mode="multi",
                                    page_action="native",
                                    page_current=0,
                                    page_size=10,
                                    style_table={'overflowX': 'auto', 'backgroundColor': '#2a2a2a'},
                                    style_cell={
                                        'minWidth': '100px', 'width': '150px', 'maxWidth': '200px',
                                        'whiteSpace': 'normal'
                                    },
                                    style_header={
                                        'backgroundColor': '#333', 'color': '#f2f2f2', 'fontWeight': 'bold'
                                    }
                                ),
    ])

@app.callback(
    Output('datatable-interactivity', 'data'),
    [Input('player-search', 'value'),
    Input('team-filter', 'value')]
)

def update_table(player_search, team_filter):
    filtered_df = player_point.copy()
    if player_search:
        filtered_df = filtered_df[filtered_df['PLAYER'].str.contains(player_search, case=False, na=False)]
    if team_filter and team_filter != 'all':
        filtered_df = filtered_df[filtered_df['TEAM'] == team_filter]
    return filtered_df.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True)