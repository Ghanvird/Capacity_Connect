from __future__ import annotations
from dash import dcc
from planning_workspace import planning_layout

def page_planning():
    return dcc.Loading(planning_layout(), className="loading-page")
