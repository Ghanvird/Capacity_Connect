from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from app_instance import app
from common import *  # noqa
from pages import (page_default_settings, page_roster, page_new_hire, page_shrink_attr,
    page_budget, page_dataset, page_planning, page_ops)

# ---------------------- Router + Home ----------------------
@callback(
    Output("tbl-projects", "data"),
    Input("cap-plans-refresh", "n_intervals"),
    State("url-router", "pathname"),
    prevent_initial_call=False
)
def _refresh_projects_table(_n, pathname):
    path = (pathname or "").rstrip("/")
    if path not in ("", "/"):
        raise PreventUpdate
    df = make_projects_sample()
    return df.to_dict("records")

def home_layout():
    return dbc.Container([
        header_bar(),
        dbc.Row([
            dbc.Col(left_capability_panel(), width=5),
            dbc.Col(center_projects_table(), width=4),
            dbc.Col(right_kpi_cards(), width=3, className="col-kpis"),
        ], className="g-3"),
    ], fluid=True)

def not_found_layout():
    return dbc.Container([header_bar(), dbc.Alert("Page not found.", color="warning"), dcc.Link("← Home", href="/")], fluid=True)

@app.callback(Output("root","children"), Input("url-router","pathname"))
def route(pathname: str):
    path = (pathname or "").rstrip("/")

    if path.startswith("/plan/"):
        try:
            pid = int(path.rsplit("/", 1)[-1])
        except Exception:
            return not_found_layout()
        return dbc.Container([header_bar(), layout_for_plan(pid)], fluid=True)

    if path in ("/", None, ""):
        return home_layout()

    pages = {
        "/settings": page_default_settings,
        "/roster":   page_roster,
        "/newhire":  page_new_hire,
        "/shrink":   page_shrink_attr,
        "/dataset":  page_dataset,
        "/planning": lambda: dbc.Container([header_bar(), planning_layout()], fluid=True),
        "/ops":      lambda: dbc.Container([header_bar(), dbc.Alert("Operational Dashboard — todo", color="info")], fluid=True),
        "/budget":   page_budget,
    }
    fn = pages.get(path)
    return fn() if fn else not_found_layout()
