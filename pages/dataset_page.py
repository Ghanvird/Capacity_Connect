from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from app_instance import app
from common import *  # noqa

def page_dataset():
    roster = load_roster()
    hire   = load_hiring()
    sup = supply_fte_daily(roster if not roster.empty else roster_demo,
                           hire if not hire.empty else hiring_demo)
    df = pd.merge(req_df, sup, on=["date","program"], how="left").fillna({"supply_fte":0})
    df["staffing_pct"] = np.where(df["total_req_fte"]>0, (df["supply_fte"]/df["total_req_fte"])*100, np.nan)
    return dbc.Container([
        header_bar(),
        html.H5("Planner Dataset — Inputs Snapshot", className="mkc"),
        dash_table.DataTable(
            id="dataset-table",
            data=df.to_dict("records"),
            columns=pretty_columns(df),
            page_size=12, style_table={"overflowX":"auto"}, style_as_list_view=True,
            style_header={"textTransform": "none"},
        ),
        dcc.Graph(figure=px.line(df.groupby("date",as_index=False)[["total_req_fte","supply_fte"]].sum(),
                                 x="date", y=["total_req_fte","supply_fte"], markers=True), className="mt-3"),
    ], fluid=True)


