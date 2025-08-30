from __future__ import annotations
from dash import html, dcc, dash_table
from app_instance import app, server
from router import home_layout, not_found_layout
from common import _planning_ids_skeleton
from common import header_bar, sidebar_component
from planning_workspace import planning_layout, register_planning_ws
from plan_detail import plan_detail_validation_layout, register_plan_detail
from callbacks_pkg import *  # registers callbacks


# ---- Main Layout (verbatim) ----
app.layout = html.Div([
    dcc.Location(id="url-router"),
    dcc.Store(id="sidebar_collapsed", data=True, storage_type="session"),
    _planning_ids_skeleton(),
    html.Div(id="app-wrapper", className="sidebar-collapsed", children=[
        html.Div(id="sidebar", children=sidebar_component(False).children),
        html.Div(id="root")
    ]),
    dcc.Interval(id="cap-plans-refresh", interval=5000, n_intervals=0)
])


# ---- Validation Layout (verbatim) ----
# ✅ VALIDATION LAYOUT — include plan-detail skeleton
app.validation_layout = html.Div([
    dcc.Location(id="url-router"),
    dcc.Store(id="sidebar_collapsed"),
    dcc.Store(id="ws-status"),
    dcc.Store(id="ws-selected-ba"),
    dcc.Store(id="ws-refresh"),
    header_bar(),
    planning_layout(),
    plan_detail_validation_layout(),
    dash_table.DataTable(id="tbl-projects"),
])

# Register callbacks (planning + plan-detail)
register_planning_ws(app)
register_plan_detail(app)

# ---- Entrypoint (verbatim) ----
# ---------------------- Main ----------------------
if __name__ == "__main__":
    app.run(debug=True)
