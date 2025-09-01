from __future__ import annotations
import pandas as pd
from cap_db import _conn

def _init_roles():
    with _conn() as cx:
        cx.execute("""
        CREATE TABLE IF NOT EXISTS user_roles (
            username TEXT PRIMARY KEY,
            role TEXT CHECK(role in ('admin','planner','viewer'))
        )
        """)
        cx.commit()

_init_roles()

def get_user_role(username: str | None) -> str:
    if not username:
        return 'viewer'
    with _conn() as cx:
        row = cx.execute("SELECT role FROM user_roles WHERE username=?", (username,)).fetchone()
        if row and row["role"] in ("admin","planner","viewer"):
            return row["role"]
    return 'viewer'

def set_user_role(username: str, role: str) -> None:
    assert role in ("admin","planner","viewer")
    with _conn() as cx:
        cx.execute("INSERT INTO user_roles(username,role) VALUES(?,?) ON CONFLICT(username) DO UPDATE SET role=excluded.role", (username, role))
        cx.commit()

def can_delete_plans(role: str) -> bool:
    return role == 'admin'

def can_save_settings(role: str) -> bool:
    return role in ('admin','planner')

