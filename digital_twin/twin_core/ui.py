"""Shared look & feel: colours, cards, chart styling."""
import pandas as pd

TEAL, BLUE, INK, MUTED, LINE = "#0E9F8E", "#2F80ED", "#1F2D3D", "#6B7C93", "#E6EEF5"
GREEN, AMBER, RED, GREY, PURPLE = "#16A34A", "#F59E0B", "#E5484D", "#A3B1C2", "#8B5CF6"
STATUS_COL = {"normal": GREEN, "warning": AMBER, "fault": RED, "tripped": "#7F1D1D"}
FONT = "Inter, 'Segoe UI', Roboto, Arial, sans-serif"
TURBINE_COL = {0: "#2F80ED", 10: "#0E9F8E", 11: "#8B5CF6", 13: "#F59E0B", 21: "#E5484D"}

CSS = f"""<style>
.stApp {{background: linear-gradient(180deg,#EAF4FA 0%,#F6FAFD 60%);}}
section[data-testid="stSidebar"] {{background:#ffffffee;border-right:1px solid {LINE};}}
div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {{background:#fff;border-radius:14px;padding:6px 8px;
  border:1px solid {LINE};box-shadow:0 4px 16px rgba(30,80,120,.06);}}
#MainMenu, footer {{visibility:hidden;}}
[data-testid="stAppDeployButton"], .stDeployButton {{display:none !important;}}
.block-container {{padding-top:1.2rem;}}
.tw {{font-family:{FONT};color:{INK};}}
.tw .card {{background:#fff;border-radius:14px;border:1px solid {LINE};box-shadow:0 4px 16px rgba(30,80,120,.06);padding:12px 14px;margin-bottom:10px}}
.tw .k {{color:{MUTED};font-size:12px}} .tw .v {{font-weight:800;font-size:22px}}
.tw .pill {{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;font-weight:700}}
.tw table.r {{width:100%;border-collapse:collapse;font-size:13px}}
.tw table.r td {{padding:4px 2px;border-bottom:1px dashed {LINE}}} .tw table.r td:last-child {{text-align:right;font-weight:700}}
</style>"""


def html(st, s):
    st.markdown('<div class="tw">' + " ".join(x.strip() for x in s.splitlines() if x.strip()) + "</div>",
                unsafe_allow_html=True)


def pill(text, colour):
    return f'<span class="pill" style="background:{colour}22;color:{colour}">{text}</span>'


def kpis(st, items):
    cells = "".join(f'<div class="card" style="flex:1;min-width:150px"><div class="k">{k}</div><div class="v">{v}</div>'
                    f'<div class="k">{s}</div></div>' for k, v, s in items)
    html(st, f'<div style="display:flex;gap:12px;flex-wrap:wrap">{cells}</div>')


def title(st, text, sub=None):
    html(st, f"<div style='font-size:22px;font-weight:800;margin:4px 0 2px'>{text}</div>"
             + (f"<div style='color:{MUTED};font-size:13px;margin-bottom:10px'>{sub}</div>" if sub else ""))


def style(fig, title=None, height=320, ytitle=None, xtitle=None):
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=INK, family=FONT), x=0.01) if title else None,
                      height=height, margin=dict(l=10, r=10, t=44 if title else 10, b=10), paper_bgcolor="white",
                      plot_bgcolor="white", font=dict(family=FONT, color=INK, size=12), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
                      yaxis_title=ytitle, xaxis_title=xtitle, uirevision="keep")
    fig.update_xaxes(showgrid=True, gridcolor=LINE, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=LINE, zeroline=False)
    return fig


def fmt_td(td):
    if td is None or pd.isna(td):
        return "-"
    h = td.total_seconds() / 3600
    return f"{h:.1f} h"
