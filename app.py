"""
Interactive dashboard for the detected S&P 500 regimes.

Run the analysis first (it creates outputs/), then:
    python app.py
and open http://127.0.0.1:8050 in a browser.

The dashboard only READS outputs/; it never re-fits the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

# Colours checked for colour-blind separation. For k <= 3, regimes are coloured
# by volatility rank (lowest -> teal, middle -> amber, highest -> red).
# For k > 3 a general categorical palette is used in the same rank order.
COLORS_K3 = ["#1f8a70", "#e0a030", "#c8414b"]
COLORS_GENERAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK_MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_outputs():
    """Return (regimes df, profile df, metrics dict) or raise FileNotFoundError."""
    needed = ["regimes.csv", "regime_profile.csv", "metrics.json"]
    missing = [f for f in needed if not (OUT_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {OUT_DIR}. Run: python run_analysis.py")
    df = pd.read_csv(OUT_DIR / "regimes.csv", index_col="Date", parse_dates=["Date"])
    profile = pd.read_csv(OUT_DIR / "regime_profile.csv")
    metrics = json.loads((OUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    return df, profile, metrics


def regime_colors(profile: pd.DataFrame) -> dict[str, str]:
    """Map each regime label to a colour, ordered by average 20-day volatility."""
    ordered = profile.sort_values("vol_20")["regime"].tolist()
    palette = COLORS_K3 if len(ordered) <= 3 else COLORS_GENERAL
    return {r: palette[i % len(palette)] for i, r in enumerate(ordered)}


def base_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15, color=INK)),
        height=height,
        template="plotly_white",
        margin=dict(l=50, r=20, t=50, b=70),
        font=dict(color=INK_MUTED, size=12),
        legend=dict(orientation="h", y=-0.12, x=0, xanchor="left", yanchor="top"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def price_figure(df, colors, split_date, log_scale: bool) -> go.Figure:
    fig = go.Figure()
    # Thin grey line for continuity, coloured markers for the regime of each day.
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="S&P 500",
                             line=dict(color="#b8bec5", width=1), hoverinfo="skip",
                             showlegend=False))
    for regime, color in colors.items():
        part = df[df["regime"] == regime]
        fig.add_trace(go.Scatter(
            x=part.index, y=part["close"], mode="markers", name=regime,
            marker=dict(color=color, size=4),
            hovertemplate="%{x|%d %b %Y}<br>Close %{y:,.0f}<br>" + regime + "<extra></extra>",
        ))
    if split_date is not None:
        fig.add_shape(type="line", x0=split_date, x1=split_date, y0=0, y1=1,
                      xref="x", yref="paper", line=dict(color=INK_MUTED, width=1, dash="dot"))
        fig.add_annotation(x=split_date, y=0.98, xref="x", yref="paper", yanchor="top",
                           xanchor="right", xshift=-4, text="out-of-sample from here",
                           showarrow=False,
                           font=dict(size=11, color=INK_MUTED))
    base_layout(fig, "S&P 500 coloured by detected regime", height=460)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[df.index.min(), df.index.max()])  # labels can't stretch the axis
    fig.update_yaxes(type="log" if log_scale else "linear", title="Index level")
    return fig


def volatility_figure(df) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["vol_20"], name="20-day", mode="lines",
                             line=dict(color="#2a78d6", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["vol_60"], name="60-day", mode="lines",
                             line=dict(color="#eb6834", width=1.5)))
    base_layout(fig, "Annualised rolling volatility")
    fig.update_yaxes(tickformat=".0%")
    return fig


def drawdown_figure(df) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["drawdown"], name="From all-time high",
                             mode="lines", fill="tozeroy", line=dict(color="#c8414b", width=1),
                             fillcolor="rgba(200,65,75,0.15)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["mdd_60"], name="Max drawdown, trailing 60d",
                             mode="lines", line=dict(color="#59636e", width=1.5)))
    base_layout(fig, "Drawdowns")
    fig.update_yaxes(tickformat=".0%")
    return fig


def distribution_figure(profile, colors) -> go.Figure:
    p = profile.sort_values("vol_20")
    fig = go.Figure(go.Bar(
        x=p["days"], y=p["regime"], orientation="h",
        marker=dict(color=[colors[r] for r in p["regime"]]),
        text=[f"{d:,} days ({s:.0%})" for d, s in zip(p["days"], p["share"])],
        textposition="outside", textfont=dict(color=INK), cliponaxis=False,
        hovertemplate="%{y}: %{x:,} days<extra></extra>",
    ))
    base_layout(fig, "Regime distribution (trading days)", height=300)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(range=[0, p["days"].max() * 1.25])  # room for outside labels
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# Small HTML building blocks
# ---------------------------------------------------------------------------
CARD = {"background": "#ffffff", "border": f"1px solid {GRID}", "borderRadius": "10px",
        "padding": "16px"}


def stat(label, value):
    return html.Div([
        html.Div(label, style={"fontSize": "12px", "color": INK_MUTED}),
        html.Div(value, style={"fontSize": "20px", "fontWeight": 600, "color": INK}),
    ], style={**CARD, "flex": "1 1 150px"})


def profile_table(profile, colors):
    header = ["Regime", "Avg daily return", "Vol 20d", "Vol 60d", "Momentum 20d",
              "Max DD 60d", "Days", "Avg run (days)"]
    rows = []
    for _, r in profile.sort_values("vol_20").iterrows():
        swatch = html.Span(style={"display": "inline-block", "width": "10px", "height": "10px",
                                  "borderRadius": "2px", "background": colors[r["regime"]],
                                  "marginRight": "8px"})
        cells = [html.Span([swatch, r["regime"]]), f"{r['ret_1d']:.3%}", f"{r['vol_20']:.1%}",
                 f"{r['vol_60']:.1%}", f"{r['mom_20']:.1%}", f"{r['mdd_60']:.1%}",
                 f"{int(r['days']):,}", f"{r['avg_run_days']:.1f}"]
        rows.append(html.Tr([html.Td(c, style={"padding": "6px 10px",
                                               "borderTop": f"1px solid {GRID}"}) for c in cells]))
    return html.Div(html.Table(
        [html.Thead(html.Tr([html.Th(h, style={"padding": "6px 10px", "textAlign": "left",
                                               "color": INK_MUTED, "fontWeight": 500,
                                               "fontSize": "12px"}) for h in header]))]
        + [html.Tbody(rows)],
        style={"borderCollapse": "collapse", "width": "100%", "fontSize": "13px", "color": INK},
    ), style={"overflowX": "auto"})


def latest_card(df, metrics, colors):
    latest = metrics["latest"]
    regime = latest["regime"]
    # How many consecutive trading days the current regime has lasted.
    streak = int((df["regime"][::-1] != regime).cumsum().eq(0).sum())
    return html.Div([
        html.Div("Latest detected regime", style={"fontSize": "12px", "color": INK_MUTED}),
        html.Div(regime, style={"fontSize": "24px", "fontWeight": 700, "color": INK,
                                "margin": "4px 0"}),
        html.Div(
            f"as of {latest['date']}  |  {streak} consecutive trading days  |  "
            f"20d vol {latest['vol_20']:.1%}  |  20d momentum {latest['mom_20']:.1%}  |  "
            f"60d max drawdown {latest['mdd_60']:.1%}",
            style={"fontSize": "13px", "color": INK_MUTED}),
        html.Div("A description of recent conditions, not a forecast.",
                 style={"fontSize": "12px", "color": INK_MUTED, "marginTop": "6px",
                        "fontStyle": "italic"}),
    ], style={**CARD, "borderLeft": f"8px solid {colors[regime]}"})


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__, title="S&P 500 Regimes")

try:
    DF, PROFILE, METRICS = load_outputs()
    COLORS = regime_colors(PROFILE)
    test_rows = DF[DF["split"] == "test"]
    SPLIT_DATE = test_rows.index[0] if len(test_rows) else None
    fmt = lambda s: "n/a" if s is None else f"{s:.3f}"

    app.layout = html.Div([
        html.H1("S&P 500 Market Regime Detection",
                style={"fontSize": "26px", "margin": "0", "color": INK}),
        html.P(f"K-Means (k={METRICS['k']}) on rolling volatility, momentum and drawdown  |  "
               f"data {METRICS['data_start']} to {METRICS['data_end']}",
               style={"color": INK_MUTED, "marginTop": "4px"}),

        latest_card(DF, METRICS, COLORS),

        html.Div([
            stat("Silhouette (train)", fmt(METRICS["silhouette_train"])),
            stat("Silhouette (out-of-sample)", fmt(METRICS["silhouette_test"])),
            stat("Regime switches", f"{METRICS['regime_switches']:,}"),
            stat("Trading days classified", f"{len(DF):,}"),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "margin": "12px 0"}),

        html.Div([
            dcc.RadioItems(id="scale", options=[{"label": " Linear", "value": "linear"},
                                                {"label": " Log", "value": "log"}],
                           value="linear", inline=True,
                           inputStyle={"marginLeft": "12px"}),
            dcc.Graph(id="price-chart", config={"displaylogo": False}),
        ], style=CARD),

        html.Div([
            html.Div(dcc.Graph(figure=volatility_figure(DF), config={"displaylogo": False}),
                     style=CARD),
            html.Div(dcc.Graph(figure=drawdown_figure(DF), config={"displaylogo": False}),
                     style=CARD),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(340px, 1fr))",
                  "gap": "12px", "margin": "12px 0"}),

        html.Div([
            html.H3("Regime characteristics (full period averages)",
                    style={"fontSize": "15px", "margin": "0 0 8px", "color": INK}),
            profile_table(PROFILE, COLORS),
        ], style=CARD),

        html.Div(dcc.Graph(figure=distribution_figure(PROFILE, COLORS),
                           config={"displaylogo": False}),
                 style={**CARD, "marginTop": "12px"}),

        html.P("Labels were assigned from each cluster's observed averages in the training period. "
               "Regimes describe past behaviour; they do not predict returns or imply a trading edge.",
               style={"fontSize": "12px", "color": INK_MUTED, "marginTop": "16px"}),
    ], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px 16px",
              "fontFamily": "Segoe UI, Roboto, Helvetica, Arial, sans-serif",
              "background": "#f6f7f9"})

    @app.callback(Output("price-chart", "figure"), Input("scale", "value"))
    def update_price(scale):
        return price_figure(DF, COLORS, SPLIT_DATE, log_scale=(scale == "log"))

except FileNotFoundError as err:
    # Friendly page instead of a crash if the analysis hasn't been run yet.
    app.layout = html.Div([
        html.H2("No results yet"),
        html.P(str(err)),
        html.Pre("python run_analysis.py\npython app.py"),
    ], style={"fontFamily": "sans-serif", "padding": "24px"})


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
