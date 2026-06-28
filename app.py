# -*- coding: utf-8 -*-
"""Dashboard Analisis Tempat Wisata - Google Maps Review Analytics"""

import io
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Analisis Tempat Wisata",
    layout="wide",
    page_icon="🌄",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  (matches the dark screenshot)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
:root { color-scheme: dark; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp, .main, .block-container {
    background-color: #0D1117 !important;
    color: #F0F6FC !important;
    padding-top: 1rem !important;
}
.block-container { max-width: 100% !important; padding-left: 2rem; padding-right: 2rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0D1B2E !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
    min-width: 260px !important;
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
.sidebar-title {
    font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
    color: #64748B !important; text-transform: uppercase; margin-bottom: 12px;
}
.filter-label {
    font-size: 13px; font-weight: 600; color: #94A3B8 !important; margin-bottom: 6px;
}
.nav-divider { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 14px 0; }

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div { background: #1C2A3A !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; color: #F0F6FC !important; }
.stTextInput > div > div > input { background: #1C2A3A !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; color: #F0F6FC !important; }
.stRadio > div { gap: 6px; }
.stRadio > div > label { background: #1C2A3A; border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 8px 14px; font-size: 13px; cursor: pointer; color: #CBD5E1 !important; }

/* ── Header banner ── */
.header-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 18px 24px; margin-bottom: 24px;
    background: #161D2B;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
}
.header-left { display: flex; align-items: center; gap: 16px; }
.app-badge {
    width: 48px; height: 48px; border-radius: 16px;
    display: grid; place-items: center; font-size: 1.4rem;
    background: linear-gradient(135deg, #6D28D9, #2563EB);
}
.header-title { font-size: 1.75rem; font-weight: 700; color: #F0F6FC; line-height: 1.2; }
.header-subtitle { font-size: 0.88rem; color: #64748B; margin-top: 2px; }
.header-right { display: flex; align-items: center; gap: 12px; }
.header-date { color: #94A3B8; font-size: 0.88rem; }
.status-chip {
    padding: 8px 14px; border-radius: 999px;
    background: rgba(34,197,94,0.12); color: #86EFAC; font-size: 0.85rem; font-weight: 500;
}

/* ── Info banner ── */
.info-banner {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 20px; margin-bottom: 20px;
    background: rgba(37,99,235,0.1); border: 1px solid rgba(37,99,235,0.25);
    border-radius: 12px; font-size: 0.88rem; color: #93C5FD;
}

/* ── Metric cards (top row) ── */
.metric-card {
    background: #161D2B;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 20px;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.metric-icon-wrap {
    width: 40px; height: 40px; border-radius: 12px;
    display: grid; place-items: center; font-size: 1.1rem;
    margin-bottom: 14px;
}
.metric-label { font-size: 0.82rem; color: #64748B; font-weight: 500; margin-bottom: 4px; }
.metric-value { font-size: 1.85rem; font-weight: 800; color: #F0F6FC; line-height: 1; }
.metric-sub { font-size: 0.82rem; margin-top: 6px; font-weight: 500; }
.metric-sub-gray { color: #64748B; }
.metric-sub-green { color: #4ADE80; }
.metric-sub-red { color: #F87171; }

/* ── Content cards ── */
.content-card {
    background: #161D2B;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.card-title { font-size: 1rem; font-weight: 700; color: #F0F6FC; margin-bottom: 4px; }
.card-sub { font-size: 0.82rem; color: #64748B; }
.section-title { font-size: 1.1rem; font-weight: 700; color: #F0F6FC; margin: 18px 0 10px 0; }

/* ── Sidebar nav pills ── */
.nav-pill {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 12px;
    font-size: 13px; font-weight: 500; cursor: pointer;
    transition: all .2s; color: #94A3B8 !important;
}
.nav-pill.active { background: #2563EB; color: #fff !important; }
.nav-pill:hover:not(.active) { background: rgba(37,99,235,0.15); color: #E0E7FF !important; }

/* ── Footer ── */
.footer-card {
    text-align: center; font-size: 0.82rem; color: #475569;
    padding: 16px; margin-top: 32px;
    border-top: 1px solid rgba(255,255,255,0.06);
}

/* ── Plotly overrides (hide toolbar) ── */
.js-plotly-plot .plotly .modebar { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    base = Path(__file__).resolve().parent
    for fname in ["hasil_akhir_wisata.csv", "wisata_dataset.csv"]:
        p = base / fname
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("Dataset tidak ditemukan. Pastikan hasil_akhir_wisata.csv ada di folder yang sama.")


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def label_sentiment(rating):
    try:
        v = float(rating)
    except Exception:
        return "Netral"
    if v >= 4:
        return "Positif"
    if v <= 2:
        return "Negatif"
    return "Netral"


# ─────────────────────────────────────────────
# GRAPH HELPERS
# ─────────────────────────────────────────────
def build_similarity_graph(df: pd.DataFrame, threshold: float = 0.2) -> nx.Graph:
    place_text = df.groupby("place")["clean_comment"].apply(lambda x: " ".join(x.dropna().astype(str)))
    if place_text.empty or len(place_text) < 2:
        return nx.Graph()
    vectorizer = TfidfVectorizer(max_features=300)
    X = vectorizer.fit_transform(place_text)
    sim = cosine_similarity(X)
    places = place_text.index.tolist()
    G = nx.Graph()
    G.add_nodes_from(places)
    for i in range(len(places)):
        for j in range(i + 1, len(places)):
            if sim[i, j] > threshold:
                G.add_edge(places[i], places[j], weight=float(sim[i, j]))
    return G


def build_network_plotly(G: nx.Graph) -> go.Figure:
    fig = go.Figure()
    if len(G) == 0:
        fig.add_annotation(text="Tidak ada data cukup untuk graph.", xref="paper", yref="paper",
                           showarrow=False, font=dict(size=15, color="#94A3B8"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    pos = nx.spring_layout(G, seed=42, k=1.2)
    ex, ey = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]; ey += [y0, y1, None]

    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                             line=dict(width=1, color="rgba(99,102,241,0.3)"), hoverinfo="none"))

    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    clo = nx.closeness_centrality(G)
    nx_x, nx_y, nt, nc, ns = [], [], [], [], []
    palette = px.colors.qualitative.T10
    for i, node in enumerate(G.nodes()):
        x, y = pos[node]
        nx_x.append(x); nx_y.append(y)
        nc.append(palette[i % len(palette)])
        ns.append(20 + deg[node] * 100)
        nt.append(f"<b>{node}</b><br>Degree:{deg[node]:.3f}<br>Betweenness:{btw[node]:.3f}<br>Closeness:{clo[node]:.3f}")

    fig.add_trace(go.Scatter(x=nx_x, y=nx_y, mode="markers+text",
                             text=list(G.nodes()), textposition="top center",
                             textfont=dict(size=9, color="#CBD5E1"),
                             hovertext=nt, hoverinfo="text",
                             marker=dict(size=ns, color=nc, line=dict(width=1, color="#0D1117"))))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#161D2B", plot_bgcolor="#161D2B",
                      margin=dict(l=10, r=10, t=10, b=10), hovermode="closest",
                      showlegend=False)
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        # Close button placeholder + title
        col_x, col_t = st.columns([1, 4])
        with col_x:
            if st.button("✕", key="close_btn", help="Tutup sidebar"):
                st.session_state["sidebar_closed"] = True
        with col_t:
            st.markdown("<div class='sidebar-title' style='margin-top:8px;'>FILTER DATA</div>", unsafe_allow_html=True)

        # ── Tempat Wisata ──
        st.markdown("<div class='filter-label'>Tempat Wisata</div>", unsafe_allow_html=True)
        places = ["Semua Tempat"] + sorted(df["place"].dropna().unique().tolist())
        selected_place = st.selectbox("", places, label_visibility="collapsed", key="place_sel")

        # ── Rating ──
        st.markdown("<div class='filter-label' style='margin-top:12px;'>Rating</div>", unsafe_allow_html=True)
        rating_options = ["Semua Rating", "5 ⭐", "4 ⭐", "3 ⭐", "2 ⭐", "1 ⭐"]
        selected_rating = st.selectbox("", rating_options, label_visibility="collapsed", key="rating_sel")

        # ── Sentimen ──
        st.markdown("<div class='filter-label' style='margin-top:12px;'>Sentimen</div>", unsafe_allow_html=True)
        selected_sentiment = st.selectbox("", ["Semua Sentimen", "Positif", "Netral", "Negatif"],
                                          label_visibility="collapsed", key="sent_sel")

        # ── Search ──
        st.markdown("<div class='filter-label' style='margin-top:12px;'>Cari Kata</div>", unsafe_allow_html=True)
        search_text = st.text_input("", placeholder="Cari kata di komentar...",
                                    label_visibility="collapsed", key="search_inp")

        st.markdown("<hr class='nav-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-title'>NAVIGASI</div>", unsafe_allow_html=True)

        nav_items = [
            ("🏠", "Ringkasan"),
            ("📊", "Dataset"),
            ("ℹ️", "Tentang"),
            ("😊", "Visualisasi Sentimen"),
            ("☁️", "WordCloud"),
            ("🔗", "Social Network Analysis"),
        ]
        if "selected_page" not in st.session_state:
            st.session_state["selected_page"] = "Ringkasan"

        for icon, label in nav_items:
            is_active = st.session_state["selected_page"] == label
            style = "background:#2563EB;color:#fff!important;" if is_active else ""
            if st.button(f"{icon}  {label}", key=f"nav_{label}",
                         use_container_width=True):
                st.session_state["selected_page"] = label

        st.markdown("""
        <div style='position:absolute;bottom:16px;left:0;right:0;text-align:center;'>
            <div style='font-size:11px;color:#334155;'>© 2026 Dashboard Analisis Wisata</div>
            <div style='font-size:10px;color:#334155;'>Google Maps Review Analytics</div>
        </div>""", unsafe_allow_html=True)

    return selected_place, selected_rating, selected_sentiment, search_text


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
def render_header():
    today = datetime.now().strftime("%d %B %Y")
    st.markdown(f"""
    <div class='info-banner'>
        ℹ️ Dashboard ini menganalisis ulasan untuk berbagai tempat wisata dan menampilkan insight sentimen, kata kunci, dan jaringan co-occurrence.
    </div>
    <div class='header-row'>
        <div class='header-left'>
            <div class='app-badge'>📊</div>
            <div>
                <div class='header-title'>Dashboard Analisis Tempat Wisata</div>
                <div class='header-subtitle'>Google Maps Review Analytics</div>
            </div>
        </div>
        <div class='header-right'>
            <div class='header-date'>{today}</div>
            <div class='status-chip'>🟢 Online</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────
def render_metrics(filtered: pd.DataFrame):
    total = len(filtered)
    places = filtered["place"].nunique()
    avg_r = filtered["rating"].astype(float).mean() if total else 0
    pos = int((filtered["sentiment"] == "Positif").sum())
    neg = int((filtered["sentiment"] == "Negatif").sum())
    pos_pct = pos * 100 / total if total else 0
    neg_pct = neg * 100 / total if total else 0

    cards = [
        ("💬", "Total Review", f"{total:,}", f"Jumlah ulasan yang dianalisis", "#8B5CF6", "metric-sub-gray"),
        ("📊", "Jumlah Tempat", f"{places}", "Destinasi wisata unik", "#2563EB", "metric-sub-gray"),
        ("⭐", "Rating Rata-rata", f"{avg_r:.2f}", "Skala 1-5", "#F59E0B", "metric-sub-gray"),
        ("😊", "Review Positif", f"{pos_pct:.0f}%", f"{pos_pct:.1f}% Positif", "#F97316", "metric-sub-green"),
        ("😞", "Review Negatif", f"{neg_pct:.0f}%", f"{neg_pct:.1f}% Negatif", "#EF4444", "metric-sub-red"),
    ]

    cols = st.columns(5, gap="small")
    for col, (icon, label, value, sub, color, sub_cls) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-icon-wrap' style='background:{color}22;'>
                    <span style='font-size:1.2rem;'>{icon}</span>
                </div>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{value}</div>
                <div class='metric-sub {sub_cls}'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    return total, places, avg_r, pos, neg, pos_pct, neg_pct


# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────
def page_ringkasan(filtered: pd.DataFrame, total, pos_pct, neg_pct):
    st.markdown("<div class='section-title'>Distribusi & Analisis</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown("<div class='content-card'><div class='card-title'>Distribusi Rating</div></div>", unsafe_allow_html=True)
        rating_counts = filtered["rating"].astype(int).value_counts().sort_index()
        pcts = (rating_counts / rating_counts.sum() * 100).round(1)
        labels = [f"{pcts.get(i,0)}%" for i in rating_counts.index]
        fig = go.Figure(go.Bar(
            x=rating_counts.index.astype(str),
            y=rating_counts.values,
            text=[f"{v:,} ({p})" for v, p in zip(rating_counts.values, labels)],
            textposition="outside",
            marker=dict(color="#7C3AED", opacity=0.85,
                        line=dict(color="#A78BFA", width=0)),
            hovertemplate="Rating %{x}<br>%{y} ulasan<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="#161D2B", paper_bgcolor="#161D2B",
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="Rating", tickfont=dict(size=13)),
            yaxis=dict(title="Jumlah Review", gridcolor="rgba(255,255,255,0.05)"),
            bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown("<div class='content-card'><div class='card-title'>Distribusi Sentimen</div></div>", unsafe_allow_html=True)
        sent_counts = filtered["sentiment"].value_counts()
        color_map = {"Positif": "#22C55E", "Netral": "#60A5FA", "Negatif": "#EF4444"}
        colors = [color_map.get(s, "#94A3B8") for s in sent_counts.index]
        dominant = sent_counts.idxmax() if not sent_counts.empty else "-"
        dominant_pct = int(sent_counts.max() / total * 100) if total else 0
        fig2 = go.Figure(go.Pie(
            labels=sent_counts.index,
            values=sent_counts.values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color="#0D1117", width=2)),
            textinfo="label+percent",
            textfont=dict(size=11, color="#F0F6FC"),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        ))
        fig2.add_annotation(text=f"<b>{dominant_pct}%</b><br><span style='font-size:11px'>{dominant}</span>",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(size=18, color="#22C55E" if dominant == "Positif" else "#F0F6FC"))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="#161D2B",
            height=320, margin=dict(l=0, r=0, t=10, b=10),
            legend=dict(orientation="v", x=1.02, y=0.5,
                        font=dict(color="#CBD5E1", size=11)),
            showlegend=True,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Row 2: Rating per place + Top places ──
    col3, col4 = st.columns([2, 1], gap="medium")
    with col3:
        st.markdown("<div class='content-card'><div class='card-title'>Rating Rata-rata per Tempat</div></div>", unsafe_allow_html=True)
        avg_r = filtered.groupby("place")["rating"].mean().sort_values(ascending=False)
        fig3 = px.bar(x=avg_r.index, y=avg_r.values,
                      labels={"x": "", "y": "Rating"},
                      color=avg_r.values,
                      color_continuous_scale=["#3B82F6", "#22C55E", "#F59E0B"])
        fig3.update_layout(template="plotly_dark", plot_bgcolor="#161D2B", paper_bgcolor="#161D2B",
                           height=280, margin=dict(l=10, r=10, t=10, b=50),
                           xaxis_tickangle=-40, coloraxis_showscale=False)
        fig3.update_traces(hovertemplate="%{x}<br>Rating: %{y:.2f}<extra></extra>")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with col4:
        st.markdown("<div class='content-card'><div class='card-title'>Top 10 Tempat by Review</div></div>", unsafe_allow_html=True)
        top = filtered["place"].value_counts().head(10)
        fig4 = px.bar(x=top.values, y=top.index, orientation="h",
                      color=top.values, color_continuous_scale="purples",
                      labels={"x": "Jumlah", "y": ""})
        fig4.update_layout(template="plotly_dark", plot_bgcolor="#161D2B", paper_bgcolor="#161D2B",
                           height=280, margin=dict(l=5, r=10, t=10, b=10),
                           yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})


def page_sentimen(filtered: pd.DataFrame):
    st.markdown("<div class='section-title'>Analisis Sentimen</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    sent = filtered["sentiment"].value_counts()
    cmap = {"Positif": "#22C55E", "Netral": "#60A5FA", "Negatif": "#EF4444"}

    with c1:
        st.markdown("<div class='content-card'><div class='card-title'>Pie Sentimen</div></div>", unsafe_allow_html=True)
        fig = px.pie(names=sent.index, values=sent.values, hole=0.52,
                     color=sent.index, color_discrete_map=cmap)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#161D2B", height=360,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("<div class='content-card'><div class='card-title'>Bar Sentimen</div></div>", unsafe_allow_html=True)
        sent_all = sent.reindex(["Positif", "Netral", "Negatif"], fill_value=0)
        fig2 = px.bar(x=sent_all.index, y=sent_all.values,
                      color=sent_all.index, color_discrete_map=cmap,
                      labels={"x": "Sentimen", "y": "Jumlah"})
        fig2.update_layout(template="plotly_dark", plot_bgcolor="#161D2B", paper_bgcolor="#161D2B",
                           height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Sentiment timeline per place
    st.markdown("<div class='content-card'><div class='card-title'>Sentimen per Tempat Wisata</div></div>", unsafe_allow_html=True)
    sent_place = filtered.groupby(["place", "sentiment"]).size().unstack(fill_value=0)
    for col in ["Positif", "Netral", "Negatif"]:
        if col not in sent_place.columns:
            sent_place[col] = 0
    fig3 = go.Figure()
    for s, c in cmap.items():
        if s in sent_place.columns:
            fig3.add_trace(go.Bar(name=s, x=sent_place.index, y=sent_place[s],
                                  marker_color=c))
    fig3.update_layout(barmode="stack", template="plotly_dark",
                       plot_bgcolor="#161D2B", paper_bgcolor="#161D2B",
                       height=320, margin=dict(l=10, r=10, t=10, b=50),
                       xaxis_tickangle=-40)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


def page_wordcloud(filtered: pd.DataFrame):
    st.markdown("<div class='section-title'>WordCloud & Kata Paling Umum</div>", unsafe_allow_html=True)
    text = " ".join(filtered["clean_comment"].astype(str).tolist()).strip()
    if not text:
        st.info("Tidak ada teks untuk ditampilkan."); return

    st.markdown("<div class='content-card'><div class='card-title'>WordCloud Komentar</div></div>", unsafe_allow_html=True)
    wc = WordCloud(width=1100, height=380, background_color="#161D2B",
                   colormap="cool", max_words=120).generate(text)
    fig_wc, ax = plt.subplots(figsize=(13, 4), facecolor="#161D2B")
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    st.pyplot(fig_wc); plt.close(fig_wc)

    st.markdown("<div class='content-card'><div class='card-title'>20 Kata Paling Sering Muncul</div></div>", unsafe_allow_html=True)
    counts = Counter(text.split())
    top_words = pd.DataFrame(counts.most_common(20), columns=["Kata", "Frekuensi"])
    fig = px.bar(top_words, x="Frekuensi", y="Kata", orientation="h",
                 color="Frekuensi", color_continuous_scale="purples")
    fig.update_layout(template="plotly_dark", plot_bgcolor="#161D2B", paper_bgcolor="#161D2B",
                      height=480, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def page_sna(filtered: pd.DataFrame):
    st.markdown("<div class='section-title'>Social Network Analysis</div>", unsafe_allow_html=True)
    G = build_similarity_graph(filtered)
    c1, c2 = st.columns([2, 1], gap="medium")
    with c1:
        st.markdown("<div class='content-card'><div class='card-title'>Graf Kemiripan Antar Tempat</div></div>", unsafe_allow_html=True)
        st.plotly_chart(build_network_plotly(G), use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.markdown("<div class='content-card'><div class='card-title'>Top 10 Centrality</div></div>", unsafe_allow_html=True)
        if len(G) == 0:
            st.write("Graph kosong.")
        else:
            deg = nx.degree_centrality(G)
            btw = nx.betweenness_centrality(G, normalized=True)
            clo = nx.closeness_centrality(G)
            rows = [{"Tempat": n, "Degree": deg[n], "Betweenness": btw[n], "Closeness": clo[n]}
                    for n in G.nodes()]
            cdf = pd.DataFrame(rows).sort_values("Degree", ascending=False).head(10).reset_index(drop=True)
            st.dataframe(cdf.style.format({"Degree": "{:.3f}", "Betweenness": "{:.3f}", "Closeness": "{:.3f}"}),
                         use_container_width=True)


def page_dataset(filtered: pd.DataFrame):
    st.markdown("<div class='section-title'>Dataset Review Wisata</div>", unsafe_allow_html=True)
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=420)
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", data=csv, file_name="wisata_reviews.csv", mime="text/csv")
    with c2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            filtered.to_excel(w, index=False, sheet_name="Dataset")
        buf.seek(0)
        st.download_button("⬇ Download Excel", data=buf, file_name="wisata_reviews.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def page_tentang():
    st.markdown("<div class='section-title'>Tentang Dashboard</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='content-card'>
        <div class='card-title'>📊 Dashboard Analisis Tempat Wisata</div>
        <p style='color:#94A3B8;margin-top:10px;line-height:1.7;'>
            Dashboard ini dibuat untuk menganalisis ulasan tempat wisata Indonesia berdasarkan data Google Maps Review,
            menampilkan insight sentimen, kata kunci populer, dan Social Network Analysis antar destinasi.
        </p>
        <hr style='border-color:rgba(255,255,255,0.08);margin:16px 0;'>
        <div class='card-title'>🛠 Teknologi</div>
        <ul style='color:#94A3B8;line-height:2;margin-top:8px;'>
            <li>Streamlit — framework dashboard interaktif</li>
            <li>Plotly — visualisasi data interaktif</li>
            <li>NetworkX — Social Network Analysis</li>
            <li>Scikit-Learn — TF-IDF & cosine similarity</li>
            <li>WordCloud — visualisasi kata kunci</li>
            <li>Pandas — manajemen data</li>
        </ul>
        <hr style='border-color:rgba(255,255,255,0.08);margin:16px 0;'>
        <div class='card-title'>📂 Dataset</div>
        <p style='color:#94A3B8;margin-top:8px;'>
            300 ulasan dari 30 destinasi wisata Indonesia — mencakup Bali, Lombok, Jawa, Sulawesi, dan Papua.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    df = load_data()
    df["comment"] = df["comment"].fillna("").astype(str)
    df["clean_comment"] = df.get("clean_comment", df["comment"].apply(clean_text)).fillna("").astype(str)
    if "sentiment" not in df.columns:
        df["sentiment"] = df["rating"].apply(label_sentiment)

    sel_place, sel_rating, sel_sentiment, search_text = render_sidebar(df)

    # Apply filters
    filtered = df.copy()
    if sel_place != "Semua Tempat":
        filtered = filtered[filtered["place"] == sel_place]
    if sel_rating != "Semua Rating":
        r = int(sel_rating[0])
        filtered = filtered[filtered["rating"].astype(float) == r]
    if sel_sentiment != "Semua Sentimen":
        filtered = filtered[filtered["sentiment"] == sel_sentiment]
    if search_text:
        filtered = filtered[filtered["comment"].str.contains(search_text, case=False, na=False)]

    render_header()
    total, places, avg_r, pos, neg, pos_pct, neg_pct = render_metrics(filtered)

    page = st.session_state.get("selected_page", "Ringkasan")

    if page == "Ringkasan":
        page_ringkasan(filtered, total, pos_pct, neg_pct)
    elif page == "Visualisasi Sentimen":
        page_sentimen(filtered)
    elif page == "WordCloud":
        page_wordcloud(filtered)
    elif page == "Social Network Analysis":
        page_sna(filtered)
    elif page == "Dataset":
        page_dataset(filtered)
    elif page == "Tentang":
        page_tentang()

    st.markdown("""
    <div class='footer-card'>
        Sumber Data: Google Maps Review · Python · Streamlit · Plotly · NetworkX · Scikit-Learn
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
