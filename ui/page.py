def configure_page(st, *, standalone_mode: bool = False):
    st.set_page_config(page_title="Система интеллектуального мониторинга зон", layout="wide")
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(36, 61, 92, 0.28), transparent 30%),
                    radial-gradient(circle at top right, rgba(21, 95, 77, 0.20), transparent 25%),
                    linear-gradient(180deg, #0f1720 0%, #111c27 45%, #121722 100%);
                color: #e6edf4;
            }
            .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem; max-width: 1480px;}
            h1, h2, h3, h4, h5, h6, p, li, span, div {font-family: "IBM Plex Sans", "Segoe UI", sans-serif;}
            h1 {margin-bottom: 0.35rem; letter-spacing: 0.02em;}
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #111924 0%, #0b121b 100%);
                border-right: 1px solid rgba(148, 163, 184, 0.18);
            }
            .stSelectbox div[data-baseweb="select"] input {pointer-events: none;}
            .stSelectbox div[data-baseweb="select"] {cursor: pointer;}
            .stButton>button {
                width: 100%;
                border-radius: 12px;
                font-size: 15px;
                border: 1px solid rgba(103, 130, 160, 0.35);
                background: linear-gradient(180deg, #162434 0%, #11202d 100%);
                color: #edf3f8;
            }
            .stButton>button:hover {
                border-color: rgba(88, 166, 255, 0.55);
                color: #ffffff;
            }
            [data-testid="stMetric"] {
                background: linear-gradient(180deg, rgba(20, 30, 42, 0.96), rgba(15, 24, 34, 0.96));
                border: 1px solid rgba(122, 144, 168, 0.18);
                padding: 0.9rem;
                border-radius: 16px;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
            }
            [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
                color: #e6edf4;
            }
            [data-testid="stDataFrame"], [data-testid="stTable"] {
                border-radius: 16px;
                overflow: hidden;
            }
            .st-emotion-cache-13ln4jf, .st-emotion-cache-1r6slb0 {
                border-radius: 18px;
            }
            .access-shell {
                padding: 1rem 1.2rem;
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(18, 29, 43, 0.90), rgba(10, 16, 25, 0.92));
                border: 1px solid rgba(122, 144, 168, 0.18);
                margin-bottom: 1rem;
            }
            .access-badge {
                display: inline-block;
                padding: 0.28rem 0.62rem;
                border-radius: 999px;
                margin-right: 0.5rem;
                background: rgba(88, 166, 255, 0.14);
                border: 1px solid rgba(88, 166, 255, 0.25);
                color: #c7ddff;
                font-size: 0.86rem;
            }
            .status-online {
                color: #7ee787;
            }
            .status-offline {
                color: #ff7b72;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if standalone_mode:
        return
    st.markdown(
        """
        <div class="access-shell">
            <div class="access-badge">Enterprise Zone Monitoring</div>
            <div class="access-badge">24/7 Video Pipeline</div>
            <div class="access-badge">Incident Analytics</div>
            <h1>Веб-система интеллектуального мониторинга контролируемых зон</h1>
            <p>
                Интерфейс предназначен для мониторинга камер и зон предприятия, автоматического выявления инцидентов,
                контроля состояния источников видео и последующей аналитической обработки тревог и наблюдений.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
