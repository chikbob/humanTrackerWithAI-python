def configure_page(st):
    st.set_page_config(page_title="Мониторинг прохода сотрудников", layout="wide")
    st.markdown(
        """
        <style>
            .block-container {padding-top: 0.8rem; padding-bottom: 1rem; max-width: 1400px;}
            h1 {margin-bottom: 0.4rem;}
            .stSelectbox div[data-baseweb="select"] input {pointer-events: none;}
            .stSelectbox div[data-baseweb="select"] {cursor: pointer;}
            .stButton>button {width: 100%; border-radius: 10px; font-size: 16px;}
            .stRadio>div {justify-content: center;}
            [data-testid="stMetric"] {
                background: #f6f8fb;
                border: 1px solid #d9e1ea;
                padding: 0.8rem;
                border-radius: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🏭 Веб-система мониторинга прохода сотрудников")
    st.caption(
        "Онлайн-мониторинг входной зоны предприятия, журнал проходов, события проходной и аналитика по перемещению сотрудников."
    )
