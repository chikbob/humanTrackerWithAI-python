def configure_page(st):
    st.set_page_config(page_title="Мониторинг и анализ объектов", layout="wide")
    st.markdown(
        """
        <style>
            .block-container {padding-top: 0.8rem; padding-bottom: 1rem; max-width: 1400px;}
            h1 {margin-bottom: 0.4rem;}
            .stSelectbox div[data-baseweb="select"] input {pointer-events: none;}
            .stSelectbox div[data-baseweb="select"] {cursor: pointer;}
            .stButton>button {width: 100%; border-radius: 10px; font-size: 16px;}
            .stRadio>div {justify-content: center;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("📡 Система мониторинга и интеллектуального анализа объектов")
    st.caption("Детекция и трекинг в реальном времени, журнал событий, уведомления, динамика и экспорт отчётов.")
