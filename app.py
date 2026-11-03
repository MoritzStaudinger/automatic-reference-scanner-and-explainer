"""
ARSE — Automatic Reference Scanner and Explainer.

Entry point and router: `streamlit run app.py`.  Pages:
  ui/check.py      Check a PDF (the tool)
  pages/API.py     HTTP API reference
  pages/About.py   Method, sources, limitations
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="ARSE · Reference checker",
    page_icon=":material/fact_check:",
    layout="wide",
    menu_items={"About": "ARSE checks the references in academic PDFs against "
                         "scholarly indexes. Built at TU Wien."},
)

page = st.navigation(
    [
        st.Page("ui/check.py", title="Check a PDF", icon=":material/fact_check:",
                default=True),
        st.Page("pages/API.py", title="API", icon=":material/api:", url_path="api"),
        st.Page("pages/About.py", title="About", icon=":material/info:", url_path="about"),
    ],
    position="top",
)
page.run()
