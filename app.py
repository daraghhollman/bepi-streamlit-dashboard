import streamlit as st

main_page = st.Page("./pages/predictions.py", title="Predictions", icon=":material/planet:")
about_page = st.Page("./pages/about.py", title="About", icon=":material/info:")

pg = st.navigation([main_page, about_page])
pg.run()
