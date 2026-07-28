import streamlit as st

main_page = st.Page("./pages/dashboard.py", title="Dashboard", icon=":material/planet:")
about_page = st.Page("./pages/about.py", title="About", icon=":material/info:")

pg = st.navigation([main_page, about_page])
pg.run()
