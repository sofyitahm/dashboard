import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
  page_title = "titanic!", 
  page_icon = "?", # emoji ikon untuk ikon tab, bisa dengan "random"
  layout = "wide", #centered atau #wide
  menu_items = {
    'Get Help' : None, #atau string
    'Report a bug' : "https://www.google.com",
    'About' : "Ini adalah about"
  }, 
  initial_sidebar_state = "auto", 
)

