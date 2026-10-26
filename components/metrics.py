import streamlit as st

def metric_card(slot,label,value,context=""):
    slot.metric(label,value,help=context)
