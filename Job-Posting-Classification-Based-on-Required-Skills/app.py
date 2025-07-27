import streamlit as st
import pandas as pd
from clustering_utils import perform_clustering, save_model
from scraper import scrape_jobs

st.title("Job Clustering Based on Skills")

if st.button("Scrape and Cluster Jobs"):
    df = scrape_jobs()
    labels, model, vectorizer = perform_clustering(df["Skills"])
    df["Cluster"] = labels
    save_model(model, vectorizer)
    st.dataframe(df)
    df.to_csv("clustered_jobs.csv", index=False)
    st.download_button("Download Clustered Jobs CSV", data=df.to_csv(index=False), file_name="clustered_jobs.csv", mime="text/csv")