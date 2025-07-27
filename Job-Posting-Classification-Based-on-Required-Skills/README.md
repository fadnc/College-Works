# Job Clustering Project

This project scrapes job data from karkidi.com and clusters jobs based on required skills using unsupervised machine learning.

## Features
- Scrapes job data (title, company, skills, etc.)
- Clusters jobs using Agglomerative Clustering
- Saves model for reuse
- Daily scraping script
- Notifies users about matching jobs

## How to Run
```
pip install -r requirements.txt
streamlit run app.py
```