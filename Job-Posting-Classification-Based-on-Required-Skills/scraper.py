import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_jobs():
    jobs = []
    for page in range(1, 3):
        url = f"https://www.karkidi.com/job-opportunities/page:{page}"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        job_cards = soup.find_all("div", class_="job-card")
        for card in job_cards:
            title = card.find("h2").text.strip()
            company = card.find("h3").text.strip()
            location = card.find("span", class_="location").text.strip()
            exp = card.find("span", class_="experience").text.strip()
            skills = card.find("div", class_="skills").text.strip()
            jobs.append({
                "Title": title, "Company": company,
                "Location": location, "Experience": exp, "Skills": skills
            })
    return pd.DataFrame(jobs)