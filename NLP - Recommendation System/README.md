# 📌 Text-Based Recommendation with Constraints

This project provides a simple **text preprocessing** and **recommendation system** based on TF‑IDF and cosine similarity.  
It can filter a dataset by attributes such as education, community, religion, and income before ranking results that match a user query.

---

## ✨ Features
✅ Preprocesses text (lowercase, remove numbers, punctuation, stopwords)  
✅ Combines multiple columns into one searchable field  
✅ Filters data by optional constraints (e.g., education, community, religion, income)  
✅ Ranks results using **cosine similarity** between user query and dataset entries  
✅ Returns the top `N` most relevant entries

---

## 📂 Dataset
The script expects a CSV file named **`dataset_combined.csv`** with at least these columns:
- `Name`
- `Education Qualification`
- `Community`
- `Religion`
- `Income`

---

## ⚙️ Installation & Setup
1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
