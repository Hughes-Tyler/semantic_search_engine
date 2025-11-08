import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------
# Articles with text and URLs
articles_df = pd.read_csv("Results/survey_monkey_help_articles_with_text.csv")

# Monkey queries with expected article URLs
queries_df = pd.read_excel("Data/Survey Monkey Queries - Hughes.xlsx")

# Fix column name if it’s misspelled (e.g., "Expected Articke")
if "Expected Articke" in queries_df.columns:
    queries_df.rename(columns={"Expected Articke": "Expected articles"}, inplace=True)

# ---------------------------------------------------------------------
# Step 2: Prepare TF-IDF vectorizer on the article text
# ---------------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(articles_df["Text"].fillna(""))

# ---------------------------------------------------------------------
# Step 3: Define a search function
# ---------------------------------------------------------------------
def search_articles(query, top_n=10):
    """Return top N most relevant articles (Title + URL) for a given query."""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    return articles_df.iloc[top_indices][["Title", "URL"]]

# ---------------------------------------------------------------------
# Step 4: Loop through queries and evaluate accuracy by URL match
# ---------------------------------------------------------------------
results = []

for _, row in queries_df.iterrows():
    query = row["Query"]
    expected_url = str(row["Expected articles"]).strip().lower()
    top_articles = search_articles(query)

    # Normalize URLs for comparison
    top_urls = [str(u).strip().lower() for u in top_articles["URL"]]
    match_found = expected_url in top_urls

    results.append({
        "Query": query,
        "Expected Article URL": row["Expected articles"],
        "Match Found": match_found,
        "Rank (if found)": top_urls.index(expected_url) + 1 if match_found else None,
        "Top URLs": "; ".join(top_articles["URL"].tolist())
    })

# ---------------------------------------------------------------------
# Step 5: Export evaluation results
# ---------------------------------------------------------------------
output_df = pd.DataFrame(results)
output_df.to_csv("Results/tfidf_url_match_evaluation.csv", index=False)

print("✅ Results saved to Results/tfidf_url_match_evaluation.csv")
print(output_df.head(10))
