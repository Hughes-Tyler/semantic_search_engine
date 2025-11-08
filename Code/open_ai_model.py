import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# ---------------------------------------------------------------------
# Load environment and initialize
# ---------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")
client = OpenAI(api_key=api_key)

# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------
ARTICLES_FILE = "Results/survey_monkey_help_articles_with_text.csv"
QUERIES_FILE = "Data/Survey Monkey Queries - Hughes.xlsx"
EMBEDDINGS_FILE = "Results/article_embeddings.npy"

# ---------------------------------------------------------------------
# Load existing data
# ---------------------------------------------------------------------
df = pd.read_csv(ARTICLES_FILE)
article_embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True)

encoding = tiktoken.encoding_for_model("text-embedding-3-small")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

# ---------------------------------------------------------------------
# Semantic search function
# ---------------------------------------------------------------------
def semantic_search(query: str, top_n: int = 10):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_emb = np.array(resp.data[0].embedding)
    norms_query = np.linalg.norm(query_emb)
    norms_articles = np.linalg.norm(article_embeddings, axis=1)
    similarities = (article_embeddings @ query_emb) / (norms_articles * norms_query)
    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = df.iloc[top_indices][['Title', 'URL']].copy()
    results['Score'] = similarities[top_indices]
    return results

# ---------------------------------------------------------------------
# Evaluate all queries from file
# ---------------------------------------------------------------------
queries_df = pd.read_excel(QUERIES_FILE)
results = []

for _, row in queries_df.iterrows():
    query = row['Query']
    expected_url = str(row['Expected articles']).strip().lower()  # Expecting this to be a URL
    top_articles = semantic_search(query)
    
    top_urls = [u.strip().lower() for u in top_articles['URL']]
    match_found = expected_url in top_urls
    rank = top_urls.index(expected_url) + 1 if match_found else None
    
    results.append({
        "Query": query,
        "Expected Article (URL)": row['Expected articles'],
        "Match Found": match_found,
        "Rank (if found)": rank,
        "Top Results (URLs)": "; ".join(top_articles['URL'].tolist()),
        "Top Result Titles": "; ".join(top_articles['Title'].tolist())
    })
    print(f"✅ Processed query: {query} | Match found: {match_found}")

# ---------------------------------------------------------------------
# Save evaluation
# ---------------------------------------------------------------------
output_df = pd.DataFrame(results)
output_path = "Results/semantic_query_match_evaluation.csv"
output_df.to_csv(output_path, index=False)

# ---------------------------------------------------------------------
# Optional: print summary stats
# ---------------------------------------------------------------------
accuracy = output_df['Match Found'].mean() * 100
print(f"\n🎯 Evaluation complete! Saved to {output_path}")
print(f"Overall accuracy: {accuracy:.2f}% ({output_df['Match Found'].sum()} of {len(output_df)})")

print("\nSample results:")
print(output_df.head(10).to_string(index=False))
