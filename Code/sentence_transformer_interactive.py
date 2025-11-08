import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")
client = OpenAI(api_key=api_key)


CSV_FILE = "Results/survey_monkey_help_articles_with_text.csv"
df = pd.read_csv(CSV_FILE)

encoding = tiktoken.encoding_for_model("text-embedding-3-small")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


EMBEDDINGS_FILE = "Results/article_embeddings.npy"
TOKEN_COUNT_FILE = "Results/total_tokens.txt"

if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(TOKEN_COUNT_FILE):
    print("Loading cached embeddings...")
    article_embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    with open(TOKEN_COUNT_FILE, "r") as f:
        total_tokens = int(f.read())
    print(f"Total tokens used previously: {total_tokens}")
else:
    print("Computing embeddings for all articles...")
    article_embeddings = []
    total_tokens = 0
    for i, text in enumerate(df['Text']):
        tokens = count_tokens(text)
        total_tokens += tokens

        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        emb = np.array(resp.data[0].embedding)
        article_embeddings.append(emb)

        if (i+1) % 50 == 0 or (i+1) == len(df):
            print(f"Processed {i+1}/{len(df)} articles. Total tokens so far: {total_tokens}")

    article_embeddings = np.vstack(article_embeddings)
    np.save(EMBEDDINGS_FILE, article_embeddings)
    with open(TOKEN_COUNT_FILE, "w") as f:
        f.write(str(total_tokens))
    print(f"Embeddings cached. Total tokens used for corpus: {total_tokens}")


# -----------------------------
def semantic_search(query: str, top_n: int = 10):
    query_tokens = count_tokens(query)
    print(f"Query tokens: {query_tokens}")

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

user_query = input("Enter your search query: ")
top_articles = semantic_search(user_query)
print("\nTop 10 relevant articles:")
print(top_articles.to_string(index=False))
