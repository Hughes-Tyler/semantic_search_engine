import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the CSV
df = pd.read_csv("Results/survey_monkey_help_articles_with_text.csv")

# Step 2: Prepare the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')  # remove common words
tfidf_matrix = vectorizer.fit_transform(df['Text'])  # convert articles to vectors

# Step 3: Function to search for query
def search_articles(query, top_n=10):
    query_vec = vectorizer.transform([query])  # vectorize the query
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()  # similarity scores
    top_indices = similarities.argsort()[::-1][:top_n]  # top N indices
    results = df.iloc[top_indices][['Title', 'URL']]  # get titles and URLs
    return results

# Example usage
user_query = input("Enter your search query: ")
top_articles = search_articles(user_query)
print("\nTop 10 relevant articles:")
print(top_articles.to_string(index=False))
