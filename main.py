from dotenv import load_dotenv
import requests
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
import streamlit as st
import os
from datetime import date, timedelta
import csv
import io
import re
import nltk

# ── Setup ──────────────────────────────────────────────────────

# Load environment variables from .env file
load_dotenv()

# Get the API key from the .env file
api_key = os.getenv("NEWS_API_KEY")

# Load stopwords once for better performance

# safe initialization
try:
    english_stopwords = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    english_stopwords = set(stopwords.words("english"))

# Load VADER
nltk.download("vader_lexicon", quiet=True)

# Helper function to clear old results
def clear_results():
    keys = [
        "headlines",
        "sentiment_list",
        "source_list",
        "all_words",
        "topic",
        "total_results",
        "total_fetched"
    ]

    for key in keys:
        st.session_state.pop(key, None)

# ── User Interface ─────────────────────────────────────────────

st.title("News Sentiment Analyzer")

# Text input for the topic
topic = st.text_input("Enter a topic:")

# Date range selector (default is last 7 days)
col1, col2 = st.columns(2)
with col1:
    from_date = st.date_input("From:", value=date.today() - timedelta(days=7))
with col2:
    to_date = st.date_input("To:", value=date.today())

# Dropdown to choose whether to search in title, content, or both
search_in = st.selectbox("Search in:", ["Title", "Title and Content", "Content"])

# Map the user's choice to the NewsAPI parameter
search_map = {
    "Title": "title",
    "Title and Content": "title,content",
    "Content": "content"
}

# ── Main Logic ─────────────────────────────────────────────────

if st.button("Analyze"):

    # Show an error if the user didn't enter a topic
    if not topic:
        clear_results()
        st.error("Please enter a topic to analyze.")
    else:
        # Build the NewsAPI URL with the user's inputs
        url = (f"https://newsapi.org/v2/everything?q={topic}&language=en"
               f"&from={from_date}&to={to_date}&searchIn={search_map[search_in]}"
               f"&sortBy=publishedAt&apiKey={api_key}")

        # Fetch the news articles from NewsAPI
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            clear_results()
            st.error(f"Network error: {e}")
            st.stop()

        # Handle API errors
        if data.get("status") != "ok":
            clear_results()
            st.error(f"Error fetching news: {data.get('message', 'Unknown error')}")

        # Handle case where no articles are found
        elif data["totalResults"] == 0:
            clear_results()
            st.warning("No articles found for this topic and date range.")

        else:
            # Initialize the sentiment analyzer
            sia = SentimentIntensityAnalyzer()

            # Store results in session state so they persist across reruns
            st.session_state.headlines = []
            st.session_state.sentiment_list = []
            st.session_state.source_list = []
            st.session_state.all_words = []
            st.session_state.topic = topic
            st.session_state.total_results = data["totalResults"]
            st.session_state.total_fetched = len(data["articles"])

            # ── Process each article ───────────────────────────

            for article in data["articles"]:
                title = article["title"]

                # Skip removed or empty articles
                if not title or title == "[Removed]":
                    continue

                article_date = article["publishedAt"][:10]  # Extract date (YYYY-MM-DD)
                url_link = article["url"]
                source = article["source"]["name"]

                # Get sentiment scores for the headline
                score = sia.polarity_scores(title)

                # Classify sentiment based on the compound score
                # Compound > 0.05 = Positive, < -0.05 = Negative, else Neutral
                if score["compound"] >= 0.05:
                    st.session_state.sentiment_list.append("Positive")
                    st.session_state.headlines.append(("🟢 Positive", title, article_date, source, url_link))
                elif score["compound"] <= -0.05:
                    st.session_state.sentiment_list.append("Negative")
                    st.session_state.headlines.append(("🔴 Negative", title, article_date, source, url_link))
                else:
                    st.session_state.sentiment_list.append("Neutral")
                    st.session_state.headlines.append(("⚪ Neutral", title, article_date, source, url_link))

                # Store the source name for the source breakdown chart
                st.session_state.source_list.append(source)

                # Extract words from the title, remove stop words and short words
                words = re.findall(r"\b[a-zA-Z]+\b", title.lower())
                for word in words:
                    if word not in english_stopwords and len(word) > 2:
                        st.session_state.all_words.append(word)

# ── Display Results (outside the button block so they persist when the user filters the headlines) ──

if "headlines" in st.session_state:

    # Shortcut for st.session_state for better readability
    ss = st.session_state

    # Show how many articles were fetched vs total available
    st.write(f"Found **{ss.total_fetched}** articles out of **{ss.total_results}** total results.")

    # Explain the limitation of NewsAPI
    st.caption(
        "NewsAPI returns a limited number of articles per request. "
        "Analysis is based on the articles retrieved."
    )

    # Prevent crashes if all articles were skipped
    if not ss.sentiment_list:
        st.warning("No valid headlines were found.")
        st.stop()

    # Count how many headlines fall into each sentiment category
    w = Counter(ss.sentiment_list)

    # Determine the dominant sentiment and its percentage
    total = len(ss.sentiment_list)
    top_sentiment, top_count = w.most_common(1)[0]
    top_percentage = round((top_count / total) * 100)

    # Display a colored summary banner based on the dominant sentiment
    if top_sentiment == "Positive":
        st.success(
            f"📈 The news around **{ss.topic}** is mostly **Positive** — {top_percentage}% of headlines are positive.")
    elif top_sentiment == "Negative":
        st.error(
            f"📉 The news around **{ss.topic}** is mostly **Negative** — {top_percentage}% of headlines are negative.")
    else:
        st.info(
            f"➡️ The news around **{ss.topic}** is mostly **Neutral** — {top_percentage}% of headlines are neutral.")

    # ── Overall Sentiment Chart ────────────────────────

    colors = ["green", "gray", "red"]
    labels = ["Positive", "Neutral", "Negative"]
    values = [w.get("Positive", 0), w.get("Neutral", 0), w.get("Negative", 0)]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=colors)
    ax.set_title(f"News Sentiment Analysis: {ss.topic}")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Headlines")
    fig.autofmt_xdate()
    st.pyplot(fig)

    # ── Source Breakdown Chart ─────────────────────────

    # Get the top 10 most frequent news sources
    source_counts = Counter(ss.source_list).most_common(10)
    source_names = [s[0] for s in source_counts]
    source_values = [s[1] for s in source_counts]

    # Plot them
    fig3, ax3 = plt.subplots()
    ax3.barh(source_names, source_values, color="steelblue")
    ax3.set_title(f"Top News Sources: {ss.topic}")
    ax3.set_xlabel("Number of Articles")
    fig3.autofmt_xdate()
    st.pyplot(fig3)

    # ── Most Common Words Chart ────────────────────────

    # Get the top 15 most frequent words across all headlines
    word_counts = Counter(ss.all_words).most_common(15)
    word_names = [word[0] for word in word_counts]
    word_values = [word[1] for word in word_counts]

    # Plot them
    fig4, ax4 = plt.subplots()
    ax4.barh(word_names, word_values, color="mediumpurple")
    ax4.set_title(f"Most Common Words: {ss.topic}")
    ax4.set_xlabel("Frequency")
    fig4.autofmt_xdate()
    st.pyplot(fig4)

    # ── Headlines Filter ──────────────────────────────

    # An option to filter the headlines based on sentiment
    filter_sentiment = st.selectbox(
        "Filter headlines by:",
        ["All", "Positive", "Neutral", "Negative"]
    )

    filtered = [
        (sentiment, title, article_date, source, url_link)
        for sentiment, title, article_date, source, url_link in ss.headlines
        if filter_sentiment == "All" or filter_sentiment in sentiment
    ]

    # ── Export Filtered Results ───────────────────────

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Sentiment",
        "Title",
        "Date",
        "Source",
        "URL"
    ])

    for sentiment, title, article_date, source, url_link in filtered:
        writer.writerow([
            sentiment,
            title,
            article_date,
            source,
            url_link
        ])

    st.download_button(
        label=f"📥 Download {filter_sentiment} Headlines as CSV",
        data=output.getvalue(),
        file_name=f"{ss.topic}_{filter_sentiment.lower()}_headlines.csv",
        mime="text/csv"
    )

    # ── Display Headlines ─────────────────────────────

    st.subheader(f"Headlines ({len(filtered)})")

    if not filtered:
        st.warning("No headlines found for this filter.")
    else:
        for sentiment, title, article_date, source, url_link in filtered:
            st.write(
                f"{sentiment} — [{title}]({url_link}) "
                f"({source}, {article_date})"
            )