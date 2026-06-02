#  News Sentiment Analyzer

A Streamlit web app that fetches real-time news headlines and analyzes their sentiment using VADER — giving you a quick read on how the media is covering any topic.

---

## Features

- **Sentiment Analysis** — Classifies headlines as Positive, Neutral, or Negative using NLTK's VADER
- **Date Range Filter** — Choose a custom date range to narrow your search
- **Search Scope** — Search within the title, content, or both
- **Visual Charts** — Bar charts for sentiment distribution, top news sources, and most common words
- **Headline Filter** — Filter displayed headlines by sentiment category
- **CSV Export** — Download filtered results as a CSV file

---

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI framework
- [NewsAPI](https://newsapi.org/) — News data source
- [NLTK / VADER](https://www.nltk.org/) — Sentiment analysis
- [Matplotlib](https://matplotlib.org/) — Charts
- [python-dotenv](https://pypi.org/project/python-dotenv/) — Environment variable management

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/news-sentiment-analyzer.git
cd news-sentiment-analyzer
```

### 2. Install dependencies

```bash
pip install streamlit requests nltk matplotlib python-dotenv
```

Then download the required NLTK data:

```python
import nltk
nltk.download("vader_lexicon")
nltk.download("stopwords")
```

### 3. Get a NewsAPI key

Sign up for a free API key at [newsapi.org](https://newsapi.org/).

### 4. Create a `.env` file

```
NEWS_API_KEY=your_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Usage

1. Enter a topic (e.g., `AI`, `climate change`, `Saudi Arabia`)
2. Set a date range and choose where to search (title/content)
3. Click **Analyze**
4. Explore the sentiment summary, charts, and headlines
5. Filter by sentiment and export results as CSV

---

## Notes

- NewsAPI's free tier returns up to 100 articles per request; analysis reflects the fetched subset
- Sentiment classification uses VADER's compound score: ≥ 0.05 = Positive, ≤ −0.05 = Negative, otherwise Neutral
- Articles with removed or empty titles are automatically skipped
