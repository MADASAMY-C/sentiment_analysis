# Amazon Review Sentiment Analysis Dashboard

A Streamlit web dashboard that analyzes Amazon customer reviews using a BERT sentiment model. It classifies reviews as Positive, Neutral, or Negative and visualizes ratings, sentiment trends, word clouds, and helpful-vote metrics.

## Features

| Feature | Description |
|---------|-------------|
| **Sentiment Classification** | BERT model (`nlptown/bert-base-multilingual-uncased-sentiment`) classifies each review as Positive, Neutral, or Negative. |
| **Ratings Distribution** | Bar and pie charts showing 1–5 star rating breakdown. |
| **Monthly Sentiment Trend** | Line chart of sentiment counts over time. |
| **Word Clouds** | Word clouds per sentiment with top-word frequency tables. |
| **Helpful Votes Analysis** | Helpful vs. not helpful votes, helpful-ratio distribution, and top helpful reviews. |
| **Interactive Data Table** | Browse the full dataset with column selection and sampling. |
| **Column Aggregation** | Group data by a column and aggregate numeric metrics (mean, sum, max, min, count). |
| **Disk Caching** | Sentiments are cached in `.sentiment_cache/` so re-runs are faster. |

## Project Structure

```
sentiment_analysis/
├── App.py              # Streamlit dashboard application
├── amazon_reviews.csv  # Amazon review dataset
├── requirements.txt    # Python package dependencies
├── README.md           # Project documentation
└── .sentiment_cache/   # Auto-generated sentiment cache (created at runtime)
```

## Prerequisites

- Python 3.9 or later
- Git
- VS Code (recommended)
- Internet access (first run downloads the BERT model)

## Installation

Clone the repository:

```bash
git clone https://github.com/MADASAMY-C/sentiment_analysis.git
cd sentiment_analysis
```

Create and activate a virtual environment:

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run App.py
```

Then open `http://localhost:8501` in your browser.

## How It Works

1. **Load data** – Reads `amazon_reviews.csv` and parses review dates.
2. **Analyze sentiment** – Runs the BERT model over review text (batch size 64). Results are cached to disk under `.sentiment_cache/`.
3. **Explore** – Use the three tabs:
   - **Dashboard** – KPI cards, ratings distribution, sentiment trends, word clouds, helpful-vote charts, and top helpful reviews.
   - **Data** – Full interactive dataset table.
   - **Settings** – Column selector and aggregation tools (group by a column, aggregate numeric metrics).

## Sample Size

By default the app analyzes the first **1,000 reviews** for fast loading. To analyze the full dataset (~50K reviews), click **"Analyze Full Dataset (50K)"** in the sidebar (takes 5–10 minutes).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Missing package | Run `pip install -r requirements.txt` again. |
| Port already in use | Run `streamlit run App.py --server.port 8502`. |
| BERT model download fails | Ensure internet access and restart the app. |
| CSV not found | Run the app from the folder containing `App.py` and `amazon_reviews.csv`. |
| Slow first run | The first run downloads and caches the BERT model; subsequent runs are faster. |

## License

No license is currently specified for this repository.
