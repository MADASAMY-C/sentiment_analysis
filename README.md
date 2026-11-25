# Amazon Sentiment Analysis Dashboard

A Python Streamlit dashboard for analyzing sentiment in Amazon reviews. The app loads the included `amazon_reviews.csv` dataset, analyzes review text with NLTK's VADER sentiment analyzer, and displays ratings, sentiment trends, distributions, and word clouds.

## Features

- Amazon review sentiment classification
- Positive and negative sentiment percentages
- Monthly sentiment trends
- Rating distribution and sentiment strength charts
- Positive and negative review word clouds
- Interactive data table and column aggregation tools

## Prerequisites

- Python 3.9 or later
- Git
- VS Code (recommended)
- Internet access on the first run to download the NLTK VADER lexicon

## Clone the Project

```bash
git clone https://github.com/MADASAMY-C/sentiment_analysis.git
cd sentiment_analysis
```

## Run in VS Code

1. Open the project folder:

   ```bash
   code .
   ```

2. Open **Terminal → New Terminal** in VS Code.

3. Create a virtual environment:

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

4. Install the required packages:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install numpy altair
   ```

5. Start the Streamlit app:

   ```bash
   streamlit run App.py
   ```

6. Open `http://localhost:8501` in a browser if VS Code does not open it automatically.

VS Code may prompt you to select a Python interpreter. Select `.venv` from the project folder.

## Run with the Activated Environment

After activating the virtual environment, no VS Code terminal setup is needed for subsequent runs:

```bash
streamlit run App.py
```

## Project Files

- `App.py` - Streamlit dashboard application
- `amazon_reviews.csv` - Amazon review dataset
- `requirements.txt` - Python package dependencies

## Troubleshooting

- **Missing package:** Run `pip install -r requirements.txt` again.
- **Missing `numpy` or `altair`:** Run `pip install numpy altair`.
- **Port already in use:** Run `streamlit run App.py --server.port 8502`.
- **NLTK download failure:** Keep internet access enabled and restart the app.
- **`FileNotFoundError` for the CSV:** Start the app from the project directory containing `App.py` and `amazon_reviews.csv`.

## License

No license is currently specified for this repository.
