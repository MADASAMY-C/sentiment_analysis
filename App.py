import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from wordcloud import WordCloud
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache
import hashlib
import os

# ---------- Styling ----------
st.set_page_config(page_title="Amazon Sentiment Dashboard", layout="wide")

st.markdown("""
    <style>
        .stApp {
            background-color: #FF4E00;
            color: #FAFAFA;
        }
        section[data-testid="stSidebar"] {
            background-color: #002366 !important;
            color: white;
            font-weight: Bold;
            font-family:'Montserrat', sans-serif;
        }
        section[data-testid="stSidebar"] label {
            color: white;
            font-weight: bold;
        }
    section[data-testid="stSidebar"] .stCheckbox > div > label {
        color: #ffffff !important; /* Bright green */
        font-weight: bold;
        font-size: 16px;
    }
        h3 {
        color: #00897b;  /* Teal for headings */
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(path="amazon_reviews.csv", nrows=None):
    df = pd.read_csv(path, parse_dates=['reviewTime'], nrows=nrows)
    df = df.dropna(subset=['reviewText', 'overall', 'reviewTime'])
    return df

@st.cache_resource
def init_bert():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    # Enable GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

def bert_sentiment_batch(texts, tokenizer, model, device, batch_size=64, max_length=128):
    """Process texts in batches for speed."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        stars = probs.argmax(dim=-1).cpu().numpy() + 1
        for s in stars:
            if s >= 4:
                results.append("Positive")
            elif s == 3:
                results.append("Neutral")
            else:
                results.append("Negative")
    return results

def get_cache_key(indices):
    """Generate cache key from review indices (stable across filtering)."""
    content = ",".join(map(str, sorted(indices)))
    return hashlib.md5(content.encode()).hexdigest()[:16]

def load_cached_sentiments(indices, cache_dir=".sentiment_cache"):
    """Load pre-computed sentiments from disk cache."""
    os.makedirs(cache_dir, exist_ok=True)
    key = get_cache_key(indices)
    cache_file = os.path.join(cache_dir, f"sentiments_{key}_{len(indices)}.parquet")
    if os.path.exists(cache_file):
        try:
            cached = pd.read_parquet(cache_file)
            if len(cached) == len(indices):
                return cached['sentiment'].tolist()
        except:
            pass
    return None

def save_cached_sentiments(indices, sentiments, cache_dir=".sentiment_cache"):
    """Save sentiments to disk cache."""
    os.makedirs(cache_dir, exist_ok=True)
    key = get_cache_key(indices)
    cache_file = os.path.join(cache_dir, f"sentiments_{key}_{len(indices)}.parquet")
    df = pd.DataFrame({'text_idx': indices, 'sentiment': sentiments})
    df.to_parquet(cache_file, index=False)

@st.cache_data
def compute_bert_sentiments(texts, indices, _tokenizer, _model, _device):
    # Try disk cache first (using stable indices as key)
    cached = load_cached_sentiments(indices)
    if cached is not None:
        return cached
    # Compute with optimized settings
    sentiments = bert_sentiment_batch(texts, _tokenizer, _model, _device, batch_size=64, max_length=128)
    # Save to disk cache
    save_cached_sentiments(indices, sentiments)
    return sentiments

# ---------- Load and Prepare Data ----------
# Use smaller sample for faster loading (change nrows=None for full dataset)
SAMPLE_SIZE = 1000
df = load_data("amazon_reviews.csv", nrows=SAMPLE_SIZE)
bert_tokenizer, bert_model, device = init_bert()

df['month'] = df['reviewTime'].dt.to_period('M').dt.to_timestamp()

# Compute BERT sentiments (cached)
if 'sentiment_bert' not in df.columns:
    with st.spinner(f"Analyzing {len(df)} reviews with BERT... (one-time)"):
        df['sentiment_bert'] = compute_bert_sentiments(
            df['reviewText'].tolist(), df.index.tolist(), bert_tokenizer, bert_model, device
        )

df['sentiment'] = df['sentiment_bert']

# ---------- Top Header ----------
st.title("Amazon Review Sentiment Analysis Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("💬 Reviews", f"{len(df):,}")
col2.metric("⭐ Avg Rating", f"{df['overall'].mean():.2f}")
col3.metric("👍 Helpful Votes", f"{df.get('helpful_yes', pd.Series(dtype='int')).sum():,}")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📁 Data", "⚙️ Settings"])

# Sidebar: Sample size info
st.sidebar.header("📊 Data Settings")
st.sidebar.info(f"Analyzing {len(df):,} reviews")

if st.sidebar.button("🔄 Analyze Full Dataset (50K)", help="Takes 5-10 min"):
    with st.spinner("Loading full dataset..."):
        df_full = load_data("amazon_reviews.csv", nrows=None)
        df_full['month'] = df_full['reviewTime'].dt.to_period('M').dt.to_timestamp()
        df_full['sentiment_bert'] = compute_bert_sentiments(
            df_full['reviewText'].tolist(), df_full.index.tolist(), bert_tokenizer, bert_model, device
        )
        df_full['sentiment'] = df_full['sentiment_bert']
        st.session_state['df_full'] = df_full
        st.rerun()

# Use full dataset if available
if 'df_full' in st.session_state:
    df = st.session_state['df_full']

# ---------- Tab 1: Dashboard ----------
with tab1:
    st.subheader("📊 Dashboard Overview")

    # KPIs
    total = len(df)
    sentiments_present = df['sentiment'].unique().tolist()
    pos_pct = (df['sentiment'] == "Positive").mean() if "Positive" in sentiments_present else 0
    neg_pct = (df['sentiment'] == "Negative").mean() if "Negative" in sentiments_present else 0
    neu_pct = (df['sentiment'] == "Neutral").mean() if "Neutral" in sentiments_present else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Reviews", f"{total:,}")
    kpi2.metric("Positive %", f"{pos_pct*100:.1f}%")
    kpi3.metric("Negative %", f"{neg_pct*100:.1f}%")
    if "Neutral" in sentiments_present:
        kpi4.metric("Neutral %", f"{neu_pct*100:.1f}%")

    # Rating Distribution Chart (Altair)
    st.write("### Ratings Distribution")
    fig_rating = alt.Chart(df).mark_bar().encode(
        alt.X("overall", bin=alt.Bin(maxbins=5), title="Overall Rating"),
        y='count()'
    ).properties(
        width=600, height=400, title="Ratings Distribution"
    )
    st.altair_chart(fig_rating, use_container_width=True)

    # Sentiment Trend Over Time
    trend = df.groupby(['month', 'sentiment']).size().reset_index(name='count')
    fig_line = px.line(trend, x='month', y='count', color='sentiment',
                       title="Monthly Sentiment Trend", template="plotly_dark")
    st.plotly_chart(fig_line, use_container_width=True)

    # Sentiment Bar
    curr = df['sentiment'].value_counts().reset_index()
    curr.columns = ['sentiment', 'count']
    fig_bar = px.bar(curr, x='sentiment', y='count', color='sentiment',
                     title="Current Sentiment Distribution", template="plotly_dark")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Donut Charts - Sentiment & Rating Distribution
    donut_col1, donut_col2 = st.columns(2)
    sentiments_order = ["Positive", "Negative", "Neutral"]
    colors = {"Positive": "#002366", "Negative": "#FB8500", "Neutral": "#808080"}

    with donut_col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts().reindex(sentiments_order).fillna(0).astype(int)
        sentiment_pct = (sentiment_counts / sentiment_counts.sum() * 100).round(1)
        fig_sentiment = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.5,
            template="plotly_dark",
            color=sentiment_counts.index,
            color_discrete_map=colors,
        )
        fig_sentiment.update_traces(
            textinfo="label+percent+value",
            textfont_size=14,
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>",
            pull=[0.05 if i == sentiment_counts.values.argmax() else 0 for i in range(len(sentiment_counts))],
        )
        fig_sentiment.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            margin=dict(t=30, b=50, l=20, r=20),
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with donut_col2:
        st.subheader("Rating Distribution")
        rating_counts = df['overall'].value_counts().sort_index()
        rating_pct = (rating_counts / rating_counts.sum() * 100).round(1)
        rating_colors = ["#FB8500", "#FFB700", "#FFD600", "#A5D6A7", "#002366"]
        fig_rating = px.pie(
            names=[f"{int(r)}★" for r in rating_counts.index],
            values=rating_counts.values,
            hole=0.5,
            template="plotly_dark",
            color_discrete_sequence=rating_colors,
        )
        fig_rating.update_traces(
            textinfo="label+percent+value",
            textfont_size=14,
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>",
        )
        fig_rating.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            margin=dict(t=30, b=50, l=20, r=20),
        )
        st.plotly_chart(fig_rating, use_container_width=True)

    # sidebar
    st.sidebar.header("Filters")
    
    # ⭐ Filter by star rating
    st.sidebar.header("⭐ Filter by Rating")
    min_rating, max_rating = int(df['overall'].min()), int(df['overall'].max())
    rating_range = st.sidebar.slider("Select rating range", min_rating, max_rating, (min_rating, max_rating))
    df = df[df['overall'].between(rating_range[0], rating_range[1])]
    yr_min, yr_max = int(df['month'].dt.year.min()), int(df['month'].dt.year.max())
    years = st.sidebar.slider("Review year range", yr_min, yr_max, (yr_min, yr_max))
    df = df[df['month'].dt.year.between(years[0], years[1])]
    # ---------- Word Clouds (Enhanced) ----------
    st.header("☁️ Word Clouds by Sentiment")
    
    # Initialize stopwords
    from wordcloud import STOPWORDS
    custom_stopwords = STOPWORDS.union({"product", "amazon", "review", "reviewer", "buy", "bought", "ordered", "order", "item", "items", "one", "two", "get", "got", "would", "could", "also", "really", "very", "much", "like", "just", "make", "made", "well", "good", "great", "best", "love", "hate", "works", "work", "working", "works"})
    
    sentiments_order = ["Positive", "Negative", "Neutral"]
    wc_colors = {"Positive": "Blues", "Negative": "Oranges", "Neutral": "Greys"}
    
    wc_cols = st.columns(3)
    
    for sentiment, col in zip(sentiments_order, wc_cols):
        if sentiment in sentiments_present:
            with col:
                st.subheader(f"{sentiment} Reviews")
                
                # Filter texts
                texts = df[df['sentiment'] == sentiment]['reviewText'].dropna()
                if len(texts) == 0:
                    st.info("No reviews for this sentiment")
                    continue
                
                # Combine texts (use more reviews for better word cloud)
                blob = " ".join(texts.head(500).astype(str))
                
                # Generate word cloud with dark theme
                wc = WordCloud(
                    width=500,
                    height=350,
                    background_color='#0E1117',  # Dark background for Streamlit dark theme
                    colormap=wc_colors[sentiment],
                    stopwords=custom_stopwords,
                    max_words=100,
                    min_font_size=10,
                    max_font_size=80,
                    prefer_horizontal=0.7,
                    relative_scaling=0.5,
                    collocations=False,
                    contour_width=0,
                ).generate(blob)
                
                # Display word cloud
                fig_wc, ax_wc = plt.subplots(figsize=(6, 4), facecolor='#0E1117')
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                ax_wc.set_facecolor('#0E1117')
                st.pyplot(fig_wc, use_container_width=True)
                plt.close(fig_wc)
                
                # Show top words as a table for clarity
                word_freq = wc.words_
                if word_freq:
                    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
                    top_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                    top_df['Frequency'] = top_df['Frequency'].round(3)
                    st.dataframe(top_df, hide_index=True, use_container_width=True, height=250)

    # Sentiment Score Histogram (using BERT confidence proxy)
    st.header("Sentiment Distribution")
    fig_hist = px.histogram(
        df, x="overall", nbins=5,
        title="Rating Distribution (1-5 Stars)",
        template="plotly_dark",
        color_discrete_sequence=["#002366"],
        marginal="rug", histnorm="percent", text_auto=True
    )
    fig_hist.update_layout(
        xaxis_title="Star Rating",
        yaxis_title="Percentage of Reviews",
        bargap=0.05,
        barcornerradius=50
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ---------- Helpful Votes Analysis ----------
    cols_needed = ['helpful_yes', 'helpful_no', 'total_vote']
    if all(col in df.columns for col in cols_needed):
        st.subheader("👍 Helpful Votes Analysis")
        
        # Calculate metrics
        total_yes = df['helpful_yes'].sum()
        total_no = df['helpful_no'].sum()
        total_votes = df['total_vote'].sum()
        avg_helpful_ratio = (df['helpful_yes'] / df['total_vote'].replace(0, np.nan)).mean()
        
        # KPI cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("👍 Total Helpful Yes", f"{total_yes:,}")
        kpi2.metric("👎 Total Helpful No", f"{total_no:,}")
        kpi3.metric("📊 Total Votes Cast", f"{total_votes:,}")
        kpi4.metric("📈 Avg Helpful Ratio", f"{avg_helpful_ratio:.1%}")
        
        # Two-column layout for charts
        hv_col1, hv_col2 = st.columns(2)
        
        with hv_col1:
            st.markdown("**Helpful vs Not Helpful**")
            fig_hv = px.pie(
                names=['Helpful Yes', 'Helpful No'],
                values=[total_yes, total_no],
                hole=0.5,
                template="plotly_dark",
                color_discrete_sequence=['#00C851', '#FF4444'],
            )
            fig_hv.update_traces(
                textinfo="label+percent+value",
                textfont_size=14,
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>",
            )
            fig_hv.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                margin=dict(t=10, b=50, l=20, r=20),
            )
            st.plotly_chart(fig_hv, use_container_width=True)
        
        with hv_col2:
            st.markdown("**Helpful Ratio Distribution (per review)**")
            # Calculate helpful ratio per review (avoid division by zero)
            df_votes = df[df['total_vote'] > 0].copy()
            df_votes['helpful_ratio'] = df_votes['helpful_yes'] / df_votes['total_vote']
            
            fig_ratio = px.histogram(
                df_votes, x='helpful_ratio', nbins=20,
                template="plotly_dark",
                color_discrete_sequence=['#002366'],
                marginal="box",
            )
            fig_ratio.update_layout(
                xaxis_title="Helpful Ratio (Yes / Total Votes)",
                yaxis_title="Number of Reviews",
                bargap=0.05,
                margin=dict(t=10, b=40, l=40, r=20),
            )
            fig_ratio.add_vline(x=avg_helpful_ratio, line_dash="dash", line_color="#FB8500",
                               annotation_text=f"Avg: {avg_helpful_ratio:.1%}")
            st.plotly_chart(fig_ratio, use_container_width=True)
        
        # Helpful ratio by sentiment
        st.markdown("**Average Helpful Ratio by Sentiment**")
        if 'sentiment' in df.columns:
            sentiment_helpful = df[df['total_vote'] > 0].groupby('sentiment').apply(
                lambda x: (x['helpful_yes'] / x['total_vote']).mean()
            ).reset_index()
            sentiment_helpful.columns = ['Sentiment', 'Avg Helpful Ratio']
            sentiment_helpful['Avg Helpful Ratio'] = (sentiment_helpful['Avg Helpful Ratio'] * 100).round(1)
            
            colors_map = {"Positive": "#002366", "Negative": "#FB8500", "Neutral": "#808080"}
            fig_sent_hv = px.bar(
                sentiment_helpful, x='Sentiment', y='Avg Helpful Ratio',
                color='Sentiment', color_discrete_map=colors_map,
                template="plotly_dark", text='Avg Helpful Ratio',
            )
            fig_sent_hv.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_sent_hv.update_layout(
                yaxis_title="Average Helpful Ratio (%)",
                showlegend=False,
                margin=dict(t=10, b=40, l=40, r=20),
            )
            st.plotly_chart(fig_sent_hv, use_container_width=True)
        
        # Top helpful reviews table
        st.markdown("**Top 10 Most Helpful Reviews**")
        top_helpful = df.nlargest(10, 'helpful_yes')[['reviewerName', 'overall', 'helpful_yes', 'helpful_no', 'total_vote', 'reviewText']]
        top_helpful['Helpful Ratio'] = (top_helpful['helpful_yes'] / top_helpful['total_vote'].replace(0, np.nan) * 100).round(1)
        top_helpful = top_helpful.rename(columns={
            'reviewerName': 'Reviewer', 'overall': 'Rating',
            'helpful_yes': '👍 Yes', 'helpful_no': '👎 No',
            'total_vote': 'Total Votes', 'reviewText': 'Review Text'
        })
        st.dataframe(top_helpful, hide_index=True, use_container_width=True, height=350)

# ---------- Tab 2: Data Table ----------
with tab2:
    st.subheader("📁 Full Dataset")
    st.dataframe(df)

# ---------- Tab 3: Settings + Aggregation ----------
with tab3:
    st.subheader("⚙️ Settings")

    # Column Selector
    selected_cols = st.multiselect("Choose columns to display:", df.columns.tolist(), default=["reviewerName", "overall"])
    st.dataframe(df[selected_cols].head(20))

    # Grouping
    st.sidebar.header("Column Filters")
    group_by_col = st.sidebar.selectbox("Group by column (categorical/time):",
                                        ['reviewTime', 'reviewerName'])

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_metrics = st.sidebar.multiselect("Select numeric columns to chart:",
                                              numeric_cols, default=["overall", "helpful_yes"])

    agg_func = st.sidebar.selectbox("Aggregation function:", ['mean', 'sum', 'max', 'min', 'count'])

    if group_by_col == "reviewTime":
        df['reviewTime'] = pd.to_datetime(df['reviewTime'], errors='coerce')

    if group_by_col and selected_metrics:
        agg_df = df.groupby(group_by_col)[selected_metrics].agg(agg_func).reset_index()
        st.subheader(f"{agg_func.upper()} of selected metrics by {group_by_col}")
        st.dataframe(agg_df)

        melted = agg_df.melt(id_vars=[group_by_col], value_vars=selected_metrics,
                             var_name='Metric', value_name='Value')
        chart = alt.Chart(melted).mark_line(point=True).encode(
            x=group_by_col + ':T' if group_by_col == "reviewTime" else group_by_col + ':O',
            y='Value:Q',
            color='Metric:N',
            tooltip=[group_by_col, 'Metric', 'Value']
        ).properties(
            width=800,
            height=400,
            title=f"{agg_func.capitalize()} of Metrics by {group_by_col}"
        ).interactive()
        st.altair_chart(chart)
    else:
        st.warning("Please select a group column and at least one numeric metric.")

# ---------- Optional Debug ----------
st.write("Selected columns:", selected_cols)

st.subheader("📝 Sample Reviews with Sentiment")
st.caption("Model: BERT (nlptown/bert-base-multilingual-uncased-sentiment)")

# Show balanced sample: 5 from each sentiment
sample_dfs = []
for sent in ["Positive", "Negative", "Neutral"]:
    subset = df[df['sentiment'] == sent]['reviewText'].head(5)
    if len(subset) > 0:
        sample_dfs.append(pd.DataFrame({'reviewText': subset, 'sentiment': sent}))
if sample_dfs:
    sample_df = pd.concat(sample_dfs, ignore_index=True)
    st.dataframe(sample_df)
else:
    st.dataframe(df[['reviewText', 'sentiment']].head(10))

# Save full results before applying filters
df_all_results = df.copy()