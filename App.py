import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
from wordcloud import WordCloud

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

# ---------- Download resources ----------
nltk.download('vader_lexicon')



@st.cache_data
def load_data(path="amazon_reviews.csv", nrows=None):
    df = pd.read_csv(path, parse_dates=['reviewTime'], nrows=nrows)
    df = df.dropna(subset=['reviewText', 'overall', 'reviewTime'])
    return df

@st.cache_resource
def init_sia():
    return SentimentIntensityAnalyzer()

# ---------- Load and Prepare Data ----------
df = load_data("amazon_reviews.csv", nrows=50000)
sia = init_sia()

df['compound'] = df['reviewText'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['sentiment'] = df['compound'].apply(lambda c: "Positive" if c >= 0 else "Negative")
df['month'] = df['reviewTime'].dt.to_period('M').dt.to_timestamp()

# ---------- Top Header ----------
st.title("Amazon Review Sentiment Analysis Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("💬 Reviews", f"{len(df):,}")
col2.metric("⭐ Avg Rating", f"{df['overall'].mean():.2f}")
col3.metric("👍 Helpful Votes", f"{df.get('helpful_yes', pd.Series(dtype='int')).sum():,}")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📁 Data", "⚙️ Settings"])

# ---------- Tab 1: Dashboard ----------
with tab1:
    st.subheader("📊 Dashboard Overview")

    # KPIs
    total = len(df)
    pos_pct = (df['sentiment'] == "Positive").mean()
    neg_pct = (df['sentiment'] == "Negative").mean()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Reviews", f"{total:,}")
    kpi2.metric("Positive %", f"{pos_pct*100:.1f}%")
    kpi3.metric("Negative %", f"{neg_pct*100:.1f}%")

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

    # Donut Chart
    donut_cols = st.columns(2)
    for sentiment, col in zip(["Positive", "Negative"], donut_cols):
        frac = (df['sentiment'] == sentiment).mean()
        fig = px.pie(names=[sentiment, ""], values=[frac, 1-frac],
                     hole=0.7, template="plotly_dark")
        fig.update_traces(showlegend=False, textinfo="percent",
                          marker_colors=["#002366", "#FB8500"])
        col.subheader(f"{sentiment} Reviews")
        col.plotly_chart(fig, use_container_width=True)

  

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
    st.header("Word Clouds")
    wc_cols = st.columns(2)
    for sentiment, col in zip(["Positive", "Negative"], wc_cols):
            texts = df[df['sentiment'] == sentiment]['reviewText'].dropna().head(200)
            blob = " ".join(texts)
            if blob:
                wc = WordCloud(width=400, height=300, background_color='white').generate(blob)
                col.subheader(sentiment)
                col.image(wc.to_array(), use_container_width=True)


    # Sentiment Score Histogram
    st.header("Sentiment Strength Distribution")
    fig_hist = px.histogram(
        df, x="compound", nbins=40,
        title="Distribution of VADER Compound Scores",
        template="plotly_dark",
        color_discrete_sequence=["#002366"],
        marginal="rug", histnorm="percent", text_auto=True
    )
    fig_hist.update_layout(
        xaxis_title="Compound Score (−1 to 1)",
        yaxis_title="Percentage of Reviews",
        bargap=0.05,
        barcornerradius=50
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Pie Chart: Helpful Votes
    cols_needed = ['helpful_yes', 'helpful_no', 'total_vote']
    if all(col in df.columns for col in cols_needed):
        sizes = (df['helpful_yes'].sum(), df['helpful_no'].sum(), df['total_vote'].sum())
        labels = ['Helpful Yes', 'Helpful No', 'Total Vote']
        custom_colors = ['navy', 'yellow', 'orange']
        fig_pie, ax = plt.subplots(figsize=(5, 5))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=custom_colors)
        ax.set_title('Overall Distribution of Review Metrics')
        ax.axis('equal')
        st.pyplot(fig_pie)

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
st.dataframe(df[['reviewText', 'sentiment']].head(10))

# Save full results before applying filters
df_all_results = df.copy()

