import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
import time

st.markdown(
    """
    <style>
        .stApp {
            background-color: #FF4E00;
            color: #FAFAFA;
        }
        .css-xyz {
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("""
<style>
[data-testid="stSidebar"] {background-color: #002366 !important;}
[data-testid="stSidebarContent"] {color: #FFFFFF !important;}
[data-testid="stSidebarNav"] span {color: #FFFFFF !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.custom-chart .plotly-graph-div {
  border-radius: 12px;
  overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

  

nltk.download('vader_lexicon')

@st.cache_data
def load_data(path, nrows=None):
    df = pd.read_csv(path, parse_dates=['reviewTime'], nrows=nrows)
    df = df[['reviewText', 'overall', 'reviewTime']].dropna()
    return df

@st.cache_resource
def init_sia():
    return SentimentIntensityAnalyzer()

st.set_page_config(page_title="Amazon Sentiment Dashboard", layout="wide")
st.title("Amazon Review Sentiment Analysis Dashboard")

# Load
df = load_data("amazon_reviews.csv", nrows=50000)
sia = init_sia()
df['compound'] = df['reviewText'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['sentiment'] = df['compound'].apply(lambda c: "Positive" if c >= 0 else "Negative")
df['month'] = df['reviewTime'].dt.to_period('M').dt.to_timestamp()

# Sidebar filter
st.sidebar.header("Filters")
yr_min, yr_max = int(df['month'].dt.year.min()), int(df['month'].dt.year.max())
years = st.sidebar.slider("Review year range", yr_min, yr_max, (yr_min, yr_max))
df = df[df['month'].dt.year.between(years[0], years[1])]

# Top KPI metrics
total = len(df)
pos_pct = (df['sentiment'] == "Positive").mean()
neg_pct = (df['sentiment'] == "Negative").mean()

c1, c2, c3 = st.columns(3)
c1.metric("Total Reviews", f"{total:,}")
c2.metric("Positive %", f"{pos_pct*100:.1f}%")
c3.metric("Negative %", f"{neg_pct*100:.1f}%")

# Sentiment trend over time
trend = df.groupby(['month', 'sentiment']).size().reset_index(name='count')
fig_line = px.line(trend, x='month', y='count', color='sentiment',
                   title="Monthly Sentiment Trend", template="plotly_dark")
st.plotly_chart(fig_line, use_container_width=True)

# Current sentiment distribution
curr = df['sentiment'].value_counts().reset_index()
curr.columns = ['sentiment', 'count']
fig_bar = px.bar(curr, x='sentiment', y='count', color='sentiment',
                 title="Current Sentiment Distribution", template="plotly_dark")
st.plotly_chart(fig_bar, use_container_width=True)

# Donut-style KPI cards
cols = st.columns(2)
for sentiment, col in zip(["Positive", "Negative"], cols):
    frac = (df['sentiment'] == sentiment).mean()
    fig = px.pie(names=[sentiment, ""], values=[frac, 1-frac],
                 hole=0.7, template="plotly_dark")
    fig.update_traces(showlegend=False, textinfo="percent",
                      marker_colors=["#002366" if sentiment=="Positive" else "#002366", "#FB8500"])
    col.subheader(f"{sentiment} Reviews")
    col.plotly_chart(fig, use_container_width=True)

# Optional word clouds
if st.sidebar.checkbox("Show Word Clouds", True):
    st.header("Word Clouds")
    wc_cols = st.columns(2)
    for sentiment, col in zip(["Positive", "Negative"], wc_cols):
        texts = df[df['sentiment']==sentiment]['reviewText'].dropna().head(200)
        blob = " ".join(texts)
        if blob:
            wc = WordCloud(width=400, height=300, background_color='white').generate(blob)
            col.subheader(sentiment)
            col.image(wc.to_array(), use_container_width=True)
            
# Sentiment strength histogram
st.header("Sentiment Strength Distribution")
fig_hist = px.histogram(
    df,
    x="compound",
    nbins=40,
    title="Distribution of VADER Compound Scores",
    template="plotly_dark",
    color_discrete_sequence=["#002366"],
    marginal="rug",         # show rug plot under axis
    histnorm="percent",     # normalize counts to percentage
    text_auto=True          # show % values on bars :contentReference[oaicite:2]{index=2}
)
fig_hist.update_layout(
    xaxis_title="Compound Score (−1 to 1)",
    yaxis_title="Percentage of Reviews",
    bargap=0.05,
    barcornerradius=50 # 🎯 Add this line for rounded corners
)
st.plotly_chart(fig_hist, use_container_width=True)

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.3, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)

@st.fragment
def release_the_balloons():
    st.button("Release the balloons", help="Fragment rerun")
    st.balloons()

with st.spinner("Inflating balloons..."):
    time.sleep(5)
release_the_balloons()
st.button("Inflate more balloons", help="Full rerun")
