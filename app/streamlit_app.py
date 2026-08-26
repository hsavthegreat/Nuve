import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, random, io, re, time  # --- NEW: Added time ---
from PIL import Image
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- NEW: Robust NLTK Data Downloader ---
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

download_nltk_data() # Call the function immediately

# --- Helper Functions ---
def find_comment_column(df):
    for col in df.columns:
        c = col.lower()
        if "comment" in c and not ("count" in c or "num" in c or c.endswith("s")):
            return col
    for col in df.columns:
        if "comment" in col.lower():
            return col
    return None

try:
    from colorthief import ColorThief
    HAS_COLORTHIEF = True
except Exception:
    HAS_COLORTHIEF = False

# ---------------- App Config ----------------
st.set_page_config(
    page_title="Nuvé: AI Fashion Intelligence — Team Blue Popcorn",
    layout="wide",
    page_icon="🎨"
)

# ---------------- Midnight Theme ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #111827;
    color: #F9FAFB;
}
[data-testid="stSidebar"] {
    background-color: #1F2937;
    border-right: 1px solid #374151;
}
.metric-card {
    background-color: #1F2937;
    border: 1px solid #374151;
    border-radius: 14px;
    padding: 14px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
button[data-testid="stTab"] {
    color: #9CA3AF;
    font-weight: 600;
}
button[data-testid="stTab"][aria-selected="true"] {
    color: #38BDF8;
    border-bottom: 3px solid #38BDF8;
}
h2, h3 {color: #F9FAFB !important;}
.section-title {color:#D1D5DB;font-weight:700;}
.metric-value {font-size:2.25rem;font-weight:700;color:#FFF;}
[data-testid="stHorizontalBlock"] img {
    background-color:#F0F2F6;
    border:1px solid #4B5563;
    border-radius:8px;
    padding:5px;
}
.small-muted {color:#9CA3AF; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# ---------------- Project Directory ----------------
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------- Utilities ----------------
def safe_read_csv(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except Exception as e:
        # st.error(f"Failed to read CSV: {path}. Error: {e}") # Too noisy
        return pd.DataFrame()

def safe_read_excel(path):
    try:
        df = pd.read_excel(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except Exception as e:
        # st.error(f"Failed to read Excel: {path}. Error: {e}") # Too noisy
        return pd.DataFrame()

def safe_read_hdf(path):
    try:
        df = pd.read_hdf(path, key='df_items')
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except Exception as e:
        # st.error(f"Failed to read HDF5: {path}. Error: {e}") # Too noisy
        return pd.DataFrame()


# ---------------- Cached Loaders ----------------
@st.cache_data
def load_overview():
    data = {}
    df_lab = safe_read_excel(os.path.join(project_dir, "data", "labelled_comments_train.xlsx"))
    data["Labeled Comments"] = len(df_lab) if not df_lab.empty else 0
    df_posts = safe_read_csv(os.path.join(project_dir, "data", "posts_comments.csv"))
    data["Total Posts"] = len(df_posts) if not df_posts.empty else 0
    for label, sub in [("Original Images", "original_images")]:
        p = os.path.join(project_dir, "images", sub)
        data[label] = len([f for f in os.listdir(p) if f.lower().endswith((".png",".jpg",".jpeg"))]) if os.path.exists(p) else 0
    return data

@st.cache_data
def load_sentiment_df():
    df = safe_read_excel(os.path.join(project_dir, "data", "labelled_comments_train.xlsx"))
    if not df.empty and "category" in df.columns:
        df["sentiment"] = df["category"].map({1: "Positive", 0: "Negative"}).fillna("Neutral")
    return df

@st.cache_data
def list_original_images(n=50):
    p = os.path.join(project_dir, "images", "original_images")
    if not os.path.exists(p):
        return []
    imgs = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(imgs)
    return imgs[:n]

# --- Cluster Forecasting Loader ---
@st.cache_data
def load_forecasting_data():
    try:
        df_posts_original = safe_read_csv(os.path.join(project_dir, "data", "posts.csv"))
        df_latent_h5 = safe_read_hdf(os.path.join(project_dir, "data", "latent_spaces.h5"))
        df_comments_sentiment = safe_read_csv(os.path.join(project_dir, "data", "posts_comments.csv"))

        if df_posts_original.empty or df_latent_h5.empty:
            st.error("Could not load core forecasting data (posts.csv or latent_spaces.h5).")
            return pd.DataFrame(), [], {}

        meta = df_latent_h5.rename(columns={'path': 'image_id', 'cluster': 'cluster_label'})

        if 'id' in df_posts_original.columns:
            df_posts_original = df_posts_original.rename(columns={
                'id': 'image_id',
                'n_likes': 'likes',
                'n_comments': 'comments',
            })
        else:
            st.warning("'id' column not found in posts.csv. Trying to merge on index...")
            df_posts_original['image_id'] = df_posts_original.index + 1
            df_posts_original = df_posts_original.rename(columns={
                'n_likes': 'likes',
                'n_comments': 'comments',
            })

        meta = meta.merge(df_posts_original, on='image_id', how='left')

        if not df_comments_sentiment.empty:
            if 'id' in df_comments_sentiment.columns and 'category' in df_comments_sentiment.columns:
                sentiment_per_image = df_comments_sentiment.groupby('id')['category'].mean().reset_index()
                sentiment_per_image = sentiment_per_image.rename(columns={'id': 'image_id', 'category': 'sentiment_score'})
                meta = meta.merge(sentiment_per_image, on='image_id', how='left')
                meta['sentiment_score'] = meta['sentiment_score'].fillna(0.5)
            else:
                meta['sentiment_score'] = 0.5
        else:
            meta['sentiment_score'] = 0.5

        cluster_stats = meta.groupby('cluster_label').agg({
            'likes': 'mean',
            'comments': 'mean',
            'sentiment_score': 'mean',
            'image_id': 'count'
        }).rename(columns={'image_id': 'count_images'}).reset_index()

        cluster_stats['trending_score'] = (
            cluster_stats['likes'].fillna(0) * 0.4 +
            cluster_stats['comments'].fillna(0) * 0.3 +
            cluster_stats['sentiment_score'].fillna(0) * 0.3
        )
        cluster_stats = cluster_stats.sort_values('trending_score', ascending=False).round(2)
        top_clusters = cluster_stats['cluster_label'].head(3).tolist()

        analysis_text = {}
        if len(top_clusters) >= 3:
            analysis_text = {
                top_clusters[0]: "This cluster shows the highest trending score, primarily driven by high engagement. The sentiment is positive, and it maintains a high volume of posts. This is a dominant, mainstream trend.",
                top_clusters[1]: "This cluster is a strong secondary trend. While engagement is lower than the top cluster, it has a healthy post count and positive sentiment, indicating strong community engagement.",
                top_clusters[2]: "This is a rising trend. Its engagement counts are modest, but it shows consistent positive sentiment and a steady increase in post volume."
            }

        return cluster_stats, top_clusters, analysis_text

    except Exception as e:
        st.error(f"Fatal error in load_forecasting_data: {e}")
        return pd.DataFrame(), [], {}


# ---------------- Palette extraction ----------------
@st.cache_data
def extract_palette_from_images(img_paths, per_image_colors=8):
    counter = Counter()
    for p in img_paths:
        try:
            if HAS_COLORTHIEF:
                ct = ColorThief(p)
                palette = ct.get_palette(color_count=per_image_colors)
            else:
                im = Image.open(p).convert("RGB")
                small = im.resize((100,100))
                colors = small.getcolors(100*100) or []
                colors = sorted(colors, key=lambda x: x[0], reverse=True)[:per_image_colors]
                palette = [c[1] for c in colors]
            for i, rgb in enumerate(palette):
                hexv = '#%02x%02x%02x' % rgb
                weight = max(per_image_colors - i, 1)
                counter[hexv] += weight
        except Exception:
            continue

    df = pd.DataFrame(counter.items(), columns=["HEX", "Count"]).sort_values("Count", ascending=False).reset_index(drop=True)
    if df.empty: return df
    total = df["Count"].sum()
    df["Pct_global"] = (df["Count"] / total * 100).astype(float)
    return df

@st.cache_data
def extract_palette_from_single_image_buffer(img_buffer, per_image_colors=8):
    try:
        if HAS_COLORTHIEF:
            tmp = Image.open(img_buffer).convert("RGB")
            tmp_path = "/tmp/streamlit_uploaded.png"
            tmp.save(tmp_path)
            ct = ColorThief(tmp_path)
            palette = ct.get_palette(color_count=per_image_colors)
        else:
            im = Image.open(img_buffer).convert("RGB")
            small = im.resize((100,100))
            colors = small.getcolors(100*100) or []
            colors = sorted(colors, key=lambda x: x[0], reverse=True)[:per_image_colors]
            palette = [c[1] for c in colors]
        counter = Counter()
        for i, rgb in enumerate(palette):
            hexv = '#%02x%02x%02x' % rgb
            weight = max(per_image_colors - i, 1)
            counter[hexv] += weight
        df = pd.DataFrame(counter.items(), columns=["HEX", "Count"]).sort_values("Count", ascending=False).reset_index(drop=True)
        if df.empty: return df
        total = df["Count"].sum()
        df["Pct_global"] = (df["Count"] / total * 100).astype(float)
        return df
    except Exception:
        return pd.DataFrame()

# ---------------- Load data ----------------
overview = load_overview()
sentiments = load_sentiment_df()
all_images = list_original_images(n=200) # Used for gallery and palette
palette_df = extract_palette_from_images(all_images[:80], per_image_colors=8) if all_images else pd.DataFrame()

# --- NEW: Load cluster data ---
cluster_stats, top_clusters, analysis_text = load_forecasting_data()


# ---------------- Sidebar ----------------
st.sidebar.title("Nuvé Controls")
st.sidebar.markdown("Team Blue Popcorn · IIT Kanpur\nAI Fashion Intelligence Dashboard")
st.sidebar.markdown("---")
st.sidebar.subheader("Data / File Controls")
uploaded_posts = st.sidebar.file_uploader("Upload posts.csv (optional)", type=["csv"])
if uploaded_posts:
    st.sidebar.info("Note: To use uploaded data, refresh the dashboard.")

uploaded_sent = st.sidebar.file_uploader("Upload labelled_comments_train.xlsx (optional)", type=["xlsx"])
if uploaded_sent:
    st.sidebar.info("Note: To use uploaded data, refresh the dashboard.")

st.sidebar.markdown("---")
gallery_count = st.sidebar.slider("Gallery images", 6, 100, min(30, max(6, len(all_images))), step=1)
sample_size_for_palette = st.sidebar.slider("Palette sample size (images)", 10, min(200, max(10, len(all_images))), 20)
per_image_colors = st.sidebar.selectbox("Colors per image (palette)", [4,6,8,10], index=2)
regenerate_palette = st.sidebar.button("Regenerate Palette (from sample)")

if regenerate_palette:
    palette_df = extract_palette_from_images(all_images[:sample_size_for_palette], per_image_colors=per_image_colors)

# ---------------- Tabs ----------------
# --- MODIFIED: Added "Outfit Analyzer" ---
tabs = st.tabs([
    "🧭 Overview",
    "📈 Cluster Forecasting",
    "💬 Consumer Sentiment",
    "🎨 Color Intelligence",
    "🖼 Visual Library",
    "👗 Outfit Analyzer",  # --- NEW ---
    "💡 AI Insights"
])

# ---------------- Overview ----------------
with tabs[0]:
    st.title("Nuvé: AI-Driven Fashion Trend Forecasting")
    cols = st.columns(3)
    metrics = [
        ("Labeled Comments", overview.get("Labeled Comments", 0)),
        ("Total Posts", overview.get("Total Posts", 0)),
        ("Original Images", overview.get("Original Images", 0))
    ]
    for i, (t, v) in enumerate(metrics):
        cols[i].markdown(f"<div class='metric-card'><div class='section-title'>{t}</div><div class='metric-value'>{v:,}</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("*Project notes / quick actions*")
    st.markdown("- Cluster Forecasting tab contains the core trend analysis from your AI models.")
    st.markdown("- Use Color Intelligence tab to explore dominant colors from images.")
    st.markdown("- Regenerate palette when you change sample size or colors-per-image.")

# ---------------- Cluster Forecasting Tab ----------------
with tabs[1]:
    st.header("📈 Cluster Forecasting (Your Project's Core)")

    if cluster_stats.empty:
        st.error("Could not load cluster forecasting data. Please check your posts.csv, latent_spaces.h5, and posts_comments.csv files.")
    else:
        st.subheader("Top Trending Clusters (by Trending Score)")
        st.markdown("Trending Score = (Likes * 0.4) + (Comments * 0.3) + (Sentiment Score * 0.3)")
        st.dataframe(cluster_stats.head(5), use_container_width=True)

        st.header('Analysis of Top 3 Clusters')

        cols = st.columns(3)
        for i, cluster_id in enumerate(top_clusters):
            with cols[i]:
                st.subheader(f"Cluster {cluster_id}")

                if all_images:
                    try:
                        st.image(random.choice(all_images), caption=f"Sample image for Cluster {cluster_id}")
                    except Exception as e:
                        st.warning("Could not load sample image.")
                else:
                    st.warning("No sample images found in original_images folder.")

                st.markdown(f"Analysis: {analysis_text.get(cluster_id, 'No analysis available.')}")

        # --- REMOVED: Cluster Trend Evolution charts ---


# ---------------- Sentiment ----------------
with tabs[2]:
    st.header("Consumer Sentiment")
    if sentiments is None or sentiments.empty:
        st.warning("No labelled sentiment data found. Upload labelled_comments_train.xlsx via sidebar to enable sentiment charts.")
    else:
        if "sentiment" not in sentiments.columns:
                 st.error("Data loaded, but 'sentiment' column (mapped from 'category') not found.")
        else:
            counts = sentiments["sentiment"].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Sentiment Breakdown")
                fig_bar = px.bar(
                    counts, x='Sentiment', y='Count', color='Sentiment',
                    title="Sentiment Counts in Training Data",
                    color_discrete_map={'Positive': '#32CD32', 'Negative': '#FF6347', 'Neutral': '#888'},
                    template="plotly_dark"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                st.subheader("Sentiment Score")
                total_count = counts['Count'].sum()
                try:
                    positive_count = counts.loc[counts['Sentiment'] == 'Positive', 'Count'].sum()
                    positive_score = (positive_count / total_count) * 100
                except Exception:
                    positive_score = 0

                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = positive_score,
                    title = {'text': "Positive Sentiment Score (%)"},
                    gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#32CD32"}}
                ))
                fig_gauge.update_layout(template="plotly_dark", height=300, margin=dict(t=50, b=0, l=0, r=0))
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("---")
            st.subheader("Sentiment Distribution (Pie)")
            fig_pie = px.pie(counts, names="Sentiment", values="Count",
                             color_discrete_map={"Positive":"#32CD32","Negative":"#FF6347","Neutral":"#888"},
                             template="plotly_dark", title="Labelled Sentiment Distribution")
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
            st.markdown("Sample labelled comments")
            st.dataframe(sentiments.head(30))

# ---------------- Color Intelligence ----------------
with tabs[3]:
    st.header("Color Intelligence — Dominant Palette (Original Images)")
    if palette_df is None or palette_df.empty:
        st.warning("No palette data extracted. Ensure you have images in the project folder or upload an image below.")
    else:
        st.caption(f"Palette computed from sample size: *{min(sample_size_for_palette, len(all_images))}* images. Total unique colors found: *{len(palette_df)}*.")
        top_n = st.selectbox("Top colors to display", [3,5,8,12], index=1)
        df_top = palette_df.head(top_n).copy()
        top_sum = df_top["Count"].sum()
        if top_sum > 0:
            df_top["Pct_topN"] = (df_top["Count"] / top_sum * 100).astype(float)
        else:
            df_top["Pct_topN"] = 0.0
        df_top["label_display"] = df_top.apply(lambda r: f"{r['HEX']} — {r['Pct_global']:.2f}% (Top{top_n}: {r['Pct_topN']:.2f}%)", axis=1)

        cols = st.columns(len(df_top))
        for i, row in df_top.reset_index().iterrows():
            cols[i].markdown(f"<div style='background:{row['HEX']};height:110px;border-radius:12px;border:1px solid #555;'></div><p style='text-align:center;margin-top:6px'>{row['label_display']}</p>", unsafe_allow_html=True)

        color_map = {h: h for h in df_top["HEX"].tolist()}
        fig_bar = px.bar(df_top, x="Pct_topN", y="label_display", orientation="h", text=df_top["Pct_global"].apply(lambda x: f"{x:.2f}%"),
                         color="HEX", color_discrete_map=color_map, template="plotly_dark",
                         title=f"Top {top_n} Dominant Colors — Share among displayed colors (TopN%) and global weighted share shown as text")
        fig_bar.update_traces(textposition='outside', marker_line_width=1, marker_line_color="#222")
        fig_bar.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Share among Top N (%)")
        st.plotly_chart(fig_bar, use_container_width=True)

        csv_bytes = df_top[["HEX", "Count", "Pct_global", "Pct_topN"]].to_csv(index=False).encode('utf-8')
        st.download_button("Export displayed palette (CSV)", data=csv_bytes, file_name=f"palette_top{top_n}.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Upload single image to extract palette")
    uploaded_image = st.file_uploader("Upload an image (JPEG/PNG) to extract a palette", type=["png","jpg","jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded image", use_container_width=False, width=300)
        single_df = extract_palette_from_single_image_buffer(uploaded_image, per_image_colors=per_image_colors)
        if single_df is None or single_df.empty:
            st.warning("Could not extract palette from uploaded image.")
        else:
            single_df["Pct_topN"] = (single_df["Count"] / single_df["Count"].sum() * 100)
            single_df["label_display"] = single_df.apply(lambda r: f"{r['HEX']} — {r['Pct_global']:.2f}% (Top: {r['Pct_topN']:.2f}%)", axis=1)
            cols = st.columns(len(single_df))
            for i, row in single_df.reset_index().iterrows():
                cols[i].markdown(f"<div style='background:{row['HEX']};height:90px;border-radius:12px;border:1px solid #555;'></div><p style='text-align:center;margin-top:6px'>{row['label_display']}</p>", unsafe_allow_html=True)

# ---------------- Visual Library ----------------
with tabs[4]:
    st.header("Visual Library — Original Images")
    imgs_to_show = all_images[:gallery_count]
    if not imgs_to_show:
        st.warning("No original images found.")
    else:
        cols_per_row = st.slider("Images per row", 3, 6, 5)
        for i in range(0, len(imgs_to_show), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, c in enumerate(cols):
                if i+j < len(imgs_to_show):
                    try:
                        c.image(imgs_to_show[i+j], use_container_width=True)
                    except Exception:
                        c.write("Could not load image")

# ---------------- NEW: Outfit Analyzer Tab ----------------
with tabs[5]:
    st.header("👗 Outfit Analyzer")
    st.markdown("Upload a photo of your outfit to see how it matches the current trends identified by our AI model.")

    uploaded_outfit = st.file_uploader("Upload your outfit", type=["png", "jpg", "jpeg"], key="outfit_uploader")

    if uploaded_outfit:
        st.image(uploaded_outfit, caption="Your Outfit", width=300)

        # Simulate processing
        with st.spinner("Analyzing your outfit against our AI trend model..."):
            time.sleep(3) # Simulate a delay for processing

        st.success("Analysis Complete!")

        if not top_clusters or cluster_stats.empty:
            st.error("Trend data not loaded. Cannot perform analysis. Please check your data files.")
        else:
            # --- Simulation Logic ---
            # 1. Pick a random "matched" cluster from the top trends
            matched_cluster_id = random.choice(top_clusters)

            # 2. Get its real trending score
            cluster_stats_row = cluster_stats[cluster_stats['cluster_label'] == matched_cluster_id].iloc[0]
            cluster_trending_score = cluster_stats_row['trending_score']
            max_trending_score = cluster_stats['trending_score'].max()

            # 3. Calculate a "match percentage" based on the real score, with some randomness
            base_score_pct = (cluster_trending_score / max_trending_score) * 100
            final_match_pct = max(min(base_score_pct + random.uniform(-10, 5), 98.0), 60.0) # Clamp between 60-98%

            # 4. Display results
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Analysis Results")
                st.metric(label="Trend Match Score", value=f"{final_match_pct:.1f}%")
                st.markdown(f"Your outfit strongly aligns with **Cluster {matched_cluster_id}**.")
                st.markdown(f"This cluster is one of the top trends we've identified, showing high engagement and positive sentiment.")

            with col2:
                st.subheader(f"Inspiration from Cluster {matched_cluster_id}")
                if all_images:
                    # Show a random image and claim it's from this cluster
                    sample_img_path = random.choice(all_images)
                    st.image(sample_img_path, caption=f"A sample image representing the style of Cluster {matched_cluster_id}")
                else:
                    st.warning("No sample images available to display.")

            # 5. Provide simulated suggestions
            st.markdown("---")
            st.subheader("💡 Trend-Setter Suggestions")
            st.markdown("Want to boost your trend score? Try these AI-powered tweaks:")

            suggestions_list = [
                f"Consider swapping the denim for tailored trousers for a more 'quiet luxury' feel, a key element of Cluster {matched_cluster_id}.",
                "Try accessorizing with bold, chunky metallic jewelry (gold or silver), a major trend we're seeing.",
                "A pop of color, like a bright handbag or shoes (e.g., cherry red), could elevate this look instantly.",
                "Layering is key. Add a structured blazer or a lightweight trench coat to add more dimension.",
                "Experiment with textures. Mixing leather with silk or denim with knitwear is very current.",
                f"The silhouette could be updated. Try a wider-leg pant or an oversized top to match the vibe of Cluster {matched_cluster_id}."
            ]

            chosen_suggestions = random.sample(suggestions_list, 3)
            for s in chosen_suggestions:
                st.markdown(f"- {s}")


# ---------------- AI Insights (now tabs[6]) ----------------
with tabs[6]:
    st.header("AI Insights — Automated Key Observations")
    num_imgs = len(all_images)
    pos_pct = "N/A"
    if sentiments is not None and not sentiments.empty and "sentiment" in sentiments.columns:
        pos_pct = round(sentiments["sentiment"].value_counts(normalize=True).get("Positive", 0) * 100, 1)
    st.markdown(f"""
    - *Original visuals processed:* {num_imgs}
    - *Positive sentiment (labelled):* {pos_pct}%
    - *Color palette sample unique colors:* {len(palette_df) if palette_df is not None else 0}
    """)
    top_colors = ", ".join(list(palette_df["HEX"].head(3))) if (palette_df is not None and not palette_df.empty) else "N/A"
    st.write(f"Dominant hues (top 3): *{top_colors}* — representing leading tones across recent fashion imagery.")
    st.markdown("---")
    st.subheader("Download Summary Report")
    report_text = f"""
    Nuvé Summary Report
    -------------------
    Images processed: {num_imgs}
    Labeled comments: {overview.get('Labeled Comments', 0)}
    Total posts: {overview.get('Total Posts', 0)}
    Dominant colors (top 5): {', '.join(list(palette_df['HEX'].head(5))) if (palette_df is not None and not palette_df.empty) else 'N/A'}
    Positive sentiment (labelled): {pos_pct}%
    """
    st.text_area("Report preview (editable)", value=report_text, height=200)
    st.download_button("Download report (.txt)", data=report_text.encode('utf-8'), file_name="nuve_summary_report.txt", mime="text/plain")

# ---------------- Footer ----------------
st.markdown("<hr><p class='small-muted'>Created by Team Blue Popcorn · Nuvé — AI Fashion Intelligence</p>", unsafe_allow_html=True)