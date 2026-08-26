# Forecasting Future Fashion Trends using AI

## Overview

This project aims to predict future fashion trends by analyzing Instagram posts. It uses a mix of machine learning techniques, image processing, and sentiment analysis to group similar fashion styles and assess public sentiment towards them. Also includes color palette and outfit trend analysis.

## Methodology

Our project involves several stages:

1. **Data Collection:** Data is gathered from Instagram posts using the Apify platform. The data includes the post's image, the number of likes and comments, and a sample of 20 comments per post.

2. **Image Segmentation:** The images from the posts are processed using YOLO and SAM to isolate the clothes from the background. The result is a collection of segmented images where only the clothes are visible against a black background.

3. **Feature Extraction and Dimensionality Reduction:** The segmented images are then passed through an autoencoder to extract a compact representation (latent space) of the clothing items. PCA is further applied to reduce the dimensionality of the latent space.

4. **Clustering:** The PCA-transformed latent spaces are then clustered using the K-Means algorithm. Each cluster represents a distinct fashion style.

5. **Sentiment Analysis:** The comments associated with each post are analyzed using an AdaBoost classifier to gauge public sentiment towards the fashion styles represented in the posts.

6. **Data Visualization:** A Streamlit dashboard presents statistics for each cluster, including the number of images, corresponding likes and comments, and the distribution of sentiment (positive, negative).

## Folder Structure

```
Nuve/
├── src/
│   ├── pipeline.py          # Main processing pipeline (extracted from code.ipynb)
│   └── sentiment/
│       └── model.py         # Sentiment analysis model
├── app/
│   └── streamlit_app.py     # Streamlit dashboard application
├── notebooks/
│   ├── autoencoder_training.ipynb   # Autoencoder training experiments
│   └── clustering.ipynb             # Clustering analysis
├── data/                      # Datasets (not tracked - see .gitignore)
├── images/                    # Original and segmented images (not tracked)
├── models/                    # Trained model weights (not tracked)
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── Makefile                   # Common commands
└── .gitignore                 # Git ignore rules
```

## Installation

### Prerequisites
- Python 3.10+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/hsavthegreat/Nuve.git
cd Nuve

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### External Models
- **SAM (Segment Anything Model)**: Download weights from [Meta's SAM repository](https://github.com/facebookresearch/segment-anything) and place in `models/sam/`
- **YOLO**: Weights are downloaded automatically by ultralytics on first run

## Usage

### 1. Run the Main Pipeline
```bash
# Using Makefile
make pipeline

# Or directly with Python
python -m src.pipeline
```

### 2. Train Sentiment Analysis Model
```bash
# Using Makefile
make train-sentiment

# Or directly
python -m src.sentiment.model
```

### 3. Launch Streamlit Dashboard
```bash
# Using Makefile
make serve

# Or directly
streamlit run app/streamlit_app.py
```

### 4. Run Notebooks (Exploratory)
```bash
jupyter notebook notebooks/
```

## Development

### Code Formatting
```bash
make format      # Format with black and isort
make lint        # Lint with ruff
make type-check  # Type check with mypy
```

### Testing
```bash
make test        # Run tests with pytest
```

### Clean
```bash
make clean       # Remove cache, build artifacts
```

## Technologies Used

- **Python 3.10+**: Core language
- **YOLO (ultralytics)**: Object detection for clothing localization
- **SAM (Segment Anything Model)**: Precise clothing segmentation
- **PyTorch**: Deep learning framework for autoencoder
- **scikit-learn**: PCA, K-Means, AdaBoost, preprocessing
- **NLTK**: Text tokenization and stopwords
- **Streamlit**: Interactive dashboard
- **OpenCV**: Image processing

## Project Structure Details

### `src/pipeline.py`
Main orchestration script that runs the full pipeline:
- Loads and preprocesses data
- Runs image segmentation (YOLO + SAM)
- Extracts features via autoencoder
- Performs clustering
- Saves results for dashboard

### `src/sentiment/model.py`
Sentiment analysis pipeline:
- Loads labeled comments from `data/labelled_comments_train.xlsx`
- Cleans and tokenizes text (emoji handling, stopword removal)
- Trains AdaBoost classifier with TF-IDF features
- Evaluates with classification report
- Saves model to `models/sentiment_analysis_model.joblib`
- Predicts on unlabeled comments in `data/posts_comments.csv`

### `app/streamlit_app.py`
Interactive dashboard showing:
- Cluster statistics (count, likes, comments)
- Sentiment distribution per cluster
- Sample images from each cluster
- Color palette analysis

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make test lint`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Meta AI for Segment Anything Model (SAM)
- Ultralytics for YOLO implementation
- Apify for Instagram data collection platform
