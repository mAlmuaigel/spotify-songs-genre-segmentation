# Spotify Songs' Genre Segmentation

Machine learning project for analyzing Spotify song data using clustering and building a song recommendation prototype based on audio feature similarity.

## Overview

This project analyzes a Spotify dataset of songs using data preprocessing, exploratory data analysis, visualization, correlation analysis, clustering, and a song recommendation system. The goal is to explore how songs group together based on their audio features and to build a simple content-based recommendation prototype.

## Dataset

The project uses `spotify_dataset.csv`, which contains 32,833 original records. Five rows with missing essential song metadata (track_name or track_artist) were removed, leaving 32,828 songs for the analysis.

The dataset includes song metadata (track name, artist, album, release date, popularity) and Spotify audio features (danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms), along with playlist information (playlist name, playlist genre, playlist subgenre).

## Data Preprocessing

The script performs the following preprocessing steps:

- **Missing value handling:** Rows missing essential fields (track_name, track_artist) are removed. Missing values in non-essential numeric audio feature columns are filled with 0.
- **Duplicate checking:** Duplicate rows are identified and reported. Track IDs that appear in multiple playlists are also reported separately — this is legitimate (the same song can belong to multiple playlists) and is not treated as an error.
- **Feature clipping:** Spotify-defined features bounded to [0, 1] (danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence) are clipped to valid range. Extreme loudness and tempo values are flagged but kept.
- **Outlier review:** IQR-based outlier counts are reported for all clustering features, but no rows are deleted based on outliers alone.
- **Feature scaling:** StandardScaler is applied to audio features for clustering and recommendation.

## Exploratory Data Analysis

The project generates visualizations covering:

- **Playlist genre distribution** — count of songs per playlist genre
- **Playlist subgenre distribution** — top 15 playlist subgenres
- **Playlist distribution** — top 15 playlists by song count
- **Popularity vs duration** — scatter plot colored by energy
- **Danceability vs energy** — scatter by playlist genre
- **Audio features by genre** — boxplots of danceability, energy, valence, acousticness, speechiness, and tempo across genres
- **Correlation matrix** — pairwise correlations among audio features
- **Pairwise scatter** — pairwise relationships among key audio features colored by genre

## Clustering

The clustering analysis uses the following audio features: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, and duration_ms.

The process:

1. Standardizes the audio features using StandardScaler.
2. Evaluates K-Means for k from 2 to 11 using inertia (elbow method) and silhouette score (computed on a fixed 5,000-song sample for speed, while KMeans is trained on the full dataset).
3. Selects k=2 because it had the highest silhouette score among the tested values.
4. Runs K-Means with k=2 and visualizes the result with PCA (2 components).
5. Runs hierarchical clustering (AgglomerativeClustering, ward linkage) as a supplementary comparison on a fixed 5,000-song sample — K-Means is the main method and runs on the full dataset. The adjusted Rand index is computed on that same sample.
6. Analyzes cluster composition by playlist genre.
7. Profiles cluster audio-feature means (standardized z-scores for comparability across features).

Clustering groups songs based on similarities in the selected audio features. It does not discover objectively correct music genres — the clusters represent groupings derived from the audio feature space.

**Best K-Means configuration:** k = 2  
**Silhouette score:** approximately 0.177

The silhouette score is modest, indicating that the clusters have some internal cohesion but also significant overlap. This is expected given that Spotify audio features span a continuum and do not map cleanly to discrete genre boundaries.

## Recommendation System

The project includes a song recommendation prototype that finds songs similar to a given query song based on audio feature similarity.

The method:

- Selects the same 10 audio features used for clustering.
- Standardizes them with StandardScaler so that features with larger numeric ranges (duration_ms, tempo, loudness) do not dominate the similarity calculation.
- Computes cosine similarity directly (on-demand for each query, not a pre-built full matrix) to find the most similar songs.

The lookup logic prefers exact case-insensitive track-name matches. If multiple exact matches exist, it reports them and uses the first. If no exact match is found, it falls back to substring matching. Ambiguity is reported rather than silently resolved.

The system recommends songs based on similarity in the selected audio features. It does not use lyrics, musical semantics, artist preferences, or any content beyond the provided audio features.

## Results

**Clustering:**

- Best K-Means k: 2
- Silhouette score: ~0.177
- PCA visualization shows the two clusters projected onto the first two principal components
- Hierarchical clustering is a supplementary comparison performed on a fixed 5,000-song sample (K-Means is the main method and runs on the full dataset). It uses a sparse k-NN connectivity graph (k=10) to avoid the full O(n²) distance matrix, and the adjusted Rand index is computed on that same sample.
- Cluster composition by playlist genre is analyzed

**Recommendation examples:**

The recommendation results file contains example recommendations for "Blinding Lights" (The Weeknd) and "Levitating" (invention_), each with 5 similar songs, similarity scores, and genres.

## Project Files

- `project2_spotify.py` — Main analysis, clustering, visualization, and recommendation script.
- `spotify_dataset.csv` — Spotify dataset used by the project.
- `project2_recommendation_results.txt` — Example recommendation output generated by the project.
- `results/` — Contains the generated visualization PNG files.

## How to Run

1. Install Python 3.
2. Open the project folder.
3. Install dependencies:

```
pip install -r requirements.txt
```

4. Make sure `spotify_dataset.csv` is in the same directory as `project2_spotify.py`.
5. Run:

```
python project2_spotify.py
```

The script generates the visualization PNG files inside the `results/` folder and writes recommendation results to `project2_recommendation_results.txt`.
