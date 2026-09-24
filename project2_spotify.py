# ═══════════════════════════════════════════════════════════════
# Project 2: Spotify Songs' Genre Segmentation
# ═══════════════════════════════════════════════════════════════

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ── 1. Load Data ─────────────────────────────────────────────
# Expected: spotify_dataset.csv placed in the same directory as this script.
DATA_PATH = 'spotify_dataset.csv'
print("=" * 60)
print("PROJECT 2: SPOTIFY SONGS' GENRE SEGMENTATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nDataset Info:")
print(df.info())
print(f"\nStatistical Summary:")
print(df.describe().to_string())

# ── 2. Data Preprocessing ────────────────────────────────────
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# --- 2a. Missing Value Handling ---
print(f"\nMissing values before handling:")
missing_before = df.isnull().sum()
missing_mask = missing_before[missing_before > 0]
if len(missing_mask) > 0:
    print(missing_mask.to_string())
else:
    print("  (none)")

# Essential columns for recommendations must never be missing
essential_cols = ['track_name', 'track_artist']
n_before = len(df)
rows_missing_essential = df[essential_cols].isnull().any(axis=1).sum()
if rows_missing_essential > 0:
    df = df.dropna(subset=essential_cols)
    print(f"\nRemoved {rows_missing_essential} rows missing track_name or track_artist "
          f"({n_before} -> {len(df)} remaining).")
else:
    print(f"\nNo rows missing essential fields (track_name, track_artist).")

# Fill missing values in non-essential NUMERIC columns with 0 (absence of the property).
# String/ID columns (track_id, album, playlist) are left as-is — they are not features.
non_essential_cols = [c for c in df.columns if c not in essential_cols]
numeric_non_essential = df[non_essential_cols].select_dtypes(include=np.number).columns
non_ess_missing = df[numeric_non_essential].isnull().sum()
cols_with_missing = non_ess_missing[non_ess_missing > 0]
if len(cols_with_missing) > 0:
    n_filled = cols_with_missing.sum()
    df[cols_with_missing.index] = df[cols_with_missing.index].fillna(0)
    print(f"Filled {n_filled} missing values in numeric non-essential columns with 0: "
          f"{', '.join(cols_with_missing.index)}")
else:
    print("No missing values in numeric non-essential columns.")

# --- 2b. Duplicates ---
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Track ID analysis: same track_id in multiple playlists is legitimate,
# not an error. Report it for transparency.
n_total_rows = len(df)
n_unique_track_ids = df['track_id'].nunique()
n_total_track_ids = df['track_id'].count()
n_repeated_track_ids = n_total_track_ids - n_unique_track_ids
n_track_ids_multi = (df['track_id'].value_counts() > 1).sum()
print(f"\nTrack ID analysis (same song in multiple playlists is legitimate):")
print(f"  Total rows: {n_total_rows}")
print(f"  Unique track IDs: {n_unique_track_ids}")
print(f"  Track IDs appearing more than once: {n_track_ids_multi}")
print(f"  Repeated track_id occurrences: {n_repeated_track_ids} "
      f"({n_repeated_track_ids/n_total_rows*100:.1f}% of rows)")

# --- 2c. Outlier Review (informational — no deletions) ---
# Justification:
# - danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence:
#   Spotify-defined features on [0, 1]. Values outside this range are data errors, not
#   legitimate musical variation. Clipped below.
# - key: integer 0-11. No cleaning needed.
# - loudness: dB scale, roughly -60 to 0 for music. Extreme values (>5 or <-70) are
#   likely errors. We flag them but do NOT delete — let the model handle them.
# - tempo: BPM. Genres span wide ranges (ambient ~20 to drum n bass ~300+). IQR outliers
#   are not automatically errors. We flag but do NOT delete legitimate songs.
# - duration_ms: track length. Very long tracks (>600s) are rare but legitimate (classical).
#   We do NOT delete. IQR-outlier songs are kept — statistical outlier != data error.
print(f"\nOutlier review (IQR method — informational only, no deletions):")
cluster_features_list = ['danceability', 'energy', 'loudness', 'speechiness',
                         'acousticness', 'instrumentalness', 'liveness',
                         'valence', 'tempo', 'duration_ms']
for col in cluster_features_list:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo = Q1 - 1.5 * IQR
    hi = Q3 + 1.5 * IQR
    outliers = ((df[col] < lo) | (df[col] > hi)).sum()
    if outliers > 0:
        print(f"  {col:<18} {outliers:>6} IQR-outliers ({outliers/len(df)*100:.1f}%) — kept (not deleted)")

# --- 2d. Feature Clipping (justified above) ---
# Clip Spotify-defined [0,1] features to valid range
bounded_features = ['danceability', 'energy', 'speechiness', 'acousticness',
                    'instrumentalness', 'liveness', 'valence']
n_clipped_total = 0
for col in bounded_features:
    before = df[col].copy()
    df[col] = df[col].clip(0, 1)
    clipped = (before != df[col]).sum()
    n_clipped_total += clipped
    if clipped > 0:
        print(f"  Clipped {col}: {clipped} values outside [0,1] -> boundary")
print(f"\nTotal values clipped to [0,1] bounds: {n_clipped_total}")

# Flag extreme loudness/tempo (kept, not deleted)
extreme_loud = ((df['loudness'] > 5) | (df['loudness'] < -70)).sum()
extreme_bpm = ((df['tempo'] > 300) | (df['tempo'] < 10)).sum()
if extreme_loud > 0:
    print(f"  Loudness extremes (>5 or <-70 dB): {extreme_loud} rows — kept")
if extreme_bpm > 0:
    print(f"  Tempo extremes (>300 or <10 BPM): {extreme_bpm} rows — kept")

print(f"\nPreprocessing complete. Final dataset: {len(df)} songs, {len(df.columns)} columns.")
print(f"Missing values remaining: {df.isnull().sum().sum()}")

# ── 3. Data Visualization ────────────────────────────────────
print("\n" + "=" * 60)
print("DATA VISUALIZATION")
print("=" * 60)

# Ensure results directory exists before saving plots
os.makedirs('results', exist_ok=True)

# --- 3a. Playlist Genre Distribution ---
fig, ax = plt.subplots(figsize=(12, 6))
genre_counts = df['playlist_genre'].value_counts()
colors = sns.color_palette('viridis', len(genre_counts))
bars = ax.bar(genre_counts.index, genre_counts.values, color=colors)
ax.set_title('Distribution of Playlist Genres', fontweight='bold', fontsize=14)
ax.set_xlabel('Playlist Genre')
ax.set_ylabel('Number of Songs')
ax.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, genre_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
        str(val), ha='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('results/project2_genre_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: project2_genre_distribution.png")

# --- 3b. Playlist Subgenre Distribution (Top 15) ---
fig, ax = plt.subplots(figsize=(14, 6))
subgenre_counts = df['playlist_subgenre'].value_counts().head(15)
colors = sns.color_palette('plasma', len(subgenre_counts))
bars = ax.barh(range(len(subgenre_counts)), subgenre_counts.values, color=colors)
ax.set_yticks(range(len(subgenre_counts)))
ax.set_yticklabels(subgenre_counts.index)
ax.set_title('Top 15 Playlist Subgenres', fontweight='bold', fontsize=14)
ax.set_xlabel('Number of Songs')
ax.invert_yaxis()
for bar, val in zip(bars, subgenre_counts.values):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2.,
        str(val), va='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('results/project2_subgenre_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: project2_subgenre_distribution.png")

# --- 3c. Playlist Name Distribution (Top 15) ---
fig, ax = plt.subplots(figsize=(12, 6))
playlist_counts = df['playlist_name'].value_counts().head(15)
colors = sns.color_palette('Set2', len(playlist_counts))
bars = ax.barh(range(len(playlist_counts)), playlist_counts.values, color=colors)
ax.set_yticks(range(len(playlist_counts)))
ax.set_yticklabels(playlist_counts.index)
ax.set_title('Top 15 Playlists by Song Count', fontweight='bold', fontsize=14)
ax.set_xlabel('Number of Songs')
ax.invert_yaxis()
for bar, val in zip(bars, playlist_counts.values):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2.,
        str(val), va='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('results/project2_playlist_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: project2_playlist_distribution.png")

# --- 3d. Popularity vs Duration ---
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    df['duration_ms'] / 1000, df['track_popularity'],
    c=df['energy'], cmap='YlOrRd', alpha=0.5, s=15
)
ax.set_title('Track Popularity vs Duration (colored by Energy)', fontweight='bold')
ax.set_xlabel('Duration (seconds)')
ax.set_ylabel('Popularity')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Energy')
plt.tight_layout()
plt.savefig('results/project2_popularity_duration.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: project2_popularity_duration.png")

# --- 3e. Danceability vs Energy by Genre ---
fig, ax = plt.subplots(figsize=(12, 7))
unique_genres = df['playlist_genre'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_genres)))
for genre, color in zip(unique_genres, colors):
    subset = df[df['playlist_genre'] == genre]
    ax.scatter(
        subset['danceability'], subset['energy'],
        alpha=0.3, s=10, label=genre, color=color
    )
ax.set_title('Danceability vs Energy by Playlist Genre', fontweight='bold')
ax.set_xlabel('Danceability')
ax.set_ylabel('Energy')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('results/project2_danceability_energy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: project2_danceability_energy.png")

# --- 3f. Audio Feature Boxplots by Genre ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
feat_to_plot = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'tempo']
for ax, feat in zip(axes.flatten(), feat_to_plot):
    df.boxplot(column=feat, by='playlist_genre', ax=ax, rot=45, fontsize=8)
    ax.set_title(f'{feat.capitalize()} by Genre', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(feat.capitalize())
plt.suptitle('')
plt.tight_layout()
plt.savefig('results/project2_audio_features_by_genre.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: project2_audio_features_by_genre.png")

# --- 3g. Correlation Heatmap (Audio Features) ---
audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode',
                  'speechiness', 'acousticness', 'instrumentalness',
                  'liveness', 'valence', 'tempo', 'duration_ms', 'track_popularity']
fig, ax = plt.subplots(figsize=(12, 10))
corr = df[audio_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
    center=0, square=True, linewidths=0.5, ax=ax,
    cbar_kws={"shrink": 0.8})
ax.set_title('Correlation Matrix - Spotify Audio Features', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('results/project2_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: project2_correlation_heatmap.png")

# --- 3h. Pairwise Scatter: Key Audio Features ---
pair_cols = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness']
pair_df = df[pair_cols].copy()
pair_df['playlist_genre'] = df['playlist_genre']
g = sns.pairplot(pair_df, hue='playlist_genre', diag_kind='kde',
    plot_kws={'alpha': 0.3, 's': 10}, corner=True)
g.fig.suptitle('Pairwise Audio Feature Analysis by Genre', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/project2_pairwise_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: project2_pairwise_scatter.png")

# ── 4. Clustering Analysis ───────────────────────────────────
print("\n" + "=" * 60)
print("CLUSTERING ANALYSIS")
print("=" * 60)

# Prepare features for clustering
cluster_features = ['danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness',
                    'valence', 'tempo', 'duration_ms']

X_cluster = df[cluster_features].copy()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# --- 4a. Elbow Method for K-Means ---
print("\nRunning K-Means Elbow Method...")
inertias = []
sil_scores = []
K_range = range(2, 12)
# Use a fixed 5,000-row sample for silhouette to keep runtime reasonable
# while keeping KMeans training on the full dataset for accuracy.
sil_sample_idx = np.random.RandomState(42).choice(len(X_scaled), size=5000, replace=False)
X_sil_sample = X_scaled[sil_sample_idx]
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_sil_sample, km.labels_[sil_sample_idx]))
print("(silhouette scores computed on a fixed 5,000-song sample for speed)")

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8, label='Inertia')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Within-cluster SSE)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(K_range)

ax2 = ax1.twinx()
ax2.plot(K_range, sil_scores, 'rs-', linewidth=2, markersize=8, label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='red')
ax2.tick_params(axis='y', labelcolor='red')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('K-Means: Elbow Method & Silhouette Score', fontweight='bold')
plt.tight_layout()
plt.savefig('results/project2_elbow_method.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: project2_elbow_method.png")

best_k = K_range[np.argmax(sil_scores)]
print(f"\nBest k by silhouette score: {best_k} (score: {max(sil_scores):.4f})")

# --- 4b. K-Means Clustering (best k) ---
print(f"\nRunning K-Means with k={best_k}...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nK-Means Cluster Distribution:")
print(df['kmeans_cluster'].value_counts().sort_index())

# --- 4c. PCA Visualization of Clusters ---
print("\nRunning PCA for visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    df['pca1'], df['pca2'],
    c=df['kmeans_cluster'], cmap='tab10', alpha=0.5, s=10
)
ax.set_title(f'K-Means Clustering (k={best_k}) - PCA Projection', fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Cluster')
plt.tight_layout()
plt.savefig('results/project2_kmeans_pca.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: project2_kmeans_pca.png")

# --- 4d. Cluster Characteristics (raw means, printed only) ---
print(f"\nCluster Characteristics (raw mean values):")
cluster_summary_raw = df.groupby('kmeans_cluster')[cluster_features].mean()
print(cluster_summary_raw.round(3).to_string())

# --- 4d-bis. Cluster Audio Feature Profiles (STANDARDIZED for plotting) ---
# Standardize each feature to z-scores so duration_ms no longer dominates the chart.
# The y-axis shows mean z-score per cluster — comparable across all features.
print("\nPlotting standardized cluster profiles (z-score means)...")
cluster_scaler = StandardScaler()
cluster_features_scaled = cluster_scaler.fit_transform(df[cluster_features])
df_cluster_std = pd.DataFrame(
    cluster_features_scaled, columns=cluster_features, index=df.index
)
cluster_std_summary = df_cluster_std.groupby(df['kmeans_cluster']).mean()

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(cluster_features))
width = 0.8 / best_k
for i in range(best_k):
    offset = (i - best_k/2 + 0.5) * width
    means = cluster_std_summary.loc[i].values
    ax.bar(x + offset, means, width, label=f'Cluster {i}', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(cluster_features, rotation=45, ha='right')
ax.set_title('Cluster Audio Feature Profiles (Standardized — Z-Score Means)', fontweight='bold')
ax.set_ylabel('Mean Z-Score (standardized feature values)')
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/project2_cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: project2_cluster_profiles.png")

# ── 5. Recommendation System Prototype ───────────────────────
# Runs before hierarchical clustering so the recommendation results file is
# generated even if hierarchical clustering fails on some machine.
print("\n" + "=" * 60)
print("RECOMMENDATION SYSTEM PROTOTYPE")
print("=" * 60)

# Use StandardScaler-normalized audio features for fair cosine similarity.
# duration_ms, tempo, and loudness have very different scales from the [0,1]
# features — without scaling, raw cosine similarity over-weights large-scale features.
rec_features = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness',
                'valence', 'tempo', 'duration_ms']

rec_scaler = StandardScaler()
rec_features_scaled = rec_scaler.fit_transform(df[rec_features].values)

def recommend_songs(song_query, n=5, out=None):
    """Recommend similar songs using scaled cosine similarity (on-demand, no full matrix).

    Lookup logic:
      1. Exact case-insensitive track_name match preferred.
      2. If multiple, use artist info when available.
      3. If still ambiguous, report it clearly instead of silently picking one.
      4. Falls back to substring match if no exact match.
    """
    name_lower = df['track_name'].fillna('').str.lower()
    q = song_query.lower().strip()

    exact_mask = name_lower == q
    n_exact = exact_mask.sum()

    if n_exact == 0:
        sub_mask = df['track_name'].fillna('').str.contains(q, case=False, na=False)
        matches = df[sub_mask]
        mode = "substring"
        n_matches = len(matches)
    else:
        matches = df[exact_mask]
        mode = "exact"
        n_matches = n_exact

    if n_matches == 0:
        msg = f"  Song '{song_query}' not found (no exact or substring match)."
        print(msg)
        if out:
            out.write(msg + "\n")
        return None, None

    # Resolve ambiguity
    if n_matches > 1:
        unique_titles = matches['track_name'].dropna().unique()
        unique_artists = matches['track_artist'].dropna().unique()
        if mode == "exact" and len(unique_titles) == 1 and len(unique_artists) > 1:
            msg = (f"  Found {n_matches} exact matches for '{song_query}' by "
                   f"{', '.join(unique_artists)} (across {n_matches} playlists).")
            print(msg)
            if out:
                out.write(msg + "\n")
            print(f"  Using first entry. Specify artist for precise lookup.")
        elif mode == "exact" and len(unique_titles) > 1:
            msg = f"  Found {n_matches} exact-name matches for '{song_query}':"
            print(msg)
            if out:
                out.write(msg + "\n")
            for _, r in matches.drop_duplicates(['track_name', 'track_artist']).iterrows():
                detail = f"    - '{r['track_name']}' by {r['track_artist']}"
                print(detail)
                if out:
                    out.write(detail + "\n")
            print(f"  Using first match. Refine query with artist name for precision.")
        elif mode == "substring":
            msg = f"  Found {n_matches} substring matches for '{song_query}'. Using first match."
            print(msg)
            if out:
                out.write(msg + "\n")
        else:
            msg = f"  Found {n_matches} match(es) for '{song_query}'. Using first."
            print(msg)
            if out:
                out.write(msg + "\n")

    match_idx = matches.index[0]
    pos = df.index.get_loc(match_idx)

    # On-demand cosine similarity: 1×N, not N×N
    query_vec = rec_features_scaled[pos].reshape(1, -1)
    sims = cosine_similarity(query_vec, rec_features_scaled).flatten()

    match_name = df.iloc[pos]['track_name']
    match_artist = df.iloc[pos]['track_artist']
    match_key = (match_name, match_artist)

    top = []
    seen = set()
    for j in np.argsort(sims)[::-1]:
        if j == pos:
            continue
        row = df.iloc[j]
        key = (row['track_name'], row['track_artist'])
        if key == match_key:
            continue
        if key in seen:
            continue
        seen.add(key)
        if len(top) >= n:
            break
        top.append((j, float(sims[j])))

    header = (f"\n  Recommendations similar to '{match_name}' by {match_artist} "
              f"({mode} match, {n_matches} entry/ies):")
    print(header)
    if out:
        out.write(header + "\n")
    if not top:
        print("    (no distinct similar songs found)")
        if out:
            out.write("    (no distinct similar songs found)\n")
        return None, pos
    for i, (j, sim) in enumerate(top):
        line = (f"    {i+1}. {df.iloc[j]['track_name']} - {df.iloc[j]['track_artist']} "
                f"(Similarity: {sim:.4f}, Genre: {df.iloc[j]['playlist_genre']})")
        print(line)
        if out:
            out.write(line + "\n")
    return top, pos

# --- Demo + save results to file ---
print("\nDemo recommendations:")
with open('project2_recommendation_results.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("SPOTIFY SONG RECOMMENDATION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Method: Cosine similarity on StandardScaler-normalized audio features\n")
    f.write(f"Features: {', '.join(rec_features)}\n")
    f.write(f"Dataset: {len(df)} songs (on-demand 1xN similarity, no full NxN matrix)\n\n")

    for song_label in ["Blinding Lights", "Levitating"]:
        f.write("-" * 60 + "\n")
        f.write(f"INPUT SONG: {song_label}\n")
        f.write("-" * 40 + "\n")
        # Run recommendation WITHOUT writing to the file here — we write the
        # formatted output manually below so each example appears exactly once.
        results, pos = recommend_songs(song_label, n=5, out=None)
        if pos is not None:
            f.write(f"\nInput Song: {df.iloc[pos]['track_name']}\n")
            f.write(f"Input Artist: {df.iloc[pos]['track_artist']}\n")
        if results:
            f.write("Recommendations:\n")
            for i, (j, sim) in enumerate(results):
                line = (f"  {i+1}. {df.iloc[j]['track_name']} - {df.iloc[j]['track_artist']} "
                        f"(Similarity: {sim:.4f}, Genre: {df.iloc[j]['playlist_genre']})")
                f.write(line + "\n")
        f.write("\n")

print("\nSaved: project2_recommendation_results.txt")

# ── 4e. Hierarchical Clustering (supplementary, on a fixed sample) ──
# Hierarchical clustering is NOT a required part of the assignment — K-Means
# is the main method and runs on the full dataset above.  Hierarchical clustering
# is expensive (Ward linkage builds a large pairwise structure), so here it is
# run ONLY on a deterministic 5,000-song sample as a supplementary comparison.
# The ARI is computed on that same sample and does NOT imply full-dataset labels.
print("\nRunning Hierarchical Clustering on a fixed 5,000-song sample...")
from sklearn.neighbors import kneighbors_graph

sample_size = min(5000, len(X_scaled))
sample_idx = np.random.RandomState(42).choice(len(X_scaled), size=sample_size, replace=False)
X_hier_sample = X_scaled[sample_idx]

connectivity = kneighbors_graph(X_hier_sample, n_neighbors=10, include_self=False)
agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward', connectivity=connectivity)
hierarchical_sample_labels = agg.fit_predict(X_hier_sample)

kmeans_sample_labels = df['kmeans_cluster'].to_numpy()[sample_idx]
ari = adjusted_rand_score(kmeans_sample_labels, hierarchical_sample_labels)

print(f"\nHierarchical Clustering (sample, n={sample_size}):")
print(f"  Cluster 0: {(hierarchical_sample_labels == 0).sum()} songs")
print(f"  Cluster 1: {(hierarchical_sample_labels == 1).sum()} songs")
print(f"\nAdjusted Rand Index (K-Means vs Hierarchical, sample): {ari:.4f}")

# --- 4f. Clusters by Playlist Genre ---
# Left: K-Means clusters (full dataset — the main method).
# Right: same K-Means clusters, repeated for a clean 1x2 layout.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ct1 = pd.crosstab(df['kmeans_cluster'], df['playlist_genre'])
ct1_pct = ct1.div(ct1.sum(axis=1), axis=0) * 100
ct1_pct.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab10')
axes[0].set_title(f'K-Means Clusters vs Genre (full dataset)', fontweight='bold')
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Percentage')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[0].tick_params(axis='x', rotation=0)

# Right panel: same K-Means data, presented alongside the left panel for a
# balanced 1x2 figure. Hierarchical clustering is sample-only (see 4e) and
# does not produce full-dataset labels, so it is not shown here.
ct2 = pd.crosstab(df['kmeans_cluster'], df['playlist_genre'])
ct2_pct = ct2.div(ct2.sum(axis=1), axis=0) * 100
ct2_pct.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab10')
axes[1].set_title(f'K-Means Clusters vs Genre (full dataset)', fontweight='bold')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Percentage')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('results/project2_clusters_vs_genre.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: project2_clusters_vs_genre.png")

# --- 4g. Top Playlists per K-Means Cluster (grouped horizontal bars) ---
# Each cluster's top playlists are shown in a separate group so bars never
# overlap. Long playlist names remain readable on the y-axis.
top_clusters_kmeans = df['kmeans_cluster'].value_counts().head(5).index
PLAYLISTS_PER_CLUSTER = 5

# Collect top playlists per cluster
cluster_playlist_data = {}
for cluster in sorted(top_clusters_kmeans):
    subset = (df[df['kmeans_cluster'] == cluster]
              .groupby('playlist_name').size()
              .sort_values(ascending=False)
              .head(PLAYLISTS_PER_CLUSTER))
    cluster_playlist_data[cluster] = subset

# Layout: one row per playlist per cluster, grouped by cluster.
# Total rows = number_of_clusters * PLAYLISTS_PER_CLUSTER.
n_clusters = len(cluster_playlist_data)
total_rows = n_clusters * PLAYLISTS_PER_CLUSTER
y_positions = np.arange(total_rows)

fig, ax = plt.subplots(figsize=(14, max(6, total_rows * 0.45)))

cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
group_height = PLAYLISTS_PER_CLUSTER  # rows per cluster group

for gi, cluster in enumerate(sorted(top_clusters_kmeans)):
    playlist_series = cluster_playlist_data[cluster]
    start_y = gi * group_height
    mid_y = start_y + (group_height - 1) / 2
    # y positions: top playlist gets the top row in this group
    y_vals = np.arange(len(playlist_series))[::-1] + start_y

    for yi, (pl_name, count) in zip(y_vals, playlist_series.items()):
        ax.barh(yi, count, color=cluster_colors[gi], alpha=0.85,
                label=f'Cluster {cluster}' if yi == y_vals[0] else '')

    # Group label in the left margin.
    # NOTE: fig.transFigure uses bottom-to-top y, but the axes have
    # invert_yaxis() enabled (y=0 at top).  Flip the y position so the
    # label for the top group lands near the top of the figure.
    fig_y = 1.0 - (mid_y + 0.5) / total_rows
    fig.text(0.005, fig_y,
             f'Cluster {cluster}',
             ha='left', va='center', fontsize=10, fontweight='bold',
             color=cluster_colors[gi],
             transform=fig.transFigure)

# Build y-tick labels in the SAME order as the bars: top to bottom.
# y_vals for each group are already reversed (top playlist at top),
# so we collect labels by iterating groups top-to-bottom and within
# each group from the highest y_val down to the lowest.
ytick_labels = []
for gi, cluster in enumerate(sorted(top_clusters_kmeans)):
    playlist_series = cluster_playlist_data[cluster]
    start_y = gi * group_height
    mid_y = start_y + (group_height - 1) / 2
    # playlist_series is sorted descending by count (head(5)).
    # y_vals reverses this: top count -> top y position.
    # To match bars top-to-bottom, iterate playlist_series in reverse.
    for pl_name in reversed(playlist_series.index):
        ytick_labels.append(str(pl_name))

ax.set_yticks(y_positions)
ax.set_yticklabels(ytick_labels, fontsize=8)

ax.set_title('Top Playlists per K-Means Cluster', fontweight='bold')
ax.set_xlabel('Number of Songs')
ax.invert_yaxis()
ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
ax.set_xlim(left=0)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/project2_playlists_per_cluster.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: project2_playlists_per_cluster.png")

# ── 6. Summary ───────────────────────────────────────────────
print(f"\n{'='*60}")
print("PROJECT 2 SUMMARY")
print(f"{'='*60}")
print(f"\nDataset: {len(df)} songs, {len(df.columns)} features")
print(f"Playlist genres: {df['playlist_genre'].nunique()}")
print(f"Playlist subgenres: {df['playlist_subgenre'].nunique()}")
print(f"Unique playlists: {df['playlist_name'].nunique()}")
print(f"\nBest clustering (K-Means): k={best_k}")
print(f"Silhouette score: {max(sil_scores):.4f}")
print(f"\nK-Means cluster distribution:")
for cluster, count in df['kmeans_cluster'].value_counts().sort_index().items():
    print(f"  Cluster {cluster}: {count} songs ({count/len(df)*100:.1f}%)")
print(f"\nAll visualizations saved as PNG files in results/")
print(f"Recommendation results saved to project2_recommendation_results.txt")
print(f"Plots generated: 13")
