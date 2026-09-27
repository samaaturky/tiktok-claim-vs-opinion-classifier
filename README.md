# TikTok Claim Classification Project

A machine learning project that classifies TikTok videos as containing a **claim** or an **opinion**, built as part of a data science course capstone activity.

## Business Context

TikTok's content moderation team needs to prioritize which videos to review for potential misinformation. Claims (factual assertions) carry more risk of spreading misinformation than opinions, so this model flags claim videos for human review, increasing response time and system efficiency in the moderation pipeline.

## Objective

Build a classifier that predicts whether a video's transcription is a **claim** or an **opinion**, using video metadata, engagement metrics, and text features — optimized for **recall**, since missing an actual claim (a false negative) is the costlier error in this context.

## Approach

1. **EDA & Cleaning** — explored class balance, distributions, and outliers in engagement metrics (likes, shares, downloads, comments)
2. **Outlier handling** — applied log transformation (`log1p`) to heavily right-skewed engagement count columns
3. **Feature engineering** — extracted `text_length` from video transcriptions; built TF-IDF features from transcription text
4. **Encoding** — binary-encoded the target variable; dummy-encoded categorical predictors
5. **Modeling** — trained and tuned Random Forest and XGBoost classifiers via `GridSearchCV` with 5-fold cross-validation, using a 60/20/20 train/validation/test split
6. **Model selection** — compared both models on the validation set; selected the best performer based on recall
7. **Evaluation** — assessed the champion model's final performance and examined feature importances

## Results

The champion model (Random Forest) achieved:
- **Recall:** ~99.9%
- **Precision:** ~99.9%

The most predictive features were engagement metrics (`video_like_count_log`, `video_share_count_log`, `video_download_count_log`, `video_comment_count_log`), followed by TF-IDF text features tied to common claim/opinion phrasing patterns.

## Repository Contents

| File | Description |
|---|---|
| `tiktok-claim-vs-opinion-classifier.ipynb` | Full analysis notebook — EDA, feature engineering, modeling, and evaluation |
| `requirements.txt` | Python dependencies needed to run the notebook |
| `.gitignore` | Excludes dataset, checkpoints, and environment files from version control |
| `tiktok_dataset.csv` | Full dataset |


## Tools Used

- Python (pandas, numpy)
- scikit-learn (Random Forest, GridSearchCV, TF-IDF, evaluation metrics)
- XGBoost
- matplotlib, seaborn


