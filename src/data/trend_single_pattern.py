import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def save_trends(pattern_df, pattern_names, period):
    SAVE_DIR = "data/1_interim/simple_trends/pattern"
    os.makedirs(SAVE_DIR, exist_ok=True)
    for pattern in pattern_names:
        pattern_users = pattern_df[pattern_df["LABEL_NAME"] == pattern][
            "AUTHORID"
        ].unique()
        trend_name = f"{pattern}"
        users_df = pd.DataFrame(
            {
                "AUTHORID": pattern_users,
                "trend_name": trend_name,
            }
        )

        # Save to file
        filename = f"{SAVE_DIR}/{period}_{trend_name}.parquet"
        users_df.to_parquet(filename)
        print(
            f"Saved {len(pattern_users)} users for winter pattern '{pattern}' to {filename}"
        )

def detect_trend_single_pattern() -> pd.DataFrame:
    period1 = ["2023-06", "2023-07", "2023-08"]
    period2 = ["2023-11", "2023-12", "2024-01"]

    print("Loading data...")
    df_merged_posts = pd.read_parquet("data/1_interim/extended_data/merged_posts_extended.parquet")
    print("Data loaded")
    
    pattern_df = df_merged_posts[df_merged_posts["TYPE"] == "pattern"]
    
	# Date processing
    if not pd.api.types.is_datetime64_any_dtype(pattern_df["POST_PUBLICATION_DATE"]):
        pattern_df.loc[:, "POST_PUBLICATION_DATE"] = pd.to_datetime(
			pattern_df["POST_PUBLICATION_DATE"]
		)
    pattern_df.loc[:, "month_year"] = pattern_df["POST_PUBLICATION_DATE"].dt.to_period("M")
    
	# Count the patterns
    pattern_counts = (
        pattern_df.groupby(["month_year", "LABEL_NAME"]).size().unstack(fill_value=0)
	)
    normalized_counts = pattern_counts.copy()
    for column in normalized_counts.columns:
        scaler = MinMaxScaler()
        normalized_counts[column] = scaler.fit_transform(normalized_counts[[column]])

	# Calculate the mean for each period
    period1_mean = normalized_counts.loc[period1].mean()
    period2_mean = normalized_counts.loc[period2].mean()
    pattern_diff = period2_mean - period1_mean
    pattern_diff_sorted = pattern_diff.sort_values()


    TREND_THRESHOLD = 0.3
    winter_patterns = pattern_diff_sorted[pattern_diff_sorted > TREND_THRESHOLD].index.tolist()
    summer_patterns = pattern_diff_sorted[pattern_diff_sorted < -TREND_THRESHOLD].index.tolist()

    save_trends(pattern_df, winter_patterns, "23W")
    save_trends(pattern_df, summer_patterns, "23S")
