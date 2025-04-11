import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def save_trends(pattern_df, pattern_names, object_detection_type):
    SAVE_DIR = f"data/1_interim/simple_trends/{object_detection_type}"
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
        filename = f"{SAVE_DIR}/{trend_name}.parquet"
        users_df.to_parquet(filename)
        print(
            f"Saved {len(pattern_users)} users for '{pattern}' to {filename}"
        )

def detect_trend_single_type(object_detection_type: str) -> pd.DataFrame:
    print("Loading data...")
    df_merged_posts = pd.read_parquet(
        "data/1_interim/extended_data/filtered/merged_posts_extended_only_clothing.parquet"
    )
    print("Data loaded")

    pattern_df = df_merged_posts[df_merged_posts["TYPE"] == object_detection_type]

    # Group by shoe_brand
    labels = pattern_df["LABEL_NAME"].unique().tolist()

    # Filter out any empty or invalid brands
    labels = [label for label in labels if label and isinstance(label, str)]

    print(f"Found {len(labels)} unique {object_detection_type}")

    save_trends(pattern_df, labels, object_detection_type)
