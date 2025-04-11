import pandas as pd
import os
import json

def add_post_feature_is_collab(df_images_of_posts: pd.DataFrame) -> pd.DataFrame:
    print("Adding is_collab feature")
    df_images_of_posts['IS_COLLAB'] = df_images_of_posts["POST_CAPTION"].str.contains(
        "sponsor|commercial partner", 
        case=False,
        na=False,
    ).astype(int)
    
    print("is_collab feature added")
    return df_images_of_posts

def add_author_feature_from_author_aggregate(
        df_images_of_posts: pd.DataFrame, 
        df_merged_authors: pd.DataFrame
    ) -> pd.DataFrame:
    print("Adding authorid_stats_features")

    author_post_stats = df_images_of_posts.groupby('AUTHORID').agg(
        total_posts=('POST_ID', 'count'),
        mean_likes=('NB_LIKES', 'mean'),
        median_likes=('NB_LIKES', 'median'),
        max_likes=('NB_LIKES', 'max'),
        mean_comments=('COMMENT_COUNT', 'mean'),
        median_comments=('COMMENT_COUNT', 'median'),
        max_comments=('COMMENT_COUNT', 'max'),
        total_collabs=('IS_COLLAB', 'sum')
    ).reset_index()

    df_merged_authors = pd.merge(
        df_merged_authors, author_post_stats, 
        on="AUTHORID", how="left"
    )

    print("authorid_stats_features added")
    return df_merged_authors

def add_author_feature_from_post_aggregate(
        df_images_of_posts: pd.DataFrame, 
        df_merged_authors: pd.DataFrame
    ) -> pd.DataFrame:
    print("Adding post proportion features")
    
    # Calculate the total number of unique images per author
    author_unique_images = df_images_of_posts.groupby('AUTHORID')['IMAGE_ID'].nunique().reset_index()
    author_unique_images.rename(columns={'IMAGE_ID': 'total_unique_images'}, inplace=True)
    
    # Calculate unique images with collabs per author
    collab_images = df_images_of_posts[df_images_of_posts['IS_COLLAB'] == 1]
    collab_unique_per_author = collab_images.groupby('AUTHORID')['IMAGE_ID'].nunique().reset_index()
    collab_unique_per_author.rename(columns={'IMAGE_ID': 'collab_unique_images'}, inplace=True)
    
    # Calculate unique images with sport items per author
    sport_images = df_images_of_posts[df_images_of_posts['HAS_SPORT_ITEM'] == 1]
    sport_unique_per_author = sport_images.groupby('AUTHORID')['IMAGE_ID'].nunique().reset_index()
    sport_unique_per_author.rename(columns={'IMAGE_ID': 'sport_unique_images'}, inplace=True)
    
    # Merge the counts with author data
    df_merged_authors = pd.merge(df_merged_authors, author_unique_images, on='AUTHORID', how='left')
    df_merged_authors = pd.merge(df_merged_authors, collab_unique_per_author, on='AUTHORID', how='left')
    df_merged_authors = pd.merge(df_merged_authors, sport_unique_per_author, on='AUTHORID', how='left')
    
    # Fill NaN values with 0 (authors with no collabs or sport items)
    df_merged_authors['collab_unique_images'] = df_merged_authors['collab_unique_images'].fillna(0)
    df_merged_authors['sport_unique_images'] = df_merged_authors['sport_unique_images'].fillna(0)
    
    # Calculate proportions
    df_merged_authors['PROP_COLLAB_POSTS'] = df_merged_authors['collab_unique_images'] / df_merged_authors['total_unique_images']
    df_merged_authors['PROP_SPORT_POSTS'] = df_merged_authors['sport_unique_images'] / df_merged_authors['total_unique_images']
    
    # Handle division by zero
    df_merged_authors['PROP_COLLAB_POSTS'] = df_merged_authors['PROP_COLLAB_POSTS'].fillna(0)
    df_merged_authors['PROP_SPORT_POSTS'] = df_merged_authors['PROP_SPORT_POSTS'].fillna(0)
    
    # Drop intermediate columns
    df_merged_authors.drop(['collab_unique_images', 'sport_unique_images', 'total_unique_images'], axis=1, inplace=True)
    
    print("Post proportion features added")
    return df_merged_authors


def add_post_feature_has_sport_item(df_merged_posts: pd.DataFrame) -> pd.DataFrame:
    print("Adding has_sport_item feature")
    with open("data/stored/unique_labels_by_type.json", "r", encoding="utf-8") as f:
        unique_labels_by_type = json.load(f)

    sport_labels = unique_labels_by_type["object_detection_sport"]

    mask_sport_items = ~(
        (df_merged_posts["TYPE"] == "object_detection")
        & (~df_merged_posts["LABEL_NAME"].isin(sport_labels))
    )

    sport_images = df_merged_posts[mask_sport_items]["IMAGE_ID"].unique()
    df_merged_posts["HAS_SPORT_ITEM"] = (
        df_merged_posts["IMAGE_ID"].isin(sport_images).astype(int)
    )

    print("has_sport_item feature added")
    return df_merged_posts

def add_author_feature_base_segmentation(df_merged_authors: pd.DataFrame) -> pd.DataFrame:
    print("Adding base_segmentation feature")
    mask_mainstream = df_merged_authors["NB_FOLLOWERS"] <= 12000
    mask_trendy = (
        (df_merged_authors["NB_FOLLOWERS"] > 12000) & 
        (df_merged_authors["NB_FOLLOWERS"] <= 40000)
    )
    mask_edgy = df_merged_authors["NB_FOLLOWERS"] > 40000

    df_merged_authors.loc[mask_mainstream, "BASELINE_SEGMENTATION"] = "MAINSTREAM"
    df_merged_authors.loc[mask_trendy, "BASELINE_SEGMENTATION"] = "TRENDY"
    df_merged_authors.loc[mask_edgy, "BASELINE_SEGMENTATION"] = "EDGY"

    print("base_segmentation feature added")
    return df_merged_authors

def merge_author_dataframes(
        df_authors: pd.DataFrame, df_authors_segmentations: pd.DataFrame
    ) -> pd.DataFrame:
    print("Merging author dataframes")

    df_merged_authors = pd.merge(
        df_authors,
        df_authors_segmentations,
        on="AUTHORID",
        how="left",
    )

    # Merging back the two NB_FOLLOWERS columns
    df_merged_authors["NB_FOLLOWERS"] = df_merged_authors["NB_FOLLOWERS_x"]
    if df_merged_authors["NB_FOLLOWERS_y"].dtype == 'object':
        df_merged_authors["NB_FOLLOWERS_y"] = df_merged_authors["NB_FOLLOWERS_y"].astype(float)
    mask = df_merged_authors["NB_FOLLOWERS"].isna()
    df_merged_authors.loc[mask, "NB_FOLLOWERS"] = df_merged_authors.loc[mask, "NB_FOLLOWERS_y"]
    df_merged_authors["NB_FOLLOWERS"] = df_merged_authors["NB_FOLLOWERS"].astype(float)
    df_merged_authors.drop(columns=["NB_FOLLOWERS_x", "NB_FOLLOWERS_y"], inplace=True)

    print("Author dataframes merged")
    return df_merged_authors


def filter_for_object_detection_clothing(
    df_images_labels: pd.DataFrame,
) -> pd.DataFrame:
    print("Filtering for object_detection_clothing")
    with open("data/stored/unique_labels_by_type.json", "r", encoding="utf-8") as f:
        unique_labels_by_type = json.load(f)

    clothing_labels = unique_labels_by_type["object_detection_clothing"]

    mask_clothing_items = ~(
        (df_images_labels["TYPE"] == "object_detection")
        & (~df_images_labels["LABEL_NAME"].isin(clothing_labels))
    )

    df_images_labels = df_images_labels[mask_clothing_items]

    print("object_detection_clothing filtered")
    return df_images_labels


def data_preparation_extend_raw_data():

    df_authors = pd.read_parquet("data/0_raw/mart_authors.parquet")
    df_authors_segmentations = pd.read_parquet("data/0_raw/mart_authors_segmentations.parquet")
    df_images_of_posts = pd.read_parquet("data/0_raw/mart_images_of_posts.parquet")
    df_images_labels = pd.read_parquet("data/0_raw/mart_images_labels.parquet")

    df_merged_authors = merge_author_dataframes(df_authors, df_authors_segmentations)

    df_merged_authors = add_author_feature_base_segmentation(df_merged_authors)
    df_images_of_posts = add_post_feature_is_collab(df_images_of_posts)
    df_merged_authors = add_author_feature_from_author_aggregate(df_images_of_posts, df_merged_authors)

    df_merged_posts = pd.merge(
        df_images_of_posts,
        df_images_labels,
        on='IMAGE_ID',
        how='inner'
    )

    df_merged_posts = add_post_feature_has_sport_item(df_merged_posts)
    df_merged_authors = add_author_feature_from_post_aggregate(df_merged_posts, df_merged_authors)

    df_merged_posts_only_clothing = filter_for_object_detection_clothing(
        df_merged_posts
    )

    df_merged_posts_only_clothing_and_sport = df_merged_posts[
        df_merged_posts["HAS_SPORT_ITEM"] == 1
    ]

    SAVE_DIR = "data/1_interim/extended_data"
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "filtered"), exist_ok=True)

    print("Saving dataframes")
    df_merged_authors.to_parquet(os.path.join(SAVE_DIR, "merged_authors_extended.parquet"))
    print("merged_authors_extended.parquet saved")
    df_images_labels.to_parquet(os.path.join(SAVE_DIR, "images_labels_extended.parquet"))
    print("images_labels_extended.parquet saved")
    df_images_of_posts.to_parquet(os.path.join(SAVE_DIR, "images_of_posts_extended.parquet"))
    print("images_of_posts_extended.parquet saved")
    df_merged_posts.to_parquet(os.path.join(SAVE_DIR, "merged_posts_extended.parquet"))
    print("merged_posts_extended.parquet saved")
    df_merged_posts_only_clothing.to_parquet(
        os.path.join(
            SAVE_DIR, "filtered", "merged_posts_extended_only_clothing.parquet"
        )
    )
    print("merged_posts_extended_only_clothing.parquet saved")
    df_merged_posts_only_clothing_and_sport.to_parquet(
        os.path.join(
            SAVE_DIR, "filtered", "merged_posts_extended_only_clothing_and_sport.parquet"
        )
    )
    print("merged_posts_extended_only_clothing_and_sport.parquet saved")
    print(f"All dataframes saved to {SAVE_DIR = }")

