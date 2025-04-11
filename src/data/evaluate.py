import pandas as pd
import os
import glob

def calculate_lift(
        df_trend: pd.DataFrame, 
        df_overall: pd.DataFrame, 
        segment_column: str = "BASELINE_SEGMENTATION"
    ) -> pd.DataFrame:
    """
    Calculate lift for each segment between a trend dataset and the overall dataset.

    Lift = P(segment | trend) / P(segment)

    Args:
        df_trend: DataFrame containing trend data with author segmentation
        df_overall: DataFrame containing all authors with segmentation
        segment_column: Column name containing segmentation data

    Returns:
        DataFrame with lift values for each segment
    """
    # Get segment distributions
    trend_dist = df_trend[segment_column].value_counts(normalize=True)
    overall_dist = df_overall[segment_column].value_counts(normalize=True)

    # Calculate lift for each segment
    lift_values = {}
    for segment in overall_dist.index:
        if segment in trend_dist:
            # P(segment | trend) / P(segment)
            lift_values[segment] = trend_dist[segment] / overall_dist[segment]
        else:
            lift_values[segment] = 0

    # Create a DataFrame to display results
    lift_df = pd.DataFrame(
        {
            "Segment": list(lift_values.keys()),
            "Trend %": [trend_dist.get(seg, 0) * 100 for seg in lift_values.keys()],
            "Overall %": [overall_dist.get(seg, 0) * 100 for seg in lift_values.keys()],
            "Lift": list(lift_values.values()),
        }
    )

    return lift_df.sort_values("Lift", ascending=False)


def evaluate_trends(
        trends_directory: str, 
        df_authors_with_segmentation: pd.DataFrame | None = None, 
        segment_column: str = "BASELINE_SEGMENTATION"
    ) -> pd.DataFrame:

    trends_eval_results = []
    trend_files = glob.glob(os.path.join(trends_directory, "*.parquet"))

    if df_authors_with_segmentation is None:
        df_authors_with_segmentation = pd.read_parquet(
            "data/1_interim/extended_data/merged_authors_extended.parquet"
        )

    # Get overall distribution for lift calculation
    overall_proportions = df_authors_with_segmentation[segment_column].value_counts(normalize=True)

    for trend_file in trend_files:
        trend_name = os.path.basename(trend_file).replace('.parquet', '')
        df_trend = pd.read_parquet(trend_file)

        # Merge with author data to get segmentation
        df_trend_with_seg = pd.merge(
            df_trend,
            df_authors_with_segmentation[["AUTHORID", segment_column]],
            on="AUTHORID",
            how="left"
        )
        trend_proportions = df_trend_with_seg[segment_column].value_counts(normalize=True)

        result = {'trend_name': trend_name}
        segments = df_authors_with_segmentation[segment_column].unique().tolist()
        cols_for_max_lift = []
        for segment in segments:
            colname = f"{segment_column}={segment}"
            if segment is None:
                continue
            if segment in trend_proportions:
                result[colname] = trend_proportions[segment]
                result[f'{colname}_lift'] = trend_proportions[segment] / overall_proportions[segment]
                cols_for_max_lift.append(f'{colname}_lift')
            else:
                result[colname] = 0.0
                result[f'{colname}_lift'] = 0.0

        result['Total'] = len(df_trend_with_seg)
        result['max_lift'] = max([result[col] for col in cols_for_max_lift])
        trends_eval_results.append(result)

    df_trends_eval_results = pd.DataFrame(trends_eval_results)
    df_trends_eval_results = df_trends_eval_results.sort_values("max_lift", ascending=False)
    return df_trends_eval_results
