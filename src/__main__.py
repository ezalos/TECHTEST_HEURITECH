from fire import Fire
from src.data.data_preparation import data_preparation_extend_raw_data
from src.data.trend_single_pattern import detect_trend_single_pattern
from src.data.download_dataset import download_dataset_from_snowflake

class Main:
    def download_dataset(self):
        download_dataset_from_snowflake()

    def data_preparation(self):
        data_preparation_extend_raw_data()

    def detect_trends(self):
        detect_trend_single_pattern()


if __name__ == "__main__":
	Fire(Main)
