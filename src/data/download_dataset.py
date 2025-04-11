import os
import pandas as pd
import snowflake.connector
from pathlib import Path


def get_snowflake_connection() -> snowflake.connector.connection.SnowflakeConnection:
    """Establishes a connection to the Snowflake database"""
    try:        
        USER = os.environ.get("SNOWFLAKE_USER")
        PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD")
        ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT")
        WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE")
        DATABASE = os.environ.get("SNOWFLAKE_DATABASE")
        SCHEMA = os.environ.get("SNOWFLAKE_SCHEMA")

        conn = snowflake.connector.connect(
            user=USER,
            password=PASSWORD,
            account=ACCOUNT,
            warehouse=WAREHOUSE,
            database=DATABASE,
            schema=SCHEMA,
        )
        return conn
    except Exception as e:
        print(f"Failed to connect to Snowflake: {str(e)}")
        print("Please verify your environment variables and connection parameters")
        raise


def execute_query(
        conn: snowflake.connector.connection.SnowflakeConnection, 
        query: str
    ) -> pd.DataFrame:
    """Executes a SQL query and returns the results as a pandas DataFrame"""
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(results, columns=column_names)
    cursor.close()
    return df


def save_to_parquet(
        df: pd.DataFrame, 
        filename: str, 
        directory: str = "data/0_raw"
    ):
    """Saves a DataFrame in parquet format"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    filepath = Path(directory) / f"{filename}.parquet"
    
    df.to_parquet(filepath)
    print(f"Data saved to {filepath}")


def download_dataset_from_snowflake():
    conn = get_snowflake_connection()
    
    tables = ["MART_AUTHORS", "MART_AUTHORS_SEGMENTATIONS", 
              "MART_IMAGES_LABELS", "MART_IMAGES_OF_POSTS"]
    
    for table in tables:
        print(f"Downloading table {table}...")
        query = f"SELECT * FROM {table}"
        df = execute_query(conn, query)
        save_to_parquet(df, table.lower())

    conn.close()