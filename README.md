# Heuritech Technical tests


- [Heuritech Technical tests](#heuritech-technical-tests)
- [Installation:](#installation)
	- [Set up virtual env](#set-up-virtual-env)
	- [Data](#data)
	- [Running the Test](#running-the-test)

The test subject can be found in  : `docs/Technical Test - Senior Data Scientist.pdf`

# Installation:

## Set up virtual env 

1. Install uv

```sh
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Synchronize virtual env
```sh
uv sync
```

## Data

1. Set environment variables for Snowflake access:

```sh
export SNOWFLAKE_USER="LOUISDEVELLE"
export SNOWFLAKE_PASSWORD="XXXXXXX"
export SNOWFLAKE_ACCOUNT="HEURITECH-AWS_EUW1"
export SNOWFLAKE_WAREHOUSE="COMPUTE_XSMALL"
export SNOWFLAKE_DATABASE="TECHTEST"
export SNOWFLAKE_SCHEMA="TECHTEST"
```

2. Download dataset :

It takes 3 minutes to run on my machine

```sh
uv run python -m src download_dataset
```

3. Process and clean the dataset:

The process is a bit long, it takes 6 minutes to run on my computer.

```sh
uv run python -m src data_preparation
```

## Running the Test

Launch a Jupyter kernel with the uv virtual environment (python path is `.venv/bin/python`) and open `./technical_test.ipynb` to view the test results.

