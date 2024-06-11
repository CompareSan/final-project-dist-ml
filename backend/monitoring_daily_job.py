from datetime import datetime, timedelta
import logging
import random

import pandas as pd
import psycopg2
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.report import Report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

rand = random.Random()

reference_data = pd.read_parquet("../data/train_df.parquet")
production_data = pd.read_parquet("./predictions.parquet") # Find a way to not load the all historic dataset

now = datetime.now()
one_day_ago = now - timedelta(days=1)
production_data_last_day = production_data[production_data['timestamp'] > one_day_ago]

numerical_features = reference_data.columns.to_list()
column_mapping = ColumnMapping(
    target="target", numerical_features=numerical_features
)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name="target"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
)


def prep_db():
    conn = psycopg2.connect("host=localhost port=5432 user=postgres password=password")
    conn.set_isolation_level(0)  # Set autocommit
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        res = cur.fetchall()
        if len(res) == 0:
            cur.execute("CREATE DATABASE test;")
    conn.close()

    with psycopg2.connect("host=localhost port=5432 dbname=test user=postgres password=password") as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evidently_metrics (
                    timestamp TIMESTAMP NOT NULL,
                    prediction_drift FLOAT,
                    num_drifted_columns INT,
                    share_missing_values FLOAT
                )
            """)
        conn.commit()


def calculate_evidently_metrics():
    now = datetime.now()

    report.run(
        reference_data=reference_data, current_data=production_data_last_day, column_mapping=column_mapping
    )

    result = report.as_dict()

    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing_values = result["metrics"][2]["result"]["current"]["share_of_missing_values"]

    return now, prediction_drift, num_drifted_columns, share_missing_values



def insert_evidently_metrics(
    now, prediction_drift, num_drifted_columns, share_missing_values, curr
):
    curr.execute(
        "insert into evidently_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
        (now, prediction_drift, num_drifted_columns, share_missing_values),
    )


def batch_monitoring():
    prep_db()
    now, prediction_drift, num_drifted_columns, share_missing_values = calculate_evidently_metrics()
    with psycopg2.connect(
        "host=localhost port=5432 dbname=test user=postgres password=password"
    ) as conn:
        with conn.cursor() as curr:
            insert_evidently_metrics(
                now, prediction_drift, num_drifted_columns, share_missing_values, curr
            )
            conn.commit()


if __name__ == "__main__":
    batch_monitoring()

