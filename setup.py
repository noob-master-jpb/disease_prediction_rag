import os
import subprocess
subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
import numpy as np
import pandas as pd
from pandas_gbq import to_gbq
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()
project_id = os.getenv("project_id")
dataset_id = os.getenv("dataset_id")
table_id = os.getenv("table_id")

dataset_source = os.getenv("dataset_source")
embeddings_dataset_compress = os.getenv("embeddings_dataset_compress")
embeddings_dataset = os.getenv("embeddings_dataset")


print("project id:", project_id)
print("dataset id:", dataset_id)
print("table id:", table_id)
print("dataset source:", dataset_source)
print("embeddings dataset compress:", embeddings_dataset_compress)
print("embeddings dataset:", embeddings_dataset)

if embeddings_dataset in os.listdir():
    print(f"{embeddings_dataset} found")
else:
    if embeddings_dataset_compress in os.listdir():
        print(f"{embeddings_dataset_compress} is in the folder.")
        print("decompressing to", end=" ")
        subprocess.run(["tar", "-xvJf", embeddings_dataset_compress], check=True)
    else:
        print(f"{embeddings_dataset_compress} is not in the folder.")
        print("downloading...")
        subprocess.run(["wget", dataset_source], check=True)
        print("decompressing to", end=" ")
        subprocess.run(["tar", "-xvJf", embeddings_dataset_compress], check=True)

print("loading embeddings dataset... to pandas dataframe")
df = pd.DataFrame(pd.read_pickle(embeddings_dataset))
df['embedding'] = df['embedding'].apply(list)

print("Connecting to BigQuery...")
client = bigquery.Client(project=project_id)
print("Creating dataset...")
dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
print("Creating table...")
client.create_dataset(dataset, exists_ok=True)


print("Creating Schema...")
schema = [
    bigquery.SchemaField("disease", "STRING", mode="REQUIRED"),    
    bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED")
]


table_ref = client.dataset(dataset_id).table(table_id)
try:
    client.get_table(table_ref)
    print("Table already exists. Skipping creation.")
except:
    table = bigquery.Table(table_ref, schema=schema)
    client.create_table(table)
    print(f"Table {table_id} created in dataset {dataset_id}.")
    
chunk_size = 5000  # change based on API limit
for i in range(0, len(df), chunk_size):
    chunk = df[i:i + chunk_size]
    to_gbq(
        dataframe=chunk,
        destination_table=f"{dataset_id}.{table_id}",
        project_id=project_id,
        if_exists="append" if i > 0 else "replace"
    )

