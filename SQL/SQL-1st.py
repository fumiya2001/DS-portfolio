from google.cloud import bigquery

client = bigquery.Client()
df = client.query("SELECT 1").to_dataframe()
print(df)
