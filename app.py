import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from kaggle.api.kaggle_api_extended import KaggleApi


os.environ['KAGGLE_USERNAME'] = 'ziadmostafa284'
os.environ['KAGGLE_KEY'] = 'a5f5db63663803ed83321d3abaf373a6'


api = KaggleApi()
api.authenticate()


dataset = "priyamchoksi/credit-card-transactions-dataset"


api.dataset_download_files(dataset, path="./data", unzip=True)


data_dir = "./data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the downloaded dataset")


df = pd.read_csv(os.path.join(data_dir, csv_files[0]))


print(df.info())


print("Missing values before imputation:")
print(df.isnull().sum())


numeric_columns = df.select_dtypes(include=[np.number]).columns


imputer = SimpleImputer(strategy='mean')


df[numeric_columns] = imputer.fit_transform(df[numeric_columns])


print("\nMissing values after imputation:")
print(df.isnull().sum())


output_file = "preprocessed_data.csv"
df.to_csv(output_file, index=False)

print(f"\nPreprocessing complete. Data saved to '{output_file}'")