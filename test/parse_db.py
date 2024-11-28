#!/usr/bin/env python3

import sqlalchemy

import pandas as pd

from urllib.parse import quote_plus

# Set the database connection
ip: str = "192.168.101.2"
ip: str = "localhost"
port: int = 3306
db: str = "tcpaqr"

storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@{ip}:{port}/{db}"

# Connect to the database
engine = sqlalchemy.create_engine(storage)

# Read the complexes table where the ligand_id is tied to the id of the ligands table and the receptor_id is tied to the id of the receptors table and return all the columns from all the tables
query = sqlalchemy.text("SELECT * FROM complexes JOIN ligands ON complexes.ligand_id = ligands.id JOIN receptors ON complexes.receptor_id = receptors.id")

query = sqlalchemy.text("""
    SELECT 
        complexes.name AS complex_name,
        ligands.name AS ligand_name, 
        receptors.name AS receptor_name, 
        complexes.*, 
        ligands.*, 
        receptors.*
    FROM complexes
    JOIN ligands ON complexes.ligand_id = ligands.id
    JOIN receptors ON complexes.receptor_id = receptors.id;
""")

with engine.connect() as connection:
    result = connection.execute(query)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

# Set the columns to drop
to_drop = [
    'created_at',
    'modified_at',
    'ligand_id',
    'receptor_id',
    'id',
    'name'
    ]

# Drop the columns
df.drop(columns=to_drop, inplace=True)

# Check rows with NaN values
df.isna().sum().sum()

# Remove rows with NaN values
df.dropna(inplace=True)

# Check rows where the RadiusOfGyration is 0
df[df['RadiusOfGyration'] == 0]

x = df['ligand_name'].str.extract('_(\d+)', expand=False).astype(int)

sorted_ligand_ids = x.sort_values().reset_index(drop=True)

# Create a range of numbers from 0 to 100000
full_range = set(range(100001))

# Convert the sorted list of ligand IDs to a set
existing_ligand_ids = set(sorted_ligand_ids)

# Find the difference (numbers in full_range but not in existing_ligand_ids)
missing_ids = full_range - existing_ligand_ids

# Convert the result to a sorted list for easier viewing
missing_ids = sorted(missing_ids)

# Display the missing values
print(missing_ids)
