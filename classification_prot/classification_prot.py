import pandas as pd
import requests
from io import StringIO
from input_data import *
from utils import *


def get_pdb_descriptions(ids):
    """Takes a list of PDB codes and returns a dataframe with details for each one using the RCSB PDB API"""
    
    # create the url
    url = 'https://data.rcsb.org/rest/v1/core/entry/'

    # create the list of urls
    urls = [url + i for i in ids]

    # create the list of responses
    responses = [requests.get(i) for i in urls]
    
    d = str_to_dict(responses)
        
    # Convert the dict of dicts to a dataframe
    df = pd.DataFrame.from_dict(d, orient='index')

    # Create a keyword dict
    kw_dict = {}

    # For each row of the dataframe
    for index, row in df.iterrows():
        kw = row['keywords']
        # If there are keywords
        if kw:
            # Split the keywords
            keywords = [x.strip() for x in kw.split(',')]
            # For each keyword
            for j in keywords:
                # If the keyword is not in the dict
                if j not in kw_dict:
                    # Add it to the dict
                    kw_dict[j] = [index]
                else:
                    # Increment the count
                    kw_dict[j].append(index)

    # return the dataframe and the keyword dict
    return df, kw_dict


# split the dudez list
dudez = [d.strip() for d in dudez_list.split()]

# split the raw data
data = [i.split() for i in raw_data.split('\n') if i and i.split('\t')[0] in dudez]

# separate the protein names and PDB codes into two lists
proteins = [i[0] for i in data]
pdbs = [i[1] for i in data]

# get the descriptions
df, kw_dict = get_pdb_descriptions(pdbs)

# Map the PDB codes to the protein names
df['protein'] = df.index.map(dict(zip(pdbs, proteins)))

# For each pdb code in all the keywords, count the number of times the pdb code appears
counts = {}

# For each element in the keyword dictionary
for key, value in kw_dict.items():
    # For each pdb code in the list of pdb codes for that keyword
    for v in value:
        # If the pdb code is not in the counts dictionary
        if v not in counts:
            # Add it to the dictionary
            counts[v] = 1
        else:
            # Increment the count
            counts[v] += 1
       
# Sort the counts dictionary by value descending
counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

# Get the list of keywords
keywords = list(kw_dict.keys())

# Create an empty dict with the keywords as keys to store the representative pdb codes
representatives = {i: '' for i in keywords}

# For all the counts that are greater than 1 (prioritize the ones with more counts so more classes are represented)
for i in counts:
    if i[1] > 1:
        # Get the protein keywords from the dataframe
        protein_keywords = [k.strip() for k in df.loc[i[0], 'keywords'].split(',')]
        # Add the protein keywords to the list of keywords
        for k in protein_keywords:
            representatives[k] = i[0]
    else:
        # End of the list with counts greater than 1
        break
