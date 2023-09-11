# convert string to dictionary
def str_to_dict(responses):
    d = {}
    for i in responses:
        # Check if there is a response and if there is a rscb_id
        if i.status_code == 200 and 'rcsb_id' in i.json():
            # Check if there is a title
            if 'struct' in i.json() and 'title' in i.json()['struct']:
                # Assign the title to the dictionary
                d[i.json()['rcsb_id']] = {'title': i.json()['struct']['title']}
            else:
                print(f"Warning: {i.json()['rcsb_id']} has no title")
                # Assign an empty string to the dictionary
                d[i.json()['rcsb_id']] = {'title': ''}
            # Check if there are keywords
            if 'struct_keywords' in i.json() and 'pdbx_keywords' in i.json()['struct_keywords']:
                # Assign the keywords to the dictionary
                d[i.json()['rcsb_id']]['keywords'] = i.json()['struct_keywords']['pdbx_keywords']
            else:
                print(f"Warning: {i.json()['rcsb_id']} has no keywords")
                # Assign an empty string to the dictionary
                d[i.json()['rcsb_id']]['keywords'] = ''
            # Check if there is the length of the protein
            
        else:
            print(f"ERROR: no response for {i}")
    return d