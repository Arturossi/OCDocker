#!/usr/bin/python3

# Description
###############################################################################
'''
runprank.py is a script to run the software p2rank and then convert its output
to box coordinates to be used as input to docking software like Vina

Created by: Artur Duque Rossi
Version: 0.4
'''

import os
import time
import subprocess
import numpy as np
import pandas as pd

def __cart2pol(x, y):
    '''
    Transform Cartesian to polar coordinates
    Input:
     x [double] - The x coordinate
     y [double] - The y coordinate
    Return:
      theta [double] - The theta angle
      rho   [double] - The rho radial coordinate
    '''
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def __pol2cart(theta, rho):
    '''
    Transform polar to Cartesian coordinates
    Input:
     theta [double] - The theta angle
     rho   [double] - The rho radial coordinate
    Return:
      x [double] - The x coordinate
      y [double] - The y coordinate
    '''
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def __cart2sph(x, y, z):
    '''
    Transform Cartesian to spherical coordinates
    Input:
     x [double] - The x coordinate
     y [double] - The y coordinate
     z [double] - The z coordinate
    Return:
      az [double] - The azimuth
      el [double] - The elevation
      r  [double] - The r radial coordinate
    '''
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def __sph2cart(az, el, r):
    '''
    Transform spherical to Cartesian coordinates
    Input:
     az [double] - The azimuth
     el [double] - The elevation
     r  [double] - The radial coordinate
    Return:
      x [double] - The x coordinate
      y [double] - The y coordinate
      z [double] - The z coordinate
    '''
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def __safe_create_dir(dirname):
    '''
    Function to create a dir if not exists
    Input:
     dirname [string] - File path to be untarred
    Return:
      0 if success
      1 if folder exists
     -1 if any problem has occurred
     -2 should not appear
    '''

    try:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            return 0
        else:
            print(f"The dir {dirname} already exists, aborting its creation")
            return 1
    except Exception as e:
        print(f"Error! Exception: {e}")
        exit(-1)
    return -2

def __process_cluster(clustering, coordinates, fout, suffix = "", coordSystem = "cartesian", spacing = 4.0, boxMaxCutoff = 0.5, boxMinCutoff = 0.1, percentCutoff = 0.5):
    '''
    Function to process the cluster object and print a box file
    Input:
     clustering    [cluster result object]       - SciKit clustering object resulted from any clustering function after fitting
     coordinates   [np.array(np.array(float))]   - NumPy array of numpy arrays of 3 floats containg the X Y Z coordinates
     fout          [string]                      - The path to output box files
     suffix        [string] DEFAULT: ""          - The suffix to append to box files and to create containing folders
     coordSystem   [string] DEFAULT: "cartesian" - The coordinate system to be used. The options are cartesian, polar, spherical
     spacing       [float]  DEFAULT: 4.0         - Expansion size of the box in angstroms
     boxMaxCutoff  [float]  DEFAULT: 0.5         - If the probability value from p2rank is above this value, the pocket WILL be considered as valid, even if its value is below the cutoff (use 1.0 to disable this feature)
     boxMinCutoff  [float]  DEFAULT: 0.1         - If the probability value from p2rank is below this value, the pocket WILL be considered as valid, even if its value is above the cutoff (use 0.0 to disable this feature)
     percentCutoff [float]  DEFAULT: 0.5         - Cutoff to consider how much percentage of box overlapping will determine if two boxes should be merged
    Return:
        Nothing
    '''
    # If any clustering has been done
    if clustering:
        # Fetch each element label
        labels = clustering.labels_

        # Find which labels exists removing repeated elements
        labels_unique = np.unique(labels)

        # Convert coordinates (if necessary)
        if coordSystem.lower() == "polar": # if is polar
            # For each element in coordinates array
            for i, coordinate in enumerate(coordinates):
                # Convert the first two elements (theta, ro) to cartesian (x, y). There is no need to convert z
                coordinates[i][0], coordinates[i][1] = __pol2cart(coordinates[i][0], coordinates[i][1]) #

        elif coordSystem.lower() == "spherical": # if is spherical
            # For each element in coordinates array
            for i, coordinate in enumerate(coordinates):
                # Convert the first three elements (azimuth, elevation, radial coordinate) to cartesian (x, y, z)
                coordinates[i][0], coordinates[i][1], coordinates[i][2] = __sph2cart(coordinates[i][0], coordinates[i][1], coordinates[i][2])

        # Create a dataframe containing x, y, z coordinates and the probability and the rank from P2Rank
        clusteringdf = pd.DataFrame(coordinates,  columns=['x', 'y', 'z', 'probability', 'rank', 'residue'])

        # Add label column to the clusteringdf dataframe
        clusteringdf['label'] = labels
    else:
        # Set the cluster as the raw coordinates
        clusteringdf = pd.DataFrame(coordinates,  columns=['x', 'y', 'z', 'rank', 'residue'])

        # Set the probability as 1 (every box should be used)
        clusteringdf['probability'] = 1.0

        # Set the labels as the rank
        labels = clusteringdf['rank']

        # Add label column to the clusteringdf dataframe
        clusteringdf['label'] = labels

        # Find which labels exists removing repeated elements
        labels_unique = np.unique(labels)

    clusteringdf.to_csv('/mnt/e/Documents/OCDocker/OCDocker/data/ocdb/Astex/1g9v/teste.csv', index=False)

    # If the variable suffix is set
    if suffix:
        # Set the folder variable
        folder = f"/{suffix}"
        # Create the folder
        createDir = __safe_create_dir(f"{fout}{folder}")
        # Change the suffix (to concatenate in box filename)
        suffix = f"_{suffix}"
    else:
        # Set the folder variable as empty
        folder = ""

    # Set the cutoff as the mean of the probabilities (from P2Rank)
    cutoff = clusteringdf['probability'].mean()

    # Force the cutoff to be at maximum the boxMaxCutoff variable and at minimum boxMinCutoff
    cutoff = boxMinCutoff if cutoff < boxMinCutoff else cutoff if cutoff < boxMaxCutoff else boxMaxCutoff

    # List to hold the boxes
    boxes = []

    # For each unique label (after removing repeated labels)
    for label_unique in labels_unique:
        # If the label is -1 (means that its an outlier) or if no probability of the set is above the cutoff
        if str(label_unique) == "-1" or not (clusteringdf[clusteringdf['label'] == label_unique]['probability'] >= cutoff).any():
            # Next iteration
            continue

        # Get the residues in the pocket
        residues = list(clusteringdf[clusteringdf['label'] == label_unique]['residue'])
        residues = [int(residue) for residue in residues]
        residues.sort()

        tmpbox = {}

        # Get min/max of the x/y/z coordinates (round to 3 decimals)
        tmpbox['min_x'] = round(clusteringdf[clusteringdf['label'] == label_unique]['x'].min() - spacing, 3)
        tmpbox['max_x'] = round(clusteringdf[clusteringdf['label'] == label_unique]['x'].max() + spacing, 3)
        tmpbox['min_y'] = round(clusteringdf[clusteringdf['label'] == label_unique]['y'].min() - spacing, 3)
        tmpbox['max_y'] = round(clusteringdf[clusteringdf['label'] == label_unique]['y'].max() + spacing, 3)
        tmpbox['min_z'] = round(clusteringdf[clusteringdf['label'] == label_unique]['z'].min() - spacing, 3)
        tmpbox['max_z'] = round(clusteringdf[clusteringdf['label'] == label_unique]['z'].max() + spacing, 3)
        tmpbox['residues'] = residues

        boxes.append(tmpbox)

    restart = True

    while restart:
        # Make sure that the restart flag is False
        restart = False
        # For each box
        for index, preBox in enumerate(boxes[:-1]):
            # For each box not counting the previous ones
            for moreBox in boxes[index+1:]:
                # Start the percentages
                percentX = 0
                percentY = 0
                percentZ = 0

                # Check if X overlaps and find the percentage of overlap area which is compared to the size of both lines (remember, this is 1D) minus the size of the intersection (it is accounted twice).
                if preBox['max_x'] <= moreBox['max_x'] and preBox['max_x'] >= moreBox['min_x']:
                    percentX = (preBox['max_x'] - moreBox['min_x']) / ((preBox['max_x'] - preBox['min_x']) + (moreBox['max_x'] - moreBox['min_x']) - (moreBox['max_x'] - preBox['min_x']))
                elif preBox['min_x'] <= moreBox['max_x'] and preBox['min_x'] >= moreBox['min_x']:
                    percentX = (moreBox['max_x'] - preBox['min_x']) / ((preBox['max_x'] - preBox['min_x']) + (moreBox['max_x'] - moreBox['min_x']) - (preBox['max_x'] - moreBox['min_x']))
                # Check if percentX is 0 (means no overlap)
                if percentX == 0:
                    continue
                # Check if Y overlaps and find the percentage of overlap area which is compared to the size of both lines (remember, this is 1D) minus the size of the intersection (it is accounted twice).
                if preBox['max_y'] <= moreBox['max_y'] and preBox['max_y'] >= moreBox['min_y']:
                    percentY = (preBox['max_y'] - moreBox['min_y']) / ((preBox['max_y'] - preBox['min_y']) + (moreBox['max_y'] - moreBox['min_y']) - (moreBox['max_y'] - preBox['min_y']))
                elif preBox['min_y'] <= moreBox['max_y'] and preBox['min_y'] >= moreBox['min_y']:
                    percentY = (moreBox['max_y'] - preBox['min_y']) / ((preBox['max_y'] - preBox['min_y']) + (moreBox['max_y'] - moreBox['min_y']) - (preBox['max_y'] - moreBox['min_y']))
                # Check if percentY is 0 (means no overlap)
                if percentY == 0:
                    continue
                # Check if Z overlaps and find the percentage of overlap area which is compared to the size of both lines (remember, this is 1D) minus the size of the intersection (it is accounted twice).
                if preBox['max_z'] <= moreBox['max_z'] and preBox['max_z'] >= moreBox['min_z']:
                    percentZ = (preBox['max_z'] - moreBox['min_z']) / ((preBox['max_z'] - preBox['min_z']) + (moreBox['max_z'] - moreBox['min_z']) - (moreBox['max_z'] - preBox['min_z']))
                elif preBox['min_z'] <= moreBox['max_z'] and preBox['min_z'] >= moreBox['min_z']:
                    percentZ = (moreBox['max_z'] - preBox['min_z']) / ((preBox['max_z'] - preBox['min_z']) + (moreBox['max_z'] - moreBox['min_z']) - (moreBox['max_z'] - preBox['min_z']))
                # Check if percentZ is 0 (means no overlap)
                if percentZ == 0:
                    continue

                # If boxes overlap, merge them
                if (percentX * percentY * percentZ) > percentCutoff:
                    print(percentX * percentY * percentZ)
                    # Start a new empty dict
                    tmpBox = {}
                    # Get the smallest and biggest value in each coordinates
                    tmpBox['min_x'] = preBox['min_x'] if preBox['min_x'] < moreBox['min_x'] else moreBox['min_x']
                    tmpBox['max_x'] = preBox['max_x'] if preBox['max_x'] > moreBox['max_x'] else moreBox['max_x']
                    tmpBox['min_y'] = preBox['min_y'] if preBox['min_y'] < moreBox['min_y'] else moreBox['min_y']
                    tmpBox['max_y'] = preBox['max_y'] if preBox['max_y'] > moreBox['max_y'] else moreBox['max_y']
                    tmpBox['min_z'] = preBox['min_z'] if preBox['min_z'] < moreBox['min_z'] else moreBox['min_z']
                    tmpBox['max_z'] = preBox['max_z'] if preBox['max_z'] > moreBox['max_z'] else moreBox['max_z']
                    # Merge and sort the residues of the box removing repeated values
                    tmpBox['residues'] = list(dict.fromkeys(preBox['residues'] + moreBox['residues']))
                    tmpBox['residues'].sort()

                    # Add the new box to the end of the list
                    boxes.append(tmpBox)
                    # Remove the boxes used to generate the new one
                    boxes.pop(index + 1)
                    boxes.pop(index)
                    restart = True
                    break
            if restart:
                break

    # For each box in boxes
    for index, box in enumerate(boxes):
        # Get dimensions for each axis and its center (round to 3 decimals)
        dim_x = round(abs(box['min_x']) + abs(box['max_x']), 3)
        dim_y = round(abs(box['min_y']) + abs(box['max_y']), 3)
        dim_z = round(abs(box['min_z']) + abs(box['max_z']), 3)
        center_x = round(dim_x/2, 3)
        center_y = round(dim_y/2, 3)
        center_z = round(dim_z/2, 3)

        # Convert the values found above to string with 8 chars (complete with spaces to the left) as the .pdb file model
        min_x = " "*(8-len(str(box['min_x']))) + str(box['min_x'])
        max_x = " "*(8-len(str(box['max_x']))) + str(box['max_x'])
        min_y = " "*(8-len(str(box['min_y']))) + str(box['min_y'])
        max_y = " "*(8-len(str(box['max_y']))) + str(box['max_y'])
        min_z = " "*(8-len(str(box['min_z']))) + str(box['min_z'])
        max_z = " "*(8-len(str(box['max_z']))) + str(box['max_z'])

        dim_x = " "*(8-len(str(dim_x))) + str(dim_x)
        dim_y = " "*(8-len(str(dim_y))) + str(dim_y)
        dim_z = " "*(8-len(str(dim_z))) + str(dim_z)

        center_x = " "*(8-len(str(center_x))) + str(center_x)
        center_y = " "*(8-len(str(center_y))) + str(center_y)
        center_z = " "*(8-len(str(center_z))) + str(center_z)

        # Write out the box file (following the one given in the DUD-E database)
        with open(f'{fout}{folder}/box{index}{suffix}.pdb', 'w') as f:
            f.write(f"HEADER    CORNERS OF BOX      {min_x}{min_y}{min_z}{max_x}{max_y}{max_z}\n")
            f.write(f"REMARK    CENTER (X Y Z)      {center_x}{center_y}{center_z}\n")
            f.write(f"REMARK    DIMENSIONS (X Y Z)  {dim_x}{dim_y}{dim_z}\n")
            f.write(f"REMARK    RESIDUES            {','.join(map(str, box['residues']))}\n")
            f.write(f"ATOM      1  DUA BOX     1    {min_x}{min_y}{min_z}\n")
            f.write(f"ATOM      2  DUB BOX     1    {max_x}{min_y}{min_z}\n")
            f.write(f"ATOM      3  DUC BOX     1    {max_x}{min_y}{max_z}\n")
            f.write(f"ATOM      4  DUD BOX     1    {min_x}{min_y}{max_z}\n")
            f.write(f"ATOM      5  DUE BOX     1    {min_x}{max_y}{min_z}\n")
            f.write(f"ATOM      6  DUF BOX     1    {max_x}{max_y}{min_z}\n")
            f.write(f"ATOM      7  DUG BOX     1    {max_x}{max_y}{max_z}\n")
            f.write(f"ATOM      8  DUH BOX     1    {min_x}{max_y}{max_z}\n")
            f.write("CONECT    1    2    4    5\n")
            f.write("CONECT    2    1    3    6\n")
            f.write("CONECT    3    2    4    7\n")
            f.write("CONECT    4    1    3    8\n")
            f.write("CONECT    5    1    6    8\n")
            f.write("CONECT    6    2    5    7\n")
            f.write("CONECT    7    3    6    8\n")
            f.write("CONECT    8    4    5    7\n")

def run_prank(filein, outpath, algorithms={"AffinityPropagation": False, "AgglomerativeClustering": True, "Birch": False, "DBSCAN": False, "KMeans": False, "MeanShift": False, "MiniBatchKMeans": False, "NoCluster": False, "OPTICS": False, "SpectralClustering": False}, prank = "", threads = 1, coordSystem = "cartesian", spacing = 4.0, boxMaxCutoff = 0.5, boxMinCutoff = 0.1, percentCutoff = 0.5, pocketCutoff = 0.1, verbose = False, debug = False):
    '''
    Function to run p2rank and process its results, converting to a box space to be used in Vina
    Input:
     filein       [string]                    - Input pdb file
     outpath      [string]                    - Output dir (a new folder will be created)
     algorithms   [dict[string] bool]
                    DEFAULT: {
                                "AffinityPropagation": False,
                                "AgglomerativeClustering": True,
                                "Birch": False,
                                "DBSCAN": False,
                                "KMeans": False,
                                "MeanShift": False,
                                "MiniBatchKMeans": False,
                                "NoCluster": False,
                                "OPTICS": False,
                                "SpectralClustering": False"
                             }                - Dictionary of trues and falses of each implemented algorithm. The options are: AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans, MeanShift, MiniBatchKMeans, OPTICS, SpectralClustering
     prank        [string]  DEFAULT: ""          - p2rank file
     threads      [int]     DEFAULT: 1           - Number of threads that the p2rank should use
     coordSystem  [string]  DEFAULT: "cartesian" - The coordinate system to be used. The options are cartesian, polar, spherical
     spacing      [float]   DEFAULT: 4.0         - Expansion size of the box in angstroms
     boxMaxCutoff [float]   DEFAULT: 0.5         - Value to be used as the maximum value as probability cutoff to consider a box as valid (use 1.0 to disable this feature)
     boxMinCutoff [float]   DEFAULT: 0.5         - Value to be used as the minimum value as probability cutoff to consider a box as valid (use 0.0 to disable this feature)
     percentCutoff [float]  DEFAULT: 0.5         - Cutoff to consider how much percentage of box overlapping will determine if two boxes should be merged
     pocketCutoff [float]   DEFAULT: 0.5         - Value to consider (use 0.0 to disable this feature)
     verbose      [bool]    DEFAULT: False       - Verbose mode on/off
     debug        [bool]    DEFAULT: False       - Debug on/off
    Return:
        Nothing
    '''

    # If the prank variable is set
    if prank:
        # If the verbose mode is on
        if verbose:
            # Show the command
            print(f"P2Rank execution command: {' '.join([prank, 'predict','-threads', str(threads),  '-f', filein, '-o', outpath])}")
        # Execute the P2Rank
        subprocess.run([prank, 'predict','-threads', str(threads), '-f', filein, '-o', outpath], stdout=subprocess.DEVNULL)

    # Get the input file name (which will be used to read the output from P2Rank)
    fname = os.path.basename(filein)

    # Read the output
    data = pd.read_csv(f"{outpath}/{fname}_predictions.csv")

    # Remove spaces from the column names
    data.columns = data.columns.str.replace(' ', '')

    # Initialize the atom/probabilities list
    preatoms = []

    # For each line in the surf_atom_ids column
    for index, row in data.iterrows():
        # Split the elements using space and strip each element
        innerAtoms = [s.strip() for s in row['surf_atom_ids'].split()]

        # Add them to the preatoms list with the probability and the rank relative to the atom
        preatoms += list(((innerAtom, row['probability'], row['rank']) for innerAtom in innerAtoms))

    # Create two empty numpy arrays (one will be used to input to the clustering algorithms and the other will be passed to the analysis. Don't worry, the order of the array elements is the same in both!)
    coordinates = np.empty((0,4), float)
    coordinatesFull = np.empty((0,6), float)

    # Initialize the statistics list
    statistics = []

    # Initialize the atoms, probabilities and rank (to ensure that the data is in the same order)
    atoms = [i[0] for i in preatoms]
    probabilities = [i[1] for i in preatoms]
    rank = [i[2] for i in preatoms]

    # Read the .pdb file to capture the x/y/z coordinates
    with open(filein, 'r') as f:
        # For each line in the file
        for line in f:
            # If line start with the ATOM label
            if line.startswith("ATOM"):
                # If the atom ID is in the atom list
                if line[7:11].strip() in atoms:
                    # Finds if the atom index is in the atom list (again to ensure that the right probability is assigned to the right atom)
                    idx = atoms.index(line[7:11].strip())

                    # Check and convert (if needed) the coordinates cartesian/polar/spherical
                    if coordSystem.lower() == "cartesian": # if is cartesian, just read the values
                        v1 = line[31:38]
                        v2 = line[39:46]
                        v3 = line[47:54]
                    elif coordSystem.lower() == "polar": # if is polar, convert x and y, but keep z
                        v1, v2 = __cart2pol(line[31:38], line[39:46])
                        v3 = line[47:54]
                    elif coordSystem.lower() == "spherical": # if is spherical, convert x, y and z
                        v1, v2, v3 = __cart2sph(line[31:38], line[39:46], line[47:54])
                    else: # if the user has typed something wrong, show a warning message and use cartesian
                        print("WARNING: Unknown, coordinate system, using cartesian!")
                        v1 = line[31:38]
                        v2 = line[39:46]
                        v3 = line[47:54]

                    if probabilities[idx] >= pocketCutoff:
                        # Add the data to the numpy array as a list containing the coordinates + extra data [X, Y, Z]/[therta, rho, z]/[az, el, r]
                        coordinates = np.append(coordinates, np.array([[v1, v2, v3, rank[idx]]], float), axis=0)
                        coordinatesFull = np.append(coordinatesFull, np.array([[v1, v2, v3, probabilities[idx], rank[idx], line[23:26]]], float), axis=0)

    ############################################################################
    # Now the code will have the samme pattern:                                #
    ############################################################################
    # 1) Print the algorithm name                                              #
    # 2) Start a timer                                                         #
    # 3) Execute the algoritm                                                  #
    # 4) Check the execution time                                              #
    # 5) Process the output (all files have the same final processing)         #
    # 6) Check the total execution time (algorithm + file processing)          #
    ############################################################################

    # No Cluster
    if algorithms["NoCluster"]:
        start_time = time.time()
        suffix = ""

        if debug:
            suffix = "na"
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}\t0\n")
        if verbose:
            print(f"No processing, the execution time is 0s.")

        __process_cluster(None, coordinates, outpath, suffix = suffix, coordSystem = coordSystem, spacing = spacing, boxMaxCutoff = boxMaxCutoff, boxMinCutoff = boxMinCutoff, percentCutoff = percentCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}+fp\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"File processing time: {round(time.time() - start_time, 2)}s.\n")

    # Affinity Propagation
    if algorithms["AffinityPropagation"]:
        from sklearn.cluster import AffinityPropagation

        if verbose:
            print("Running Affinity Propagation")

        start_time = time.time()
        suffix = ""

        clustering = AffinityPropagation(random_state=0).fit(coordinates)

        if debug:
            suffix = "ap"
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Affinity Propagation execution time: {round(time.time() - start_time, 2)}s.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = suffix, coordSystem = coordSystem, spacing = spacing, boxMaxCutoff = boxMaxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}+fp\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Total execution time for Affinity Propagation: {round(time.time() - start_time, 2)}s.\n")

    # Agglomerative clustering
    if algorithms["AgglomerativeClustering"]:
        from sklearn.cluster import AgglomerativeClustering

        if verbose:
            print("Running Agglomerative Clustering")

        start_time = time.time()
        suffix = ""

        clustering = AgglomerativeClustering().fit(coordinates)

        if debug:
            suffix = "ac"
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f" Agglomerative Clustering execution time: {round(time.time() - start_time, 2)}s.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = suffix, coordSystem = coordSystem, spacing = spacing, boxMaxCutoff = boxMaxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}+fp\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Total execution time for Agglomerative Clustering: {round(time.time() - start_time, 2)}s.\n")

    # Birch
    if algorithms["Birch"]:
        from sklearn.cluster import Birch

        if verbose:
            print("Running Birch")

        start_time = time.time()
        suffix = ""

        clustering = Birch().fit(coordinates)

        if debug:
            suffix = "bi"
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Birch execution time: {round(time.time() - start_time, 2)}s.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = suffix, coordSystem = coordSystem, spacing = spacing, boxMaxCutoff = boxMaxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}+fp\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Total execution time for Birch: {round(time.time() - start_time, 2)}s.\n")

    # DBSCAN
    if algorithms["DBSCAN"]:
        from sklearn.cluster import DBSCAN

        if verbose:
            print("Running DBSCAN")

        start_time = time.time()
        suffix = ""

        clustering = DBSCAN(eps=5, min_samples=5).fit(coordinates)

        if debug:
            suffix = "db"
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"DBSCAN execution time: {round(time.time() - start_time, 2)}s.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = suffix, coordSystem = coordSystem, spacing = spacing, boxMaxCutoff = boxMaxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}+fp\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Total execution time for DBSCAN: {round(time.time() - start_time, 2)}s.\n")

    # KMeans
    if algorithms["KMeans"]:
        from sklearn.cluster import KMeans

        if verbose:
            print("Running KMeans")

        start_time = time.time()
        suffix = ""

        clustering = KMeans(n_clusters=2, random_state=0).fit(coordinates)

        if debug:
            suffix = "km"
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"KMeans execution time: {round(time.time() - start_time, 2)}s.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = suffix, coordSystem = coordSystem, spacing = spacing, boxMaxCutoff = boxMaxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}+fp\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Total execution time for KMeans: {round(time.time() - start_time, 2)}s.\n")

    # Meanshift
    if algorithms["MeanShift"]:
        from sklearn.cluster import MeanShift, estimate_bandwidth

        if verbose:
            print("Running Mean Shift")

        start_time = time.time()
        suffix = ""

        bandwidth = estimate_bandwidth(coordinates, quantile=0.2, n_samples=len(coordinates))
        clustering = MeanShift(bandwidth=bandwidth).fit(coordinates)

        if debug:
            suffix = "ms"
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Mean Shift execution time: {round(time.time() - start_time, 2)}s.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = suffix, coordSystem = coordSystem, spacing = spacing, boxMaxCutoff = boxMaxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}+fp\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Total execution time for Mean Shift: {round(time.time() - start_time, 2)}s.\n")

    # Mini Batch KMeans
    if algorithms["MiniBatchKMeans"]:
        from sklearn.cluster import MiniBatchKMeans

        if verbose:
            print("Running Mini Batch KMeans")

        start_time = time.time()
        suffix = ""

        clustering = MiniBatchKMeans(n_clusters=2).fit(coordinates)

        if debug:
            suffix = "mb"
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Mini Batch KMeans execution time: {round(time.time() - start_time, 2)}s.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = suffix, coordSystem = coordSystem, spacing = spacing, boxMaxCutoff = boxMaxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}+fp\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Total execution time for Mini Batch KMeans: {round(time.time() - start_time, 2)}s.\n")

    # OPTICS
    if algorithms["OPTICS"]:
        from sklearn.cluster import OPTICS

        if verbose:
            print("Running OPTICS")

        start_time = time.time()
        suffix = ""

        clustering = OPTICS(min_samples=5).fit(coordinates)

        if debug:
            suffix = "op"
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"OPTICS execution time: {round(time.time() - start_time, 2)}s.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = suffix, coordSystem = coordSystem, spacing = spacing, boxMaxCutoff = boxMaxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}+fp\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Total execution time for OPTICS: {round(time.time() - start_time, 2)}s.\n")

    # Spectral Clustering
    if algorithms["SpectralClustering"]:
        from sklearn.cluster import SpectralClustering

        if verbose:
            print("Running Spectral Clustering")

        start_time = time.time()
        suffix = ""

        clustering = SpectralClustering(n_clusters=2, random_state=0).fit(coordinates)

        if debug:
            suffix = "sc"
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Spectral Clustering execution time: {round(time.time() - start_time, 2)}s.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = suffix, coordSystem = coordSystem, spacing = spacing, boxMaxCutoff = boxMaxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"{suffix}+fp\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Total execution time for Spectral Clustering: {round(time.time() - start_time, 2)}s.\n")

# Execute the script
if __name__ == "__main__":
    # Variables to be manually adjusted to run the script from prompt
    prank = ""
    fname = "receptor"
    basePath = "/mnt/d/Documents/OCDocker/docking"
    fin = f"{basePath}/{fname}.pdb"
    fout = f"{basePath}/prank"
    threads = 8
    coordSystem = "cartesian"
    spacing = 4.0
    boxMaxCutoff = 0.5
    boxMinCutoff = 0.1
    percentCutoff = 0.5
    pocketCutoff = 0.1
    debug = True
    verbose = True

    # Algorith list
    algorithms = {
        "AffinityPropagation": True,
        "AgglomerativeClustering": True,
        "Birch": True,
        "DBSCAN": True,
        "KMeans": True,
        "MeanShift": True,
        "MiniBatchKMeans": True,
        "NoCluster": True,
        "OPTICS": True,
        "SpectralClustering": True
    }

    run_prank(fin, fout, algorithms = algorithms, prank = prank, threads = threads, coordSystem = coordSystem, spacing = spacing, boxMaxCutoff = boxMaxCutoff, boxMinCutoff = boxMinCutoff, percentCutoff = percentCutoff, pocketCutoff = pocketCutoff, verbose = verbose, debug = debug)
