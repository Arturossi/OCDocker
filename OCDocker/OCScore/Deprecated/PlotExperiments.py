import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
import colorcet as cc

# Computed scores
methods = ['SMINA_VINA', 'SMINA_SCORING_DKOES', 'SMINA_VINARDO', 'SMINA_OLD_SCORING_DKOES', 'SMINA_FAST_DKOES', 'SMINA_SCORING_AD4', 'VINA_VINA', 'VINA_VINARDO', 'PLANTS_CHEMPLP', 'PLANTS_PLP', 'PLANTS_PLP95', 'ODDT_RFSCORE_V1', 'ODDT_RFSCORE_V2', 'ODDT_RFSCORE_V3', 'ODDT_PLECRF_P5_L1_S65536', 'ODDT_NNSCORE']

# SF errors
sf_errors = [1.0722327928136997, 1.0398268597087328, 1.0857232084644934, 0.9952596641375944, 1.0737435041506278, 1.0097615735802834, 1.0505106625607852, 1.1593724130807914, 1.090567235696483, 1.0899206041280185, 1.0888535107840012, 0.8187143944182571, 0.7810006111799889, 0.7977128886423381, 1.160753172737404, 0.933017081139368]

# SF AUCs
sf_aucs = [0.6453305212488524, 0.705260928393168, 0.6605514740840851, 0.6802239847112391, 0.6504789623657674, 0.6931809679244665, 0.6379273659154784, 0.626151957001919, 0.6930493463150625, 0.6929375020459317, 0.6944058964922208, 0.674795056093603, 0.6881536264822288, 0.6802124885232972, 0.5753225362694987, 0.6669873908251848]

# SF Scores (SF Error - SF AUC)
sf_scores = [i - j for i, j in zip(sf_errors, sf_aucs)]

# Simple methods
simple_methods_names_all = ['mean', 'median', 'max', 'min', 'std', 'var', 'sum', 'range', 'quantile_25', 'quantile_75', 'iqr', 'skew', 'kurtosis']
simple_methods_AUCs_all = [0.734512, 0.721938, 0.703161, 0.672738, 0.520032, 0.520032, 0.734512, 0.531995, 0.704898, 0.739105, 0.501038, 0.510044, 0.516842]
simple_methods_errors_all = [0.701080, 0.718128, 2.018642, 1.664086, 1.313940, 1.214712, 153.708703, 5.205656, 0.849455, 0.859684, 1.483786, 1.819098, 4.853901]

# Set the error threshold
error_threshold = 1.2

# Get the indexes of the methods that are below the threshold
simple_methods_indexes = [i for i, error in enumerate(simple_methods_errors_all) if error < error_threshold]

# Create a list for the used methods and its errors and AUCs
simple_methods_names = [simple_methods_names_all[i] for i in simple_methods_indexes]
simple_methods_errors = [simple_methods_errors_all[i] for i in simple_methods_indexes]
simple_methods_AUCs = [simple_methods_AUCs_all[i] for i in simple_methods_indexes]

# Calculate the scores
simple_methods_scores = [i - j for i, j in zip(simple_methods_errors, simple_methods_AUCs)]

used_methods = ['Raw Scoring Function'] * len(methods) + ['Simple consensus'] * len(simple_methods_names) + [
        'XGB + GA', 'NN', 'NN', 'NN', 'NN', 'NN + AE', 'NN', 
        'NN + AE', 'XGB + GA', 'NN + Multiplo AE', 'NN + Multiplo AE', 
        'NN + Multiplo AE', 'NN + AE', 'NN + AE', 'NN + AE', 'NN + AE',
        'NN + Multiplo AE', 'NN + Multiplo AE', 'NN + Multiplo AE', 'NN',
        'XGB + GA', 'XGB + GA', 'Transformer', 'XGB + GA', 'XGB + GA',
        'Transformer', 'Transformer', 'Transformer', 'Transformer', 'Transformer',
        'NN + PCA95', 'NN + PCA95', 'NN + PCA95', 'NN + PCA95', 'NN + PCA95', 
        'NN + PCA95', 'NN + PCA90', 'NN + PCA90', 'NN + PCA90', 'NN + PCA90', 
        'NN + PCA90', 'NN + PCA90', 'NN + PCA85', 'NN + PCA85', 'NN + PCA85', 
        'NN + PCA85', 'NN + PCA85', 'NN + PCA85', 'NN + PCA80', 'NN + PCA80',
        'NN + PCA80', 'NN + PCA80', 'NN + PCA80', 'NN + PCA80', 'NN + SF Only',
        'NN + SF Only', 'NN + SF Only', 'NN + SF Only', 'NN + SF Only',
        'XGB + SF Only', 'XGB + SF Only', 'NN + SF Only', 'XGB + SF Only',
        'XGB + SF Only', 'XGB + SF Only', 'XGB + SF Only', 'XGB + PCA95',
        'XGB + PCA95', 'XGB + PCA95', 'XGB + PCA95', 'XGB + PCA95', 'XGB + PCA95',
        'XGB + PCA90', 'XGB + PCA90', 'XGB + PCA90', 'XGB + PCA90', 'XGB + PCA90',
        'XGB + PCA90', 'XGB + PCA85', 'XGB + PCA85', 'XGB + PCA85', 'XGB + PCA85',
        'XGB + PCA85', 'XGB + PCA85', 'XGB + PCA80', 'XGB + PCA80', 'XGB + PCA80',
        'XGB + PCA80', 'XGB + PCA80', 'XGB + PCA80', 'Transformer + PCA95', 
        'Transformer + PCA95', 'Transformer + PCA95', 'Transformer + PCA95', 
        'Transformer + PCA95', 'Transformer + PCA95','Transformer + PCA90',
        'Transformer + PCA90', 'Transformer + PCA90', 'Transformer + PCA90',
        'Transformer + PCA90', 'Transformer + PCA90', 'Transformer + PCA85',
        'Transformer + PCA85', 'Transformer + PCA85', 'Transformer + PCA85',
        'Transformer + PCA85', 'Transformer + PCA85', 'Transformer + PCA80',
        'Transformer + PCA80', 'Transformer + PCA80', 'Transformer + PCA80',
        'Transformer + PCA80', 'Transformer + PCA80'
    ]

# Data
data = {
    'Experiment': list(range(1, len(used_methods) + 1)),
    'Methodology': used_methods,
    ## Erro para determinar o melhor modelo
    # ERRO
    'Error (Smallest Error)': sf_errors + simple_methods_errors + [0.5994, 0.6826, 0.6830, 0.6294, 0.6092, 0.6164, 0.6228, 0.6036, 0.6164, 0.9041, 0.9228, 0.9293, 0.6033, 0.6114, 0.6087, 0.6052, 0.9406, 0.9412, 0.9449, 0.6057, 0.5998, 0.6104, 0.6412, 0.6306, 0.6157, 0.6294, 0.6280, 0.6315, 0.6243, 0.6379, 0.6103, 0.6191, 0.6134, 0.6096, 0.6164, 0.6142, 0.6201, 0.6221, 0.6185, 0.6283, 0.6238, 0.6205, 0.6281, 0.6319, 0.6229, 0.6224, 0.6235, 0.6204, 0.6259, 0.6296, 0.6173, 0.6181, 0.6168, 0.6183, 0.6540, 0.6591, 0.6655, 0.6599, 0.6603, 0.6715, 0.6765, 0.6663, 0.6782, 0.6732, 0.6678, 0.6736, 0.6426, 0.6548, 0.6484, 0.6365, 0.6417, 0.6463, 0.6432, 0.6456, 0.6432, 0.6443, 0.6387, 0.6434, 0.6441, 0.6475, 0.6443, 0.6482, 0.6406, 0.6431, 0.6401, 0.6467, 0.6413, 0.6398, 0.6492, 0.6474, 0.6304, 0.6355, 0.6318, 0.6356, 0.6357, 0.6313, 0.6394, 0.6340, 0.6330, 0.6341, 0.6296, 0.6277, 0.6332, 0.6338, 0.6345, 0.6363, 0.6351, 0.6329, 0.6353, 0.6359, 0.6361, 0.6324, 0.6354, 0.6383],
    # AUC
    'AUC (Smallest Error)': sf_aucs + simple_methods_AUCs + [0.6803, 0.6835, 0.6737, 0.7100, 0.7066, 0.7024, 0.7174, 0.7248, 0.6903, 0.4821, 0.4242, 0.3191, 0.7199, 0.7040, 0.7020, 0.7288, 0.4701, 0.6015, 0.4160, 0.7219, 0.6961, 0.6968, 0.6830, 0.6830, 0.7014, 0.6756, 0.6727, 0.7043, 0.7021, 0.6924, 0.7274, 0.7095, 0.6954, 0.6872, 0.7268, 0.6819, 0.6798, 0.6892, 0.6646, 0.6632, 0.6897, 0.6780, 0.6716, 0.6812, 0.6918, 0.6643, 0.6717, 0.6745, 0.6730, 0.6721, 0.6756, 0.6954, 0.6874, 0.7014, 0.6711, 0.6729, 0.6703, 0.6713, 0.6794, 0.6715, 0.6696, 0.6643, 0.6734, 0.6772, 0.6632, 0.6727, 0.6814, 0.6788, 0.6841, 0.6846, 0.6801, 0.6503, 0.6836, 0.6767, 0.6663, 0.6738, 0.6638, 0.6556, 0.6891, 0.6649, 0.6613, 0.6602, 0.6858, 0.6727, 0.6709, 0.6710, 0.6720, 0.6651, 0.6500, 0.6848, 0.7016, 0.6990, 0.6765, 0.6975, 0.6957, 0.6978, 0.6758, 0.6585, 0.6769, 0.6571, 0.6795, 0.6873, 0.6912, 0.6666, 0.6785, 0.6779, 0.6797, 0.6843, 0.6904, 0.6840, 0.6983, 0.6755, 0.6889, 0.6887],
    # AUC para determinar o melhor modelo
    # ERRO
    'Error (Biggest AUC)': sf_errors + simple_methods_errors + [0.6342, 0.7706, 0.8027, 0.7757, 0.6317, 0.7966, 0.7802, 0.6211, 0.6223, 0.9965, 0.9870, 0.9687, 0.6260, 0.6272, 0.6531, 0.6468, 0.9957, 1.0753, 0.9889, 0.6208, 0.6448, 0.6412, 0.6607, 0.6333, 0.6268, 0.7613, 0.7419, 0.6506, 0.7346, 0.7115, 0.6386, 1.1151, 1.3010, 0.6374, 0.6487, 0.7943, 1.9377, 0.8442, 0.9434, 0.7264, 2.8248, 0.9198, 0.6900, 0.8078, 0.8803, 0.9008, 0.6388, 0.8685, 0.7764, 0.7332, 1.8600, 2.5529, 0.9260, 0.7063, 0.7072, 0.7259, 13.4521, 2.8732, 8.9622, 0.6786, 0.6939, 26.8595, 0.6977, 0.6928, 0.6953, 0.6824, 0.6548, 0.6575, 0.6529, 0.6657, 0.6554, 0.6691, 0.6567, 0.6476, 0.6719, 0.6513, 0.6530, 0.6665, 0.6441, 0.6496, 0.6636, 0.6636, 0.6596, 0.6636, 0.6773, 0.6589, 0.6691, 0.6751, 0.6774, 0.6649, 2.4856, 0.8322, 0.6972, 6.0833, 0.6483, 0.7525, 0.9441, 1.0191, 0.7939, 0.9545, 0.9950, 0.9640, 0.9797, 0.9581, 0.8894, 0.9318, 0.9539, 0.8909, 0.9694, 0.8658, 1.7844, 1.3727, 1.1633, 0.7997],
    # AUC
    'AUC (Biggest AUC)': sf_aucs + simple_methods_AUCs + [0.7204, 0.7527, 0.7437, 0.7469, 0.7634, 0.7434, 0.7451, 0.7691, 0.7232, 0.7127, 0.7377, 0.7032, 0.7611, 0.7593, 0.7546, 0.7657, 0.6876, 0.7032, 0.7251, 0.7517, 0.7070, 0.7271, 0.7273, 0.7197, 0.7162, 0.7472, 0.7515, 0.7445, 0.7518, 0.7547, 0.7443, 0.7585, 0.7401, 0.7343, 0.7376, 0.7546, 0.7513, 0.7468, 0.7252, 0.7136, 0.7378, 0.7238, 0.7164, 0.7392, 0.7356, 0.7660, 0.7265, 0.7339, 0.7311, 0.7504, 0.7386, 0.7499, 0.7282, 0.7219, 0.6977, 0.7064, 0.7372, 0.7340, 0.7320, 0.6844, 0.6821, 0.7397, 0.6801, 0.6807, 0.6809, 0.6827, 0.6992, 0.6903, 0.6974, 0.6893, 0.7047, 0.6952, 0.6891, 0.6860, 0.6910, 0.6986, 0.6851, 0.6881, 0.6891, 0.6840, 0.6830, 0.6830, 0.6881, 0.6830, 0.6853, 0.6837, 0.6839, 0.6967, 0.6846, 0.6923, 0.7348, 0.7322, 0.7239, 0.7357, 0.7251, 0.7394, 0.7262, 0.7287, 0.7316, 0.7344, 0.7405, 0.7174, 0.7363, 0.7304, 0.7266, 0.7333, 0.7345, 0.7233, 0.7378, 0.7281, 0.7304, 0.7282, 0.7401, 0.7369],
    # Score (Erro - AUC) para determinar o melhor modelo
    # ERRO
    'Error (Smallest Error - AUC)': sf_errors + simple_methods_errors + [0.6057, 0.6996, 0.6966, 0.6384, 0.6217, 0.6280, 0.6228, 0.6211, 0.6223, 0.9368, 0.9870, 0.9687, 0.6233, 0.6272, 0.6168, 0.6180, 0.9431, 0.9956, 0.9889, 0.6123, 0.5998, 0.6238, 0.6607, 0.6333, 0.6268, 0.6329, 0.6580, 0.6419, 0.6476, 0.6422, 0.6103, 0.6350, 0.6311, 0.6165, 0.6164, 0.6238, 0.6308, 0.6249, 0.6224, 0.6376, 0.6380, 0.6260, 0.6362, 0.6446, 0.6271, 0.6291, 0.6388, 0.6296, 0.6313, 0.6364, 0.6243, 0.6241, 0.6202, 0.6278, 0.6628, 0.6616, 0.6695, 0.6741, 0.6722, 0.6755, 0.6786, 0.6778, 0.6782, 0.6732, 0.6751, 0.6800, 0.6503, 0.6575, 0.6529, 0.6365, 0.6454, 0.6524, 0.6432, 0.6476, 0.6500, 0.6513, 0.6511, 0.6534, 0.6441, 0.6496, 0.6545, 0.6556, 0.6406, 0.6431, 0.6401, 0.6509, 0.6413, 0.6428, 0.6521, 0.6474, 0.6340, 0.6459, 0.6388, 0.6458, 0.6417, 0.6402, 0.6442, 0.6562, 0.6386, 0.6520, 0.6422, 0.6392, 0.6419, 0.6406, 0.6385, 0.6513, 0.6551, 0.6347, 0.6417, 0.6455, 0.6390, 0.6438, 0.6474, 0.6455],
    # AUC
    'AUC (Smallest Error - AUC)': sf_aucs + simple_methods_AUCs + [0.7096, 0.7167, 0.7114, 0.7223, 0.7586, 0.7280, 0.7174, 0.7691, 0.7232, 0.7015, 0.7377, 0.7032, 0.7603, 0.7593, 0.7476, 0.7620, 0.6464, 0.6944, 0.7251, 0.7475, 0.6961, 0.7217, 0.7273, 0.7197, 0.7162, 0.7039, 0.7202, 0.7441, 0.7311, 0.7062, 0.7274, 0.7358, 0.7251, 0.7219, 0.7268, 0.7377, 0.7109, 0.7135, 0.7064, 0.6974, 0.7205, 0.7023, 0.7033, 0.7101, 0.7009, 0.7159, 0.7265, 0.7048, 0.7021, 0.7087, 0.7108, 0.7138, 0.7099, 0.7160, 0.6816, 0.6839, 0.6845, 0.7003, 0.6967, 0.6814, 0.6813, 0.6969, 0.6734, 0.6772, 0.6769, 0.6826, 0.6966, 0.6903, 0.6974, 0.6846, 0.6982, 0.6891, 0.6836, 0.6860, 0.6771, 0.6986, 0.6847, 0.6816, 0.6891, 0.6840, 0.6800, 0.6818, 0.6858, 0.6727, 0.6709, 0.6795, 0.6720, 0.6759, 0.6762, 0.6848, 0.7107, 0.7245, 0.7067, 0.7240, 0.7203, 0.7159, 0.6880, 0.7026, 0.6944, 0.6998, 0.7008, 0.7070, 0.7014, 0.6970, 0.7017, 0.7075, 0.7085, 0.6953, 0.7136, 0.7116, 0.7074, 0.7085, 0.7111, 0.7253],
    # SCORE
    'Score (Smallest Error - AUC)': sf_scores + simple_methods_scores + [-0.1039, -0.0171, -0.0148, -0.0839, -0.1369, -0.1000, -0.0946, -0.1480, -0.1008, 0.2353, 0.2493, 0.2655, -0.1370, -0.1321, -0.1309, -0.1440, 0.2967, 0.3012, 0.2638, -0.1352, -0.0963, -0.0978, -0.0666, -0.0863, -0.0894, -0.0710, -0.0621, -0.1023, -0.0835, -0.0640, -0.1171, -0.1009, -0.0940, -0.1054, -0.1104, -0.1139, -0.0801, -0.0886, -0.0839, -0.0598, -0.0825, -0.0763, -0.0671, -0.0655, -0.0738, -0.0868, -0.0878, -0.0752, -0.0709, -0.0723, -0.0865, -0.0897, -0.0897, -0.0881, -0.0187, -0.0224, -0.0149, -0.0263, -0.0246, -0.0059, -0.0026, -0.0191, 0.0049, -0.0040, -0.0018, -0.0025, -0.0463, -0.0327, -0.0445, -0.0482, -0.0529, -0.0367, -0.0404, -0.0384, -0.0271, -0.0473, -0.0335, -0.0282, -0.0450, -0.0345, -0.0256, -0.0261, -0.0453, -0.0296, -0.0308, -0.0286, -0.0308, -0.0332, -0.0241, -0.0374, -0.0767, -0.0785, -0.0679, -0.0782, -0.0785, -0.0757, -0.0438, -0.0464, -0.0558, -0.0478, -0.0586, -0.0677, -0.0595, -0.0563, -0.0632, -0.0562, -0.0535, -0.0606, -0.0719, -0.0660, -0.0685, -0.0647, -0.0637, -0.0797]
}

data = {
    'Methodology': ['Raw Scoring Function'] * len(methods) + ['Simple consensus'] * len(simple_methods_names),
    ## Erro para determinar o melhor modelo
    # ERRO
    'Error (Smallest Error)': sf_errors + simple_methods_errors,
    # AUC
    'AUC (Smallest Error)': sf_aucs + simple_methods_AUCs,
    # AUC para determinar o melhor modelo
    # ERRO
    'Error (Biggest AUC)': sf_errors + simple_methods_errors,
    # AUC
    'AUC (Biggest AUC)': sf_aucs + simple_methods_AUCs,
    # Score (Erro - AUC) para determinar o melhor modelo
    # ERRO
    'Error (Smallest Error - AUC)': sf_errors + simple_methods_errors,
    # AUC
    'AUC (Smallest Error - AUC)': sf_aucs + simple_methods_AUCs,
    # SCORE
    'Score (Smallest Error - AUC)': sf_scores + simple_methods_scores
}

# Check the length of each list in the data dictionary
print("Length of each list in the data dictionary:")
pprint({key: len(value) for key, value in data.items()})

from collections import Counter

print("\nMethodology Counter:")
pprint(Counter(data['Methodology']))

df = pd.DataFrame(data)

# Define the conversion dictionary
conversion_dict = {
    'study_type': 'Methodology',
    'best_rmse_number': 'Error ID',
    'best_rmse_value': 'Error (Smallest Error)',
    'best_rmse_auc': 'AUC (Smallest Error)',
    'best_auc_number': 'AUC ID',
    'best_auc_value': 'Error (Biggest AUC)',
    'best_auc': 'AUC (Biggest AUC)',
    'best_combined_number': 'Score ID',
    'best_combined_metric': 'Score (Smallest Error - AUC)',
    'best_combined_value': 'Error (Smallest Error - AUC)',
    'best_combined_auc': 'AUC (Smallest Error - AUC)'
}

# Rename the columns if they present in the dataframe
df.rename(columns=conversion_dict, inplace=True)

# Add the data from data dictionary to the end of the dataframe df (filling the missing values with NaN)
df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

# Add the Experiment column (incremental integer starting from 1)
df['Experiment'] = range(1, len(df) + 1)

# Get the Error range
min_error = min([df['Error (Smallest Error - AUC)'].min(), df['Error (Biggest AUC)'].min(), df['Error (Smallest Error - AUC)'].min()])
max_error = max([df['Error (Smallest Error - AUC)'].max(), df['Error (Biggest AUC)'].max(), df['Error (Smallest Error - AUC)'].max()])
#max_error = 1.0

# Compute the new AUCs
df['AUC (Smallest Error - AUC) new'] = df['AUC (Smallest Error - AUC)'].apply(lambda x: 1 - x if x < 0.5 else x)
df['AUC (Biggest AUC) new'] = df['AUC (Biggest AUC)'].apply(lambda x: 1 - x if x < 0.5 else x)
df['AUC (Smallest Error - AUC) new'] = df['AUC (Smallest Error - AUC)'].apply(lambda x: 1 - x if x < 0.5 else x)

# Get the AUC range
min_auc = min([df['AUC (Smallest Error - AUC) new'].min(), df['AUC (Biggest AUC) new'].min(), df['AUC (Smallest Error - AUC) new'].min()])
max_auc = max([df['AUC (Smallest Error - AUC) new'].max(), df['AUC (Biggest AUC) new'].max(), df['AUC (Smallest Error - AUC) new'].max()])
#max_auc = 0.65

error_range = max_error - min_error
auc_range = max_auc - min_auc

# If the plots folder does not exist, create it
if not os.path.exists('plots'):
    os.makedirs('plots')

# Plotting with the chosen palette and adjustments for marker and transparency
plt.figure(figsize=(20, 8))

# Palette
#palette_colour = "Set2"
#palette_colour = "tab10"
palette_colour = "tab20"
#palette_colour = "colorblind"
#palette_colour = "pastel"
#palette_colour = "bright"
#palette_colour = "dark"
#palette_colour = "deep"
#palette_colour = "muted"
#palette_colour = "viridis"
palette_colour = sns.color_palette(cc.glasbey, n_colors=df['Methodology'].nunique())

# Set alpha value
alpha = 0.9

# Create a color mapping for methodologies
color_mapping = {method: color for method, color in zip(df['Methodology'].unique(), sns.color_palette(palette_colour, n_colors=df['Methodology'].nunique()))}

for i, plot in enumerate(['Error (Smallest Error)', 'Error (Biggest AUC)', 'Error (Smallest Error - AUC)']):
    plt.subplot(1, 3, i+1)

    metric = plot.replace('Error (', '(')

    # Set the AUC column name
    auc = f"AUC {metric}"

    # Prepare the data by adding a new column indicating AUC category
    df['AUC_category'] = df[auc].apply(lambda x: '>= 0.5' if x >= 0.5 else '< 0.5')

    # Make 1 - AUC for AUC < 0.5
    df.loc[df['AUC_category'] == '< 0.5', auc] = 1 - df[auc]

    # Get the index of the AUC column
    auc_index = df.columns.get_loc(auc)

    # Plot the df_auc_ge_05 normally
    sns.scatterplot(
        data=df[df['AUC_category'] == '>= 0.5'], 
        x=plot, 
        y=df.columns[auc_index],
        hue='Methodology', 
        legend=False, 
        palette=color_mapping,
        alpha=alpha,  # Adjusting transparency
        marker='o', # You can change markers for each method if needed
    )

    # Now plot the df_auc_lt_05 with a different marker (star)
    sns.scatterplot(
        data=df[df['AUC_category'] == '< 0.5'], 
        x=plot, 
        y=df.columns[auc_index],
        hue='Methodology', 
        legend=False, 
        palette=color_mapping,
        alpha=alpha,  # Adjusting transparency
        marker='*', # You can change markers for each method if needed
        s=100,
    )

    plt.title(f'Error vs. AUC {metric}')
    #plt.xlim(min_error - error_range * 0.1, max_error + error_range * 0.1)
    # Set as minimum value of x-axis the minimum value of the error minus 10% of the error range and the maximum value of x-axis the maximum value of the error plus 10% of the error range for each plot
    error_range = df[plot].max() - df[plot].min()
    plt.xlim(df[plot].min() - error_range * 0.1, df[plot].max() + error_range * 0.1)
    #plt.ylim(-0.1, 1.1)
    plt.ylim(min_auc - auc_range * 0.1, max_auc + auc_range * 0.1)
    plt.xlabel('Error')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

# Extend the space under the plot to add the legend
plt.subplots_adjust(bottom=0.4)

# First legend for the shapes
shape_labels = ['AUC >= 0.5 (= AUC)', 'AUC < 0.5 (= 1-AUC)']
shape_handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='AUC >= 0.5 (= AUC)', markerfacecolor='gray', markersize=10),
    plt.Line2D([0], [0], marker='*', color='w', label='AUC < 0.5 (= 1-AUC)', markerfacecolor='gray', markersize=10)
]

# Second legend for the colors (Methodology)
color_labels = df['Methodology'].unique().tolist()
color_handles = [plt.Line2D([0], [0], color=color_mapping[method], lw=4) for method in color_labels]

# Place the AUC shape legend at the bottom left
plt.figlegend(handles=shape_handles, labels=shape_labels, loc='lower left', bbox_to_anchor=(0.26, 0.03), ncol=1, title='AUC')

# Place the Methodology color legend at the bottom center
plt.figlegend(handles=color_handles, labels=color_labels, loc='lower center', bbox_to_anchor=(0.57, 0.03), ncol=4, title='Methodology')

# Use tight_layout to adjust the spacing, but leave the space for the legends under the plot
plt.tight_layout(rect=[0, 0.22, 1, 1])

plt.savefig('plots/Experiments.png', bbox_inches='tight', dpi=300)
#plt.show()
plt.close('all')

"""
# Create a boxplot for each method for the three metrics for Error and AUC
plt.figure(figsize=(20, 8))

for i, plot in enumerate(['Error (Smallest Error - AUC)', 'Error (Biggest AUC)', 'Error (Smallest Error - AUC)']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(
        data=df, 
        x='Methodology', 
        y=plot, 
        palette=color_mapping,
        showfliers=False,
        hue='Methodology',
        legend=False
    )
    plt.title(f'{plot}')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

# Use tight_layout to adjust the spacing
plt.tight_layout()

plt.savefig('plots/Experiments_boxplot.png', bbox_inches='tight')
"""

# Create three new dataframes, one for Error (Smallest Error), one for Error (Biggest AUC), and one for Error (Smallest Error - AUC)
df_error_menor_erro = df[['Experiment', 'Methodology', 'Error (Smallest Error - AUC)']].copy()
df_error_maior_auc = df[['Experiment', 'Methodology', 'Error (Biggest AUC)']].copy()
df_error_menor_erro_auc = df[['Experiment', 'Methodology', 'Error (Smallest Error - AUC)']].copy()

# Rename the Error columns to just 'Error'
df_error_menor_erro.rename(columns={'Error (Smallest Error - AUC)': 'Error'}, inplace=True)
df_error_maior_auc.rename(columns={'Error (Biggest AUC)': 'Error'}, inplace=True)
df_error_menor_erro_auc.rename(columns={'Error (Smallest Error - AUC)': 'Error'}, inplace=True)

# Add the metric name to each dataframe in Methodology (except for Raw Scoring Function and Simple consensus)
df_error_menor_erro['Methodology'] = df_error_menor_erro['Methodology'].apply(lambda x: f"{x} (Smallest Error)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)
df_error_maior_auc['Methodology'] = df_error_maior_auc['Methodology'].apply(lambda x: f"{x} (Biggest AUC)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)
df_error_menor_erro_auc['Methodology'] = df_error_menor_erro_auc['Methodology'].apply(lambda x: f"{x} (Smallest Error - AUC)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)

# Concatenate the three dataframes
df_error_concat = pd.concat([df_error_menor_erro, df_error_maior_auc, df_error_menor_erro_auc])

# Do the same for AUC
df_auc_menor_erro = df[['Experiment', 'Methodology', 'AUC (Smallest Error)']].copy()
df_auc_maior_auc = df[['Experiment', 'Methodology', 'AUC (Biggest AUC)']].copy()
df_auc_menor_erro_auc = df[['Experiment', 'Methodology', 'AUC (Smallest Error - AUC)']].copy()

# Rename the AUC columns to just 'AUC'
df_auc_menor_erro.rename(columns={'AUC (Smallest Error)': 'AUC'}, inplace=True)
df_auc_maior_auc.rename(columns={'AUC (Biggest AUC)': 'AUC'}, inplace=True)
df_auc_menor_erro_auc.rename(columns={'AUC (Smallest Error - AUC)': 'AUC'}, inplace=True)

# Add the metric name to each dataframe in Methodology (except for Raw Scoring Function and Simple consensus)
df_auc_menor_erro.loc[:, 'Methodology'] = df_auc_menor_erro['Methodology'].apply(lambda x: f"{x} (Smallest Error)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)
df_auc_maior_auc.loc[:, 'Methodology'] = df_auc_maior_auc['Methodology'].apply(lambda x: f"{x} (Biggest AUC)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)
df_auc_menor_erro_auc.loc[:, 'Methodology'] = df_auc_menor_erro_auc['Methodology'].apply(lambda x: f"{x} (Smallest Error - AUC)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)

# Concatenate the three dataframes
df_auc_concat = pd.concat([df_auc_menor_erro, df_auc_maior_auc, df_auc_menor_erro_auc])

# Sort the concatenated dataframes by Methodology
df_error_concat.sort_values('Methodology', inplace=True)
df_auc_concat.sort_values('Methodology', inplace=True)

# Put the Raw Scoring Function and Simple consensus at the beginning of the dataframes
df_error_concat = pd.concat([df_error_concat[df_error_concat['Methodology'] == 'Raw Scoring Function'], df_error_concat[df_error_concat['Methodology'] == 'Simple consensus'], df_error_concat[df_error_concat['Methodology'] != 'Raw Scoring Function'], df_error_concat[df_error_concat['Methodology'] != 'Simple consensus']])
df_auc_concat = pd.concat([df_auc_concat[df_auc_concat['Methodology'] == 'Raw Scoring Function'], df_auc_concat[df_auc_concat['Methodology'] == 'Simple consensus'], df_auc_concat[df_auc_concat['Methodology'] != 'Raw Scoring Function'], df_auc_concat[df_auc_concat['Methodology'] != 'Simple consensus']])

# Remove all the methods that start with any of the following strings (empty list means no methods will be removed)
to_remove = []

for m in to_remove:
    df_error_concat = df_error_concat[~df_error_concat['Methodology'].str.startswith(m)]
    df_auc_concat = df_auc_concat[~df_auc_concat['Methodology'].str.startswith(m)]

# Set the font size
plt.rcParams['font.size'] = 10 # type: ignore

# Set the metrics
metrics = ['(Smallest Error)', '(Biggest AUC)', '(Smallest Error - AUC)']

for metric in metrics:
    # Filter the dataframes
    aux_df_error_concat = df_error_concat[df_error_concat['Methodology'].str.endswith(metric, na=False) | (df_auc_concat['Methodology'] == 'Raw Scoring Function') | (df_auc_concat['Methodology'] == 'Simple consensus')]
    aux_df_auc_concat = df_auc_concat[df_auc_concat['Methodology'].str.endswith(metric, na=False) | (df_auc_concat['Methodology'] == 'Raw Scoring Function') | (df_auc_concat['Methodology'] == 'Simple consensus')]

    # Set the aux metric (Remove parentheses from the metric)
    aux_metric = metric.replace('(', '').replace(')', '')

    # Remove the metric string (with its previous space) from the Methodology column
    aux_df_error_concat.loc[:, 'Methodology'] = aux_df_error_concat['Methodology'].apply(lambda x: x.replace(f' {metric}', ''))
    aux_df_auc_concat.loc[:, 'Methodology'] = aux_df_auc_concat['Methodology'].apply(lambda x: x.replace(f' {metric}', ''))

    # Remake the color mapping for the concatenated dataframes
    color_mapping_error = {method: color for method, color in zip(aux_df_error_concat['Methodology'].unique(), sns.color_palette(palette_colour, n_colors=aux_df_error_concat['Methodology'].nunique()))}
    color_mapping_auc = {method: color for method, color in zip(aux_df_auc_concat['Methodology'].unique(), sns.color_palette(palette_colour, n_colors=aux_df_auc_concat['Methodology'].nunique()))}

    for plot_type in ['boxplot', 'violin']:
        plt.close('all')

        plt.figure(figsize=(15, 30))  # Adjust the size of the entire figure

        # Create subplots with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10)) # type: ignore

        for i, plot in enumerate(['Error', 'AUC']):
            ax = ax1 if plot == 'Error' else ax2
            if plot_type == 'boxplot':
                sns.boxplot(
                    data=aux_df_error_concat if plot == 'Error' else aux_df_auc_concat, 
                    x='Methodology', 
                    y=plot, 
                    palette=color_mapping_error if plot == 'Error' else color_mapping_auc,
                    showfliers=False,
                    ax=ax,
                    hue='Methodology',
                    legend=False
                )
            else:
                sns.violinplot(
                    data=aux_df_error_concat if plot == 'Error' else aux_df_auc_concat, 
                    x='Methodology', 
                    y=plot, 
                    palette=color_mapping_error if plot == 'Error' else color_mapping_auc,
                    ax=ax,
                    hue='Methodology',
                    legend=False
                )

            ax.grid(True)
            ax.minorticks_on()
            ax.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')
            
            # Get the positions of the boxes
            box_positions = range(len(aux_df_error_concat['Methodology'].unique()))
            
            # Add shaded area between the markers
            ax.axvspan(
                box_positions[aux_df_error_concat['Methodology'].unique().tolist().index('Raw Scoring Function')] - 0.5, 
                box_positions[aux_df_error_concat['Methodology'].unique().tolist().index('Simple consensus')] - 0.5, 
                color='cyan', alpha=0.3
            )
            ax.axvspan(
                box_positions[df_error_concat['Methodology'].unique().tolist().index('Raw Scoring Function')] + 0.5, 
                box_positions[df_error_concat['Methodology'].unique().tolist().index('Simple consensus')] + 0.5, 
                color='lime', alpha=0.3
            )

        # Add Title to the entire figure
        fig.suptitle(f'{aux_metric}', fontsize=16) # type: ignore
        
        # Rotate x-axis labels for both subplots
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

        # Use tight_layout to adjust the spacing
        plt.tight_layout()

        plt.savefig(f'plots/Experiments_{plot_type}_{aux_metric}_concat.png', bbox_inches='tight')


plt.close('all')

# Make bar plots for the error and AUC for each metric (3 bars for each method in the same plot)
plt.figure(figsize=(20, 8))

for i, plot in enumerate(['Error (Smallest Error)', 'Error (Biggest AUC)', 'Error (Smallest Error - AUC)']):
    plt.subplot(1, 3, i+1)
    sns.barplot(
        data=df, 
        x='Methodology', 
        y=plot, 
        palette=color_mapping,
        hue='Methodology',
        legend=False
    )
    plt.title(f'{plot.replace("Error (", "").replace(")", "")}')
    plt.xticks(rotation=90)
    plt.ylabel('Error')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

# Add the title to the entire figure
plt.suptitle('Error', fontsize=16)

# Use tight_layout to adjust the spacing
plt.tight_layout()

plt.savefig('plots/Experiments_error_barplot.png', bbox_inches='tight')

plt.close('all')

plt.figure(figsize=(20, 8))

for i, plot in enumerate(['AUC (Smallest Error - AUC)', 'AUC (Biggest AUC)', 'AUC (Smallest Error - AUC)']):
    plt.subplot(1, 3, i+1)
    sns.barplot(
        data=df, 
        x='Methodology', 
        y=plot, 
        palette=color_mapping,
        hue='Methodology',
        legend=False
    )
    plt.title(f'{plot.replace("AUC (", "").replace(")", "")}')
    plt.xticks(rotation=90)
    plt.ylabel('AUC')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

# Add the title to the entire figure
plt.suptitle('AUC', fontsize=16)

# Use tight_layout to adjust the spacing
plt.tight_layout()

plt.savefig('plots/Experiments_auc_barplot.png', bbox_inches='tight')
