import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint

# Computed scores
methods = ['SMINA_VINA', 'SMINA_SCORING_DKOES', 'SMINA_VINARDO', 'SMINA_OLD_SCORING_DKOES', 'SMINA_FAST_DKOES', 'SMINA_SCORING_AD4', 'VINA_VINA', 'VINA_VINARDO', 'PLANTS_CHEMPLP', 'PLANTS_PLP', 'PLANTS_PLP95', 'ODDT_RFSCORE_V1', 'ODDT_RFSCORE_V2', 'ODDT_RFSCORE_V3', 'ODDT_PLECRF_P5_L1_S65536', 'ODDT_NNSCORE']

# SF errors
sf_errors = [1.0722327928136997, 1.0398268597087328, 1.0857232084644934, 0.9952596641375944, 1.0737435041506278, 1.0097615735802834, 1.0505106625607852, 1.1593724130807914, 1.090567235696483, 1.0899206041280185, 1.0888535107840012, 0.8187143944182571, 0.7810006111799889, 0.7977128886423381, 1.160753172737404, 0.933017081139368]

# SF AUCs
sf_aucs = [0.6453305212488524, 0.705260928393168, 0.6605514740840851, 0.6802239847112391, 0.6504789623657674, 0.6931809679244665, 0.6379273659154784, 0.626151957001919, 0.6930493463150625, 0.6929375020459317, 0.6944058964922208, 0.674795056093603, 0.6881536264822288, 0.6802124885232972, 0.5753225362694987, 0.6669873908251848]

# SF Scores (SF Error - SF AUC)
sf_scores = [i - j for i, j in zip(sf_errors, sf_aucs)]

# Simple methods
simple_methods_names_all = ['mean', 'median', 'max', 'min', 'std', 'var', 'skew', 'kurtosis', 'quantile_25', 'quantile_75']
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
        'XGB + GA', 'NN', 'NN', 'NN', 'NN', 'AE + NN', 'NN', 
        'AE + NN', 'XGB + GA', 'Multiplo AE + NN', 'Multiplo AE + NN', 
        'Multiplo AE + NN', 'AE + NN', 'AE + NN', 'AE + NN', 'AE + NN',
        'Multiplo AE + NN', 'Multiplo AE + NN', 'Multiplo AE + NN', 'NN',
        'XGB + GA', 'XGB + GA', 'Transformer', 'XGB + GA', 'XGB + GA',
        'Transformer', 'Transformer', 'Transformer', 'Transformer', 'Transformer',
        'PCA95 + NN', 'PCA95 + NN', 'PCA95 + NN', 'PCA95 + NN', 'PCA95 + NN', 
        'PCA95 + NN', 'PCA90 + NN', 'PCA90 + NN'
    ]

# Data
data = {
    'Experimento': list(range(1, len(used_methods) + 1)),
    'Metodologia': used_methods,
    'Erro (Menor Erro)': sf_errors + simple_methods_errors + [0.5994, 0.6826, 0.6830, 0.6294, 0.6092, 0.6164, 0.6228, 0.6036, 0.6164, 0.9041, 0.9228, 0.9293, 0.6033, 0.6114, 0.6087, 0.6094, 0.9406, 0.9412, 0.9449, 0.6057, 0.5998, 0.6104, 0.6412, 0.6306, 0.6157, 0.6294, 0.6280, 0.6315, 0.6243, 0.6379, 0.6103, 0.6191, 0.6134, 0.6096, 0.6164, 0.6142, 0.6201, 0.6221],
    'AUC (Menor Erro)': sf_aucs + simple_methods_AUCs + [0.6803, 0.6835, 0.6737, 0.7100, 0.7066, 0.7024, 0.7174, 0.7248, 0.6903, 0.4821, 0.4242, 0.3191, 0.7199, 0.7040, 0.7020, 0.7205, 0.4701, 0.6015, 0.4160, 0.7219, 0.6961, 0.6968, 0.6540, 0.6830, 0.7014, 0.6756, 0.6727, 0.7043, 0.7021, 0.6924, 0.7274, 0.7095, 0.6954, 0.6872, 0.7268, 0.6819, 0.6798, 0.6892],
    'Erro (Maior AUC)': sf_errors + simple_methods_errors + [0.6342, 0.7706, 0.8027, 0.7757, 0.6317, 0.7966, 0.7802, 0.6211, 0.6223, 0.9965, 0.9870, 0.9687, 0.6260, 0.6272, 0.6531, 0.6468, 0.9957, 1.0753, 0.9889, 0.6208, 0.6448, 0.6412, 0.6607, 0.6333, 0.6268, 0.7613, 0.7419, 0.6506, 0.7346, 0.7115, 0.6386, 1.1151, 1.3010, 0.6374, 0.6487, 0.7943, 1.9377, 0.8442],
    'AUC (Maior AUC)': sf_aucs + simple_methods_AUCs + [0.7204, 0.7527, 0.7437, 0.7469, 0.7634, 0.7434, 0.7451, 0.7691, 0.7232, 0.7127, 0.7377, 0.7032, 0.7611, 0.7593, 0.7546, 0.7657, 0.6876, 0.7032, 0.7251, 0.7517, 0.7070, 0.7271, 0.7273, 0.7197, 0.7162, 0.7472, 0.7515, 0.7445, 0.7518, 0.7547, 0.7443, 0.7585, 0.7401, 0.7343, 0.7376, 0.7546, 0.7513, 0.7468],
    'Erro (Menor Erro - AUC)': sf_errors + simple_methods_errors + [0.6057, 0.6996, 0.6966, 0.6384, 0.6217, 0.6280, 0.6228, 0.6211, 0.6223, 0.9368, 0.9870, 0.9687, 0.6233, 0.6272, 0.6168, 0.6207, 0.9431, 0.9956, 0.9889, 0.6123, 0.5998, 0.6238, 0.6607, 0.6333, 0.6268, 0.6329, 0.6580, 0.6419, 0.6476, 0.6422, 0.6103, 0.6350, 0.6311, 0.6165, 0.6164, 0.6238, 0.6308, 0.6249],
    'AUC (Menor Erro - AUC)': sf_aucs + simple_methods_AUCs + [0.7096, 0.7167, 0.7114, 0.7223, 0.7586, 0.7280, 0.7174, 0.7691, 0.7232, 0.7015, 0.7377, 0.7032, 0.7603, 0.7593, 0.7476, 0.7549, 0.6464, 0.6944, 0.7251, 0.7475, 0.6961, 0.7217, 0.7273, 0.7197, 0.7162, 0.7039, 0.7202, 0.7441, 0.7311, 0.7062, 0.7274, 0.7358, 0.7251, 0.7219, 0.7268, 0.7377, 0.7109, 0.7135],
    'Score (Menor Erro - AUC)': sf_scores + simple_methods_scores + [-0.1039, -0.0171, -0.0148, -0.0839, -0.1369, -0.1000, -0.0946, -0.1480, -0.1008, 0.2353, 0.2493, 0.2655, -0.1370, -0.1321, -0.1309, -0.1342, 0.2967, 0.3012, 0.2638, -0.1352, -0.0963, -0.0978, -0.0666, -0.0863, -0.0894, -0.0710, -0.0621, -0.1023, -0.0835, -0.0640, -0.1171, -0.1009, -0.0940, -0.1054, -0.1104, -0.1139, -0.0801, -0.0886]
}

# Check the length of each list in the data dictionary
print("Length of each list in the data dictionary:")
pprint({key: len(value) for key, value in data.items()})

from collections import Counter

print("\nMetodologia Counter:")
pprint(Counter(data['Metodologia']))

df = pd.DataFrame(data)

# Get the Error range
min_error = min([df['Erro (Menor Erro)'].min(), df['Erro (Maior AUC)'].min(), df['Erro (Menor Erro - AUC)'].min()])
max_error = max([df['Erro (Menor Erro)'].max(), df['Erro (Maior AUC)'].max(), df['Erro (Menor Erro - AUC)'].max()])
#max_error = 1.0

# Compute the new AUCs
df['AUC (Menor Erro) new'] = df['AUC (Menor Erro)'].apply(lambda x: 1 - x if x < 0.5 else x)
df['AUC (Maior AUC) new'] = df['AUC (Maior AUC)'].apply(lambda x: 1 - x if x < 0.5 else x)
df['AUC (Menor Erro - AUC) new'] = df['AUC (Menor Erro - AUC)'].apply(lambda x: 1 - x if x < 0.5 else x)

# Get the AUC range
min_auc = min([df['AUC (Menor Erro) new'].min(), df['AUC (Maior AUC) new'].min(), df['AUC (Menor Erro - AUC) new'].min()])
max_auc = max([df['AUC (Menor Erro) new'].max(), df['AUC (Maior AUC) new'].max(), df['AUC (Menor Erro - AUC) new'].max()])
#max_auc = 0.65

error_range = max_error - min_error
auc_range = max_auc - min_auc

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

# Set alpha value
alpha = 0.9

# Create a color mapping for methodologies
color_mapping = {method: color for method, color in zip(df['Metodologia'].unique(), sns.color_palette(palette_colour, n_colors=df['Metodologia'].nunique()))}

for i, plot in enumerate(['Erro (Menor Erro)', 'Erro (Maior AUC)', 'Erro (Menor Erro - AUC)']):
    plt.subplot(1, 3, i+1)

    metric = plot.replace('Erro (', '(')

    # Set the AUC column name
    auc = f"AUC {metric}"

    # Prepare the data by adding a new column indicating AUC category
    df['AUC_category'] = df[auc].apply(lambda x: '>= 0.5' if x >= 0.5 else '< 0.5')

    # Make 1 - AUC for AUC < 0.5
    df.loc[df['AUC_category'] == '< 0.5', auc] = 1 - df[auc]

    # Plot the df_auc_ge_05 normally
    sns.scatterplot(
        data=df[df['AUC_category'] == '>= 0.5'], 
        x=plot, 
        y=df.columns[i*2+3],  # This will select the corresponding AUC column for each plot
        hue='Metodologia', 
        legend=False, 
        palette=color_mapping,
        alpha=alpha,  # Adjusting transparency
        marker='o', # You can change markers for each method if needed
    )

    # Now plot the df_auc_lt_05 with a different marker (star)
    sns.scatterplot(
        data=df[df['AUC_category'] == '< 0.5'], 
        x=plot, 
        y=df.columns[i*2+3],  # This will select the corresponding AUC column for each plot
        hue='Metodologia', 
        legend=False, 
        palette=color_mapping,
        alpha=alpha,  # Adjusting transparency
        marker='*', # You can change markers for each method if needed
        s=100,
    )

    plt.title(f'Erro vs. AUC {metric}')
    plt.xlim(min_error - error_range * 0.1, max_error + error_range * 0.1)
    #plt.ylim(-0.1, 1.1)
    plt.ylim(min_auc - auc_range * 0.1, max_auc + auc_range * 0.1)
    plt.xlabel('Erro')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

# Extend the space under the plot to add the legend
plt.subplots_adjust(bottom=0.5)


# First legend for the shapes
shape_labels = ['AUC >= 0.5 (= AUC)', 'AUC < 0.5 (= 1-AUC)']
shape_handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='AUC >= 0.5 (= AUC)', markerfacecolor='gray', markersize=10),
    plt.Line2D([0], [0], marker='*', color='w', label='AUC < 0.5 (= 1-AUC)', markerfacecolor='gray', markersize=10)
]

# Second legend for the colors (Metodologia)
color_labels = df['Metodologia'].unique().tolist()
color_handles = [plt.Line2D([0], [0], color=color_mapping[method], lw=4) for method in color_labels]

# Place the AUC shape legend at the bottom left
plt.figlegend(handles=shape_handles, labels=shape_labels, loc='lower left', bbox_to_anchor=(0.26, 0.03), ncol=1, title='AUC')

# Place the Metodologia color legend at the bottom center
plt.figlegend(handles=color_handles, labels=color_labels, loc='lower center', bbox_to_anchor=(0.57, 0.03), ncol=4, title='Metodologia')

# Use tight_layout to adjust the spacing, but leave the space for the legends under the plot
plt.tight_layout(rect=[0, 0.13, 1, 1])

plt.savefig('experimentos.png', bbox_inches='tight')
#plt.show()
plt.close()

# Create a boxplot for each method for the three metrics for Error and AUC
plt.figure(figsize=(20, 8))

for i, plot in enumerate(['Erro (Menor Erro)', 'Erro (Maior AUC)', 'Erro (Menor Erro - AUC)']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(
        data=df, 
        x='Metodologia', 
        y=plot, 
        palette=color_mapping,
        showfliers=False
    )
    plt.title(f'{plot}')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

# Use tight_layout to adjust the spacing
plt.tight_layout()

plt.savefig('experimentos_boxplot.png', bbox_inches='tight')

# Create three new dataframes, one for Error (Menor Erro), one for Error (Maior AUC), and one for Error (Menor Erro - AUC)
df_error_menor_erro = df[['Experimento', 'Metodologia', 'Erro (Menor Erro)']].copy()
df_error_maior_auc = df[['Experimento', 'Metodologia', 'Erro (Maior AUC)']].copy()
df_error_menor_erro_auc = df[['Experimento', 'Metodologia', 'Erro (Menor Erro - AUC)']].copy()

# Rename the Error columns to just 'Erro'
df_error_menor_erro.rename(columns={'Erro (Menor Erro)': 'Erro'}, inplace=True)
df_error_maior_auc.rename(columns={'Erro (Maior AUC)': 'Erro'}, inplace=True)
df_error_menor_erro_auc.rename(columns={'Erro (Menor Erro - AUC)': 'Erro'}, inplace=True)

# Add the metric name to each dataframe in Metodologia (except for Raw Scoring Function and Simple consensus)
df_error_menor_erro['Metodologia'] = df_error_menor_erro['Metodologia'].apply(lambda x: f"{x} (Menor Erro)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)
df_error_maior_auc['Metodologia'] = df_error_maior_auc['Metodologia'].apply(lambda x: f"{x} (Maior AUC)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)
df_error_menor_erro_auc['Metodologia'] = df_error_menor_erro_auc['Metodologia'].apply(lambda x: f"{x} (Menor Erro - AUC)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)

# Concatenate the three dataframes
df_error_concat = pd.concat([df_error_menor_erro, df_error_maior_auc, df_error_menor_erro_auc])

# Do the same for AUC
df_auc_menor_erro = df[['Experimento', 'Metodologia', 'AUC (Menor Erro)']].copy()
df_auc_maior_auc = df[['Experimento', 'Metodologia', 'AUC (Maior AUC)']].copy()
df_auc_menor_erro_auc = df[['Experimento', 'Metodologia', 'AUC (Menor Erro - AUC)']].copy()

# Rename the AUC columns to just 'AUC'
df_auc_menor_erro.rename(columns={'AUC (Menor Erro)': 'AUC'}, inplace=True)
df_auc_maior_auc.rename(columns={'AUC (Maior AUC)': 'AUC'}, inplace=True)
df_auc_menor_erro_auc.rename(columns={'AUC (Menor Erro - AUC)': 'AUC'}, inplace=True)

# Add the metric name to each dataframe in Metodologia (except for Raw Scoring Function and Simple consensus)
df_auc_menor_erro['Metodologia'] = df_auc_menor_erro['Metodologia'].apply(lambda x: f"{x} (Menor Erro)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)
df_auc_maior_auc['Metodologia'] = df_auc_maior_auc['Metodologia'].apply(lambda x: f"{x} (Maior AUC)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)
df_auc_menor_erro_auc['Metodologia'] = df_auc_menor_erro_auc['Metodologia'].apply(lambda x: f"{x} (Menor Erro - AUC)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)

# Concatenate the three dataframes
df_auc_concat = pd.concat([df_auc_menor_erro, df_auc_maior_auc, df_auc_menor_erro_auc])

# Sort the concatenated dataframes by Metodologia
df_error_concat.sort_values('Metodologia', inplace=True)
df_auc_concat.sort_values('Metodologia', inplace=True)

# Put the Raw Scoring Function and Simple consensus at the beginning of the dataframes
df_error_concat = pd.concat([df_error_concat[df_error_concat['Metodologia'] == 'Raw Scoring Function'], df_error_concat[df_error_concat['Metodologia'] == 'Simple consensus'], df_error_concat[df_error_concat['Metodologia'] != 'Raw Scoring Function'], df_error_concat[df_error_concat['Metodologia'] != 'Simple consensus']])
df_auc_concat = pd.concat([df_auc_concat[df_auc_concat['Metodologia'] == 'Raw Scoring Function'], df_auc_concat[df_auc_concat['Metodologia'] == 'Simple consensus'], df_auc_concat[df_auc_concat['Metodologia'] != 'Raw Scoring Function'], df_auc_concat[df_auc_concat['Metodologia'] != 'Simple consensus']])

# Remove all the methods that start with PCA90 + NN
df_error_concat = df_error_concat[~df_error_concat['Metodologia'].str.startswith('PCA90 + NN')]
df_auc_concat = df_auc_concat[~df_auc_concat['Metodologia'].str.startswith('PCA90 + NN')]

# Leave only the methods that ends with (Menor Erro - AUC) or Raw Scoring Function or Simple consensus
#df_error_concat = df_error_concat[df_error_concat['Metodologia'].str.endswith('(Menor Erro - AUC)', na=False) | (df_error_concat['Metodologia'] == 'Raw Scoring Function') | (df_error_concat['Metodologia'] == 'Simple consensus')]
#df_auc_concat = df_auc_concat[df_auc_concat['Metodologia'].str.endswith('(Menor Erro - AUC)', na=False) | (df_auc_concat['Metodologia'] == 'Raw Scoring Function') | (df_auc_concat['Metodologia'] == 'Simple consensus')]

# Set the font size
plt.rcParams['font.size'] = 10 # type: ignore

# Set the metrics
metrics = ['(Menor Erro)', '(Maior AUC)', '(Menor Erro - AUC)']

for metric in metrics:
    # Filter the dataframes
    aux_df_error_concat = df_error_concat[df_error_concat['Metodologia'].str.endswith(metric, na=False) | (df_auc_concat['Metodologia'] == 'Raw Scoring Function') | (df_auc_concat['Metodologia'] == 'Simple consensus')]
    aux_df_auc_concat = df_auc_concat[df_auc_concat['Metodologia'].str.endswith(metric, na=False) | (df_auc_concat['Metodologia'] == 'Raw Scoring Function') | (df_auc_concat['Metodologia'] == 'Simple consensus')]

    # Set the aux metric (Remove parentheses from the metric)
    aux_metric = metric.replace('(', '').replace(')', '')

    # Remove the metric string (with its previous space) from the Metodologia column
    aux_df_error_concat['Metodologia'] = aux_df_error_concat['Metodologia'].apply(lambda x: x.replace(f' {metric}', ''))
    aux_df_auc_concat['Metodologia'] = aux_df_auc_concat['Metodologia'].apply(lambda x: x.replace(f' {metric}', ''))

    # Remake the color mapping for the concatenated dataframes
    color_mapping_error = {method: color for method, color in zip(aux_df_error_concat['Metodologia'].unique(), sns.color_palette(palette_colour, n_colors=aux_df_error_concat['Metodologia'].nunique()))}
    color_mapping_auc = {method: color for method, color in zip(aux_df_auc_concat['Metodologia'].unique(), sns.color_palette(palette_colour, n_colors=aux_df_auc_concat['Metodologia'].nunique()))}

    for plot_type in ['boxplot', 'violin']:
        plt.figure(figsize=(15, 30))  # Adjust the size of the entire figure

        # Create subplots with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10)) # type: ignore

        for i, plot in enumerate(['Erro', 'AUC']):
            ax = ax1 if plot == 'Erro' else ax2
            if plot_type == 'boxplot':
                sns.boxplot(
                    data=aux_df_error_concat if plot == 'Erro' else aux_df_auc_concat, 
                    x='Metodologia', 
                    y=plot, 
                    palette=color_mapping_error if plot == 'Erro' else color_mapping_auc,
                    showfliers=False,
                    ax=ax  # Assign the current axis
                )
            else:
                sns.violinplot(
                    data=aux_df_error_concat if plot == 'Erro' else aux_df_auc_concat, 
                    x='Metodologia', 
                    y=plot, 
                    palette=color_mapping_error if plot == 'Erro' else color_mapping_auc,
                    ax=ax  # Assign the current axis
                )

            ax.grid(True)
            ax.minorticks_on()
            ax.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')
            
            # Get the positions of the boxes
            box_positions = range(len(aux_df_error_concat['Metodologia'].unique()))
            
            # Add shaded area between the markers
            ax.axvspan(
                box_positions[aux_df_error_concat['Metodologia'].unique().tolist().index('Raw Scoring Function')] - 0.5, 
                box_positions[aux_df_error_concat['Metodologia'].unique().tolist().index('Simple consensus')] - 0.5, 
                color='cyan', alpha=0.3
            )
            ax.axvspan(
                box_positions[df_error_concat['Metodologia'].unique().tolist().index('Raw Scoring Function')] + 0.5, 
                box_positions[df_error_concat['Metodologia'].unique().tolist().index('Simple consensus')] + 0.5, 
                color='lime', alpha=0.3
            )

        # Add Title to the entire figure
        fig.suptitle(f'{aux_metric}', fontsize=16) # type: ignore
        
        # Rotate x-axis labels for both subplots
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

        # Use tight_layout to adjust the spacing
        plt.tight_layout()

        plt.savefig(f'experimentos_{plot_type}_{aux_metric}_concat.png', bbox_inches='tight')


