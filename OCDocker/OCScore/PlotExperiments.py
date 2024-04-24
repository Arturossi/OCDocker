import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Computed scores
methods = ['SMINA_VINA', 'SMINA_SCORING_DKOES', 'SMINA_VINARDO', 'SMINA_OLD_SCORING_DKOES', 'SMINA_FAST_DKOES', 'SMINA_SCORING_AD4', 'VINA_VINA', 'VINA_VINARDO', 'PLANTS_CHEMPLP', 'PLANTS_PLP', 'PLANTS_PLP95', 'ODDT_RFSCORE_V1', 'ODDT_RFSCORE_V2', 'ODDT_RFSCORE_V3', 'ODDT_PLECRF_P5_L1_S65536', 'ODDT_NNSCORE']

# SF errors
sf_errors = [1.0722327928136997, 1.0398268597087328, 1.0857232084644934, 0.9952596641375944, 1.0737435041506278, 1.0097615735802834, 1.0505106625607852, 1.1593724130807914, 1.090567235696483, 1.0899206041280185, 1.0888535107840012, 0.8187143944182571, 0.7810006111799889, 0.7977128886423381, 1.160753172737404, 0.933017081139368]

# SF AUCs
sf_aucs = [0.6453305212488524, 0.705260928393168, 0.6605514740840851, 0.6802239847112391, 0.6504789623657674, 0.6931809679244665, 0.6379273659154784, 0.626151957001919, 0.6930493463150625, 0.6929375020459317, 0.6944058964922208, 0.674795056093603, 0.6881536264822288, 0.6802124885232972, 0.5753225362694987, 0.6669873908251848]

# SF Scores (SF Error - SF AUC)
sf_scores = [i - j for i, j in zip(sf_errors, sf_aucs)]

# Data
data = {
    'Experimento': list(range(1, 22 + len(methods))),
    'Metodologia': ['Raw Scoring Function'] * len(methods) + [
        'XGB + GA', 'NN', 'NN', 'NN', 'NN', 'AE + NN', 'NN', 
        'AE + NN', 'XGB + GA', 'Multiplo AE + NN', 'Multiplo AE + NN', 
        'Multiplo AE + NN', 'AE + NN', 'AE + NN', 'AE + NN', 'AE + NN',
        'Multiplo AE + NN', 'Multiplo AE + NN', 'Multiplo AE + NN', 'NN',
        'XGB + GA'
    ],
    'Erro (Menor Erro)': sf_errors + [0.5994, 0.6826, 0.6830, 0.6294, 0.6092, 0.6164, 0.6228, 0.6036, 0.6164, 0.9041, 0.9228, 0.9293, 0.6033, 0.6114, 0.6087, 0.6094, 0.9406, 0.9412, 0.9449, 0.6057, 0.5998],
    'AUC (Menor Erro)': sf_aucs + [0.6803, 0.6835, 0.6737, 0.7100, 0.7066, 0.7024, 0.7174, 0.7248, 0.6903, 0.4821, 0.4242, 0.3191, 0.7199, 0.7040, 0.7020, 0.7205, 0.4701, 0.6015, 0.4160, 0.7219, 0.6961],
    'Erro (Maior AUC)': sf_errors + [0.6342, 0.7706, 0.8027, 0.7757, 0.6317, 0.7966, 0.7802, 0.6211, 0.6223, 0.9965, 0.9870, 0.9687, 0.6260, 0.6272, 0.6531, 0.6468, 0.9957, 1.0753, 0.9889, 0.6208, 0.6448],
    'AUC (Maior AUC)': sf_aucs + [0.7204, 0.7527, 0.7437, 0.7469, 0.7634, 0.7434, 0.7451, 0.7691, 0.7232, 0.7127, 0.7377, 0.7032, 0.7611, 0.7593, 0.7546, 0.7657, 0.6876, 0.7032, 0.7251, 0.7517, 0.7070],
    'Erro (Menor Erro - AUC)': sf_errors + [0.6057, 0.6996, 0.6966, 0.6384, 0.6217, 0.6280, 0.6228, 0.6211, 0.6223, 0.9368, 0.9870, 0.9687, 0.6233, 0.6272, 0.6168, 0.6207, 0.9431, 0.9956, 0.9889, 0.6123, 0.5998],
    'AUC (Menor Erro - AUC)': sf_aucs + [0.7096, 0.7167, 0.7114, 0.7223, 0.7586, 0.7280, 0.7174, 0.7691, 0.7232, 0.7015, 0.7377, 0.7032, 0.7603, 0.7593, 0.7476, 0.7549, 0.6464, 0.6944, 0.7251, 0.7475, 0.6961],
    'Score (Menor Erro - AUC)': sf_scores + [-0.1039, -0.0171, -0.0148, -0.0839, -0.1369, -0.1000, -0.0946, -0.1480, -0.1008, 0.2353, 0.2493, 0.2655, -0.1370, -0.1321, -0.1309, -0.1342, 0.2967, 0.3012, 0.2638, -0.1352, -0.0963]
}

from collections import Counter

print(Counter(data['Metodologia']))

df = pd.DataFrame(data)

# Get the Error range
min_error = min([df['Erro (Menor Erro)'].min(), df['Erro (Maior AUC)'].min(), df['Erro (Menor Erro - AUC)'].min()])
max_error = max([df['Erro (Menor Erro)'].max(), df['Erro (Maior AUC)'].max(), df['Erro (Menor Erro - AUC)'].max()])

# Compute the new AUCs
df['AUC (Menor Erro) new'] = df['AUC (Menor Erro)'].apply(lambda x: 1 - x if x < 0.5 else x)
df['AUC (Maior AUC) new'] = df['AUC (Maior AUC)'].apply(lambda x: 1 - x if x < 0.5 else x)
df['AUC (Menor Erro - AUC) new'] = df['AUC (Menor Erro - AUC)'].apply(lambda x: 1 - x if x < 0.5 else x)

# Get the AUC range
min_auc = min([df['AUC (Menor Erro) new'].min(), df['AUC (Maior AUC) new'].min(), df['AUC (Menor Erro - AUC) new'].min()])
max_auc = max([df['AUC (Menor Erro) new'].max(), df['AUC (Maior AUC) new'].max(), df['AUC (Menor Erro - AUC) new'].max()])

error_range = max_error - min_error
auc_range = max_auc - min_auc

# Plotting with the Set2 palette and adjustments for marker and transparency
plt.figure(figsize=(18, 6))

# Create a color mapping for methodologies
color_mapping = {method: color for method, color in zip(df['Metodologia'].unique(), sns.color_palette('Set2', n_colors=df['Metodologia'].nunique()))}

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
        alpha=0.7,  # Adjusting transparency
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
        alpha=0.7,  # Adjusting transparency
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
plt.subplots_adjust(bottom=0.2)


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
plt.figlegend(handles=shape_handles, labels=shape_labels, loc='lower left', bbox_to_anchor=(0.28, 0.025), ncol=1, title='AUC')

# Place the Metodologia color legend at the bottom center
plt.figlegend(handles=color_handles, labels=color_labels, loc='lower center', bbox_to_anchor=(0.57, 0.025), ncol=4, title='Metodologia')

# Use tight_layout to adjust the spacing, but leave the space for the legends under the plot
plt.tight_layout(rect=[0, 0.13, 1, 1])

plt.savefig('experimentos.png', bbox_inches='tight')
plt.show()