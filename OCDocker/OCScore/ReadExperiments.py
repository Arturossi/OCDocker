import optuna
from urllib.parse import quote_plus

'''
snames = [
    'XGBoost pre-optimization', 'Feature selection Custom GA', 'XGBoost optimization',
    'NN_Optimization',
    'NN_Optimization_2',
    'NN_Optimization_3_TPE',
    'NN_Optimization_5_TPE',
    'AO_Optimization_9_TPE', 'NN_Optimization_9_TPE',
    'AO_Optimization_10_TPE', 'NN_Optimization_10_TPE',
    'AO_Optimization_11_TPE', 'NN_Optimization_11_TPE',
    'Pre_XGB_Optimization_2', 'feature_selection_2', 'XGB_Optimization_2',
    'AO_Optimization_SF_12_TPE', 'AO_Optimization_LIG_12_TPE', 'AO_Optimization_REC_12_TPE', 'NN_Optimization_12_TPE',
    'AO_Optimization_LIG_14_TPE', 'AO_Optimization_REC_14_TPE', 'NN_Optimization_14_TPE',
    'AO_Optimization_LIG_15_TPE', 'AO_Optimization_REC_15_TPE', 'NN_Optimization_15_TPE',
    'AO_Optimization_16_TPE', 'NN_Optimization_16_TPE',
    'AO_Optimization_17_TPE', 'NN_Optimization_17_TPE',
    'AO_Optimization_18_TPE', 'NN_Optimization_18_TPE',
    'AO_Optimization_19_TPE', 'NN_Optimization_19_TPE',
    'AO_Optimization_LIG_20_TPE', 'AO_Optimization_REC_20_TPE', 'NN_Optimization_20_TPE',
    'AO_Optimization_LIG_21_TPE', 'AO_Optimization_REC_21_TPE', 'NN_Optimization_21_TPE',
    'AO_Optimization_LIG_22_TPE', 'AO_Optimization_REC_22_TPE', 'NN_Optimization_22_TPE',
    'NN_Optimization_23_TPE',
    'Pre_XGB_Optimization_24', 'feature_selection_24', 'XGB_Optimization_24',
    'Pre_XGB_Optimization_25', 'feature_selection_25', 'XGB_Optimization_25',
    'Trans_Optimization_26_TPE',
    'Pre_XGB_Optimization_27', 'feature_selection_27', 'XGB_Optimization_27'
    'Pre_XGB_Optimization_28', 'feature_selection_28', 'XGB_Optimization_28'
    'Trans_Optimization_29_TPE',
    'Trans_Optimization_30_TPE',
    'Trans_Optimization_31_TPE',
    'Trans_Optimization_32_TPE',
    'Trans_Optimization_33_TPE',
    'PCA95_NN_Optimization_34_TPE',
    'PCA95_NN_Optimization_35_TPE',
    'PCA95_NN_Optimization_36_TPE',
    'PCA95_NN_Optimization_37_TPE',
    'PCA95_NN_Optimization_38_TPE',
    'PCA95_NN_Optimization_39_TPE',
    'PCA90_NN_Optimization_40_TPE',
    'PCA90_NN_Optimization_41_TPE',
    'PCA90_NN_Optimization_42_TPE',
    'PCA90_NN_Optimization_43_TPE',
    'PCA90_NN_Optimization_44_TPE',
    'PCA90_NN_Optimization_45_TPE',
    'PCA85_NN_Optimization_46_TPE',
    'PCA85_NN_Optimization_47_TPE',
    'PCA85_NN_Optimization_48_TPE',
    'PCA85_NN_Optimization_49_TPE',
    'PCA85_NN_Optimization_50_TPE',
    'PCA85_NN_Optimization_51_TPE',
    'PCA80_NN_Optimization_52_TPE',
    'PCA80_NN_Optimization_53_TPE',
    'PCA80_NN_Optimization_54_TPE',
    'PCA80_NN_Optimization_55_TPE',
    'PCA80_NN_Optimization_56_TPE',
    'PCA80_NN_Optimization_57_TPE',
    ]
'''

snames = [
        'PCA80_NN_Optimization_57_TPE'
    ]

storage = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@localhost:3306/optimization"

for sname in snames:
    print(f"\nStudy: {sname}")

    study = optuna.load_study(study_name = sname, storage = storage)

    df = study.trials_dataframe()

    df = df[df['state'] == 'COMPLETE']

    df['combined_metric'] = df['value'] - df['user_attrs_AUC']

    df['number'] = df['number'].astype(int)

    print("Number of trials: ", len(df))

    best_rmse_df = df.sort_values(by=['value'], ascending=[True])
    best_auc_df = df.sort_values(by=['user_attrs_AUC'], ascending=[False])
    best_df = df.sort_values(by=['combined_metric'], ascending=[True])

    # Number is int, value is float (not in scientific notation), user_attrs_AUC is float, combined_metric is float
    print("Best RMSE")
    print(best_rmse_df[['number', 'value', 'user_attrs_AUC', 'combined_metric']].iloc[0])

    print("Best AUC")
    print(best_auc_df[['number', 'value', 'user_attrs_AUC', 'combined_metric']].iloc[0])

    print("Best Combined")
    print(best_df[['number', 'value', 'user_attrs_AUC', 'combined_metric']].iloc[0])

    print(f"{len(df)}\t{best_rmse_df['number'].iloc[0]}\t{best_rmse_df['value'].iloc[0]}\t{best_rmse_df['user_attrs_AUC'].iloc[0]}\t{best_auc_df['number'].iloc[0]}\t{best_auc_df['value'].iloc[0]}\t{best_auc_df['user_attrs_AUC'].iloc[0]}\t{best_df['number'].iloc[0]}\t{best_df['combined_metric'].iloc[0]}\t{best_df['user_attrs_AUC'].iloc[0]}")
    