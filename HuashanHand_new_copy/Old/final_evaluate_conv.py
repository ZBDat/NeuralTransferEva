from __future__ import print_function

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, accuracy_score

# %%
foldNum = 6
datastore = './dataStore'
# tempStore = './tempData/withAux'
tempStore = './tempData/withoutAux'


# tempStore = './tempData/DomainKnowledge'

# %%
def calculate_ROCAUC(Y_ture, Y_prob):
    ROCAUC = roc_auc_score(Y_ture, Y_prob) if len(np.unique(Y_ture)) > 1 else 0.0
    return ROCAUC


# %%
if __name__ == "__main__":

    print('-' * 60)
    print('Final Evaluation')
    PID_list = list()
    GroundTruth_list = list()
    Y_Pred_list = list()
    Y_Poss_0 = list()
    Y_Poss_1 = list()

    for step in range(1, foldNum + 1):
        output_file = os.path.join(tempStore, 'test_fold_{}.csv'.format(step))
        df_new = pd.read_csv(output_file)
        PID_list.extend(df_new['PID'].to_list())
        GroundTruth_list.extend(df_new['GroundTruth'].to_list())
        Y_Pred_list.extend(df_new['Y_Pred'].to_list())
        Y_Poss_0.extend(df_new['Y_Poss_0'].to_list())
        Y_Poss_1.extend(df_new['Y_Poss_1'].to_list())

    output_file = os.path.join(tempStore, 'test_info_total.csv')
    df_total = pd.DataFrame(
        {'PID': PID_list, 'GroundTruth': GroundTruth_list, 'Y_Pred': Y_Pred_list, 'Y_Poss_0': Y_Poss_0,
         'Y_Poss_1': Y_Poss_1})
    df_total.to_csv(output_file, index=None)
    # classification_report
    print('Accuracy:{}'.format(accuracy_score(y_true=GroundTruth_list, y_pred=Y_Pred_list)))
    print('ROCAUC:{}'.format(calculate_ROCAUC(Y_ture=GroundTruth_list, Y_prob=Y_Poss_1)))
    target_names = ['N', 'Y']
    print(classification_report(GroundTruth_list, Y_Pred_list, target_names=target_names))
