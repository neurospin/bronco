##########################################################################
# NSAp - Copyright (C) CEA, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Perform the trainig of various logistic regression model to yield statistics

Scikit-Learn
"""

# System import
from __future__ import print_function
import os

# Package import

# Third party import
from scipy.signal import argrelextrema
from scipy.ndimage.morphology import binary_closing
import scipy.cluster.hierarchy as sch
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn import preprocessing
from sklearn import svm
from collections import Counter
import numpy as np

# Define global parameters that can be tuned if necassary



#============================================
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
from validation_smallsample_io import smallsample_get_data
from validation_dataset_io import parse_pyradiomics_outputs
from validation_dataset_io import parse_pyradiomics_outputs_new

rootsave ='/neurospin/radiomics/studies/metastasis/workspace'
rootsave ='/volatile/frouin/reboot_metastasis/validation'
rootsavefig ='/volatile/frouin/reboot_metastasis/validation'

# AG
if 1:
    rootsavefig ='fig_oldSCRATCH'
    features_file = os.path.join(rootsave, "features_old.csv")
    parse_pyradiomics_outputs(outfile=features_file)
else:
    rootsavefig ='fig_newSCRATCH'
    features_file = os.path.join(rootsave, "features_new.csv")
    parse_pyradiomics_outputs_new(outfile=features_file)


def substract(x):
    return x[2:4]

color_ML = ['powderblue', 'mediumseagreen', 'indianred', 'gold', 'mediumpurple', 'peru', 'cornflowerblue', 'lime']
cmap = sns.diverging_palette(220, 10, as_cmap=True)
#rootsave ='/neurospin/radiomics/studies/metastasis/workspace'

# Numpy array , pandas
#data = smallsample_get_data(m_fn='/neurospin/radiomics/studies/metastasis/pyRadiomics_features_metastases_139lesions_sham_T1gado_FLAIR_1428variables.csv')
data = smallsample_get_data(m_fn=features_file)

md_df = data['md_df']
metastasis_data = data['metastasis_data']
clinical_data= data['clinical_data']
metastasis_data_nosham= data['metastasis_data_nosham']
clinical_data_nosham= data['clinical_data_nosham']
GPA_biclasses_nosham= data['GPA_biclasses_nosham']
multiplicity_lesions= data['multiplicity_lesions']
metastasis_data_without_w= data['metastasis_data_without_w']
list_features_metastasis_data= data['list_features_metastasis_data']
clear_sham = data['clear_sham']
metastasis_data_DF_nosham = data['metastasis_data_DF_nosham']


metastasis_target = clinical_data.loc[:, "GPA_score_new_bin"].copy()
metastasis_target = metastasis_target[clear_sham != "sham"]

accuracies_partial = pandas.DataFrame()
accuracies = pandas.DataFrame()
ROC_df_SVM = pandas.DataFrame()
AUC_SVM = np.float()
VFmetastasis_data_nosham = preprocessing.scale(md_df[clinical_data['segmentation_operator']!='sham'].values)
VFmetastasis_target = clinical_data[clinical_data['segmentation_operator']!='sham']["GPA_score_new_bin"]

#====================================



nb_cv = 5 # 50
linspace_fpr = np.linspace(0, 1, nb_cv)
for j in range(nb_cv):        
    #X_train, X_test, y_train, y_test = train_test_split(VFmetastasis_data_nosham[:, VFlist], VFmetastasis_target, test_size=0.3, stratify=VFmetastasis_target)
    X_train, X_test, y_train, y_test = train_test_split(VFmetastasis_data_nosham, VFmetastasis_target,
                                                        test_size=0.3, stratify=VFmetastasis_target)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)

    X_test_transformed = scaler.transform(X_test)

    # SVM lineaire et radial
    param_grid_svm = [{'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
                      {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
                      {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'degree': [2, 3, 4], 'kernel': ['poly']}]
    svr = svm.SVC(probability=True)
    clf_svm = GridSearchCV(svr, param_grid_svm, cv=5, n_jobs = -1)
    clf_svm.fit(X_train_transformed, y_train)
    print(clf_svm.best_params_)
    print(accuracy_score(clf_svm.predict(X_test_transformed), y_test))
    preds = clf_svm.predict_proba(X_test_transformed)[:, 1]
    fpr, tpr, thre = roc_curve(y_test, preds)
    ROC_df = pandas.DataFrame(dict(fpr=linspace_fpr, tpr=np.interp(linspace_fpr, fpr, tpr)))
    AUC = auc(linspace_fpr, np.interp(linspace_fpr, fpr, tpr))
    ROC_df_SVM = pandas.Panel({n: df for n, df in enumerate([ROC_df_SVM, ROC_df])}).sum(axis=0)
    AUC_SVM = np.sum([AUC_SVM, AUC])
    index_test = (clf_svm.predict(X_test_transformed) == y_test)
    index_test = index_test[index_test == True].index
    results_m = ['none'] * len(index_test)
    for i in range(len(index_test)):
        results_m[i] = index_test[i][0:4]
    results_m = pandas.DataFrame.from_dict(Counter(results_m), orient='index')
    Total_comparaison = pandas.concat([multiplicity_lesions, results_m], axis=1)
    Total_comparaison.columns = ['mets_number', 'prediction SVM correctes']

    accuracies_partial = pandas.DataFrame([accuracy_score(clf_svm.predict(X_test_transformed), y_test)])
    accuracies = pandas.concat([accuracies, accuracies_partial], axis=1)
    print(j)

AUC_SVM = AUC_SVM / nb_cv

ROC_df_SVM = ROC_df_SVM / nb_cv

accuracies = np.round(accuracies, 4).transpose()
methods = ['Support Vector Machine', 'K Nearest Neighbors', 'Random Forest', 'Log Regression L1', 'Log Regression L2',
           'Log Regression elasticnet', 'Voting Classifier'][:1]
accuracies.columns = methods
y_pos = range(len(methods))
pal = {meth: color_ML[methods.index(meth)] for meth in methods}
#L1_analysis = pandas.concat([L1_analysis['features_L1'], L1_analysis.mean(axis=1)], axis=1)
#L1_analysis.columns = ['features_L1', 'mean_coef_L1']
#L1_analysis = L1_analysis[L1_analysis.mean_coef_L1 > 0.7]

with plt.style.context('seaborn-pastel'):
    fig, ax = plt.subplots()
    ax = sns.swarmplot(data=accuracies, palette=pal, edgecolor="black", orient='h', linewidth=1.3, size = 4)
    ax = sns.boxplot(data=accuracies, palette=pal, orient="h", saturation=0.7, width=0.5, linewidth=0.7)
    ax.set_xlim([0, 1])
    plt.title('Performance of several methods to predict the GPA - T1 + T2')
    #plt.sshow()
    plt.savefig(os.path.join(rootsave, rootsavefig, '31_predict_gpabin_boxplot.png'))

#with plt.style.context("ggplot"):
#    fig, ax = plt.subplots()
#    ROC_df_SVM.plot(x='fpr', y='tpr', color="powderblue", linewidth=3, ax=ax,
#                    label='SVM AUC:' + str(round(AUC_SVM * 100, 1)) + "%", linestyle=':')
#    ROC_df_KNN.plot(x='fpr', y='tpr', color="mediumseagreen", linewidth=3, ax=ax,
#                    label='KNN AUC:' + str(round(AUC_KNN * 100, 1)) + "%", linestyle=':')
#    ROC_df_RF.plot(x='fpr', y='tpr', color="indianred", linewidth=3, ax=ax,
#                   label='RF AUC:' + str(round(AUC_RF * 100, 1)) + "%", linestyle=':')
#    ROC_df_LR_L1.plot(x='fpr', y='tpr', color="gold", linewidth=3, ax=ax,
#                      label='LR_L1 AUC:' + str(round(AUC_LR_L1 * 100, 1)) + "%", linestyle=':')
#    ROC_df_LR_L2.plot(x='fpr', y='tpr', color="mediumpurple", linewidth=3, ax=ax,
#                      label='LR_L2 AUC:' + str(round(AUC_LR_L2 * 100, 1)) + "%", linestyle=':')
#    ROC_df_LR_ela.plot(x='fpr', y='tpr', color="peru", linewidth=3, ax=ax,
#                       label='LR_ela AUC:' + str(round(AUC_LR_ela * 100, 1)) + "%", linestyle=':')
#    ROC_df_V.plot(x='fpr', y='tpr', color="cornflowerblue", linewidth=3, ax=ax,
#                  label='Voting Classifier:' + str(round(AUC_V * 100, 1)) + "%", linestyle=':')
#    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), color='black', linestyle='--', alpha=0.3)
#    plt.title("mean ROC curves - T1 + T2")
#    plt.legend(loc=4)
#    #plt.sshow()
#    plt.savefig(os.path.join(rootsave, rootsavefig, '32_predict_gpabin_auc.png'))
#
#
#FLAIR_patt = re.compile('\FLAIR_')
#liste_clusterisee_couleurs_FLAIR_L1 = [re.search(FLAIR_patt, l) for l in L1_analysis['features_L1']]
#liste_clusterisee_couleurs_FLAIR_L1 = map(str, liste_clusterisee_couleurs_FLAIR_L1)
#FLAIR_indices_L1 = [i for i, s in enumerate(liste_clusterisee_couleurs_FLAIR_L1) if 'sre' in s]
#color_labels = ['black'] * len(L1_analysis['features_L1'])
#for i in range(len(L1_analysis['features_L1'])):
#    if i in FLAIR_indices_L1: color_labels[i] = "red"
#with plt.style.context('seaborn-pastel'):
#    fig, ax = plt.subplots()
#    L1_analysis.plot(kind='barh', ax=ax)
#    ax.set_yticklabels(L1_analysis['features_L1'])
#    pos1 = ax.get_position()
#    pos2 = [pos1.x0 + 0.3, pos1.y0, pos1.width / 2.0, pos1.height]
#    ax.set_position(pos2)
#    [t.set_color(i) for (i, t) in zip(color_labels, ax.yaxis.get_ticklabels())]
#    plt.yticks(fontsize=10)
#    plt.title('Features L1 par type d image - T1')
#    #plt.sshow()
#    plt.savefig(os.path.join(rootsave, rootsavefig, '33_predict_gpabin_plot1.png'))
#
#firstorder_patt = re.compile('\_firstorder_')
#liste_clusterisee_couleurs_firstorder_L1 = [re.search(firstorder_patt, l) for l in L1_analysis['features_L1']]
#liste_clusterisee_couleurs_firstorder_L1 = map(str, liste_clusterisee_couleurs_firstorder_L1)
#firstorder_indices_L1 = [i for i, s in enumerate(liste_clusterisee_couleurs_firstorder_L1) if 'sre' in s]
#shape_patt = re.compile('\_shape_')
#liste_clusterisee_couleurs_shape_L1 = [re.search(shape_patt, l) for l in L1_analysis['features_L1']]
#liste_clusterisee_couleurs_shape_L1 = map(str, liste_clusterisee_couleurs_shape_L1)
#shape_indices_L1 = [i for i, s in enumerate(liste_clusterisee_couleurs_shape_L1) if 'sre' in s]
#glrlm_patt = re.compile('\_glrlm_')
#liste_clusterisee_couleurs_glrlm_L1 = [re.search(glrlm_patt, l) for l in L1_analysis['features_L1']]
#liste_clusterisee_couleurs_glrlm_L1 = map(str, liste_clusterisee_couleurs_glrlm_L1)
#glrlm_indices_L1 = [i for i, s in enumerate(liste_clusterisee_couleurs_glrlm_L1) if 'sre' in s]
#glszm_patt = re.compile('\_glszm_')
#liste_clusterisee_couleurs_glszm_L1 = [re.search(glszm_patt, l) for l in L1_analysis['features_L1']]
#liste_clusterisee_couleurs_glszm_L1 = map(str, liste_clusterisee_couleurs_glszm_L1)
#glszm_indices_L1 = [i for i, s in enumerate(liste_clusterisee_couleurs_glszm_L1) if 'sre' in s]
#glcm_patt = re.compile('\_glcm_')
#liste_clusterisee_couleurs_glcm_L1 = [re.search(glcm_patt, l) for l in L1_analysis['features_L1']]
#liste_clusterisee_couleurs_glcm_L1 = map(str, liste_clusterisee_couleurs_glcm_L1)
#glcm_indices_L1 = [i for i, s in enumerate(liste_clusterisee_couleurs_glcm_L1) if 'sre' in s]
#color_labels = ['None'] * len(L1_analysis['features_L1'])
#for i in range(len(L1_analysis['features_L1'])):
#    if i in shape_indices_L1: color_labels[i] = "blue"
#    if i in glrlm_indices_L1: color_labels[i] = "red"
#    if i in glszm_indices_L1: color_labels[i] = "green"
#    if i in glcm_indices_L1: color_labels[i] = "black"
#    if i in firstorder_indices_L1: color_labels[i] = "darkgoldenrod"
#with plt.style.context('seaborn-pastel'):
#    fig, ax = plt.subplots()
#    L1_analysis.plot(kind='barh', ax=ax)
#    ax.set_yticklabels(L1_analysis['features_L1'])
#    pos1 = ax.get_position()
#    pos2 = [pos1.x0 + 0.3, pos1.y0, pos1.width / 2.0, pos1.height]
#    ax.set_position(pos2)
#    [t.set_color(i) for (i, t) in zip(color_labels, ax.yaxis.get_ticklabels())]
#    plt.yticks(fontsize=10)
#    plt.title('Features L1 par famille - T1')
#    #plt.sshow()
#    plt.savefig(os.path.join(rootsave, rootsavefig, '34_predict_gpabin_plot2.png'))
#
#metastasis_data_DF_L1 = metastasis_data_DF_nosham[L1_analysis['features_L1'].tolist()]
#metastasis_data_MAT_L1 = metastasis_data_DF_L1.as_matrix()
#with plt.style.context('seaborn-pastel'):
#    fig = plt.figure(figsize=(12, 12))
#    ax1 = fig.add_axes([0.17, 0.15, 0.1, 0.73])
#    Y = sch.linkage(metastasis_data_MAT_L1, method='ward')
#    Z1 = sch.dendrogram(Y, orientation = 'left')
#    ax2 = fig.add_axes([0.3, 0.89, 0.6, 0.1])
#    Y = sch.linkage(metastasis_data_MAT_L1.transpose(), method='ward')
#    Z2 = sch.dendrogram(Y)
#    idx2 = Z2['leaves']
#    idx1 = Z1['leaves']
#    ax2.set_xticks([])
#    ax2.set_yticks([])
#    ax1.set_xticks([])
#    ax1.set_yticks([])
#    axGPA = fig.add_axes([0.285, 0.15, 0, 0.73])
#    axGPA.set_yticks(range(metastasis_data_MAT_L1.shape[0]))
#    axGPA.set_yticklabels(clinical_data_nosham.ix[idx1,:]["GPA_2cl"], fontsize=6)
#    axGPA.set_xticks([])
#    colors_labels_p = [int(substract(clinical_data_nosham.ix[idx1,:]["patient"][i])) for i in range(len(clinical_data_nosham.ix[idx1,:]["patient"]))]
#    color_labels_GPA = [ "darkgoldenrod" if clinical_data_nosham.ix[idx1,:]["GPA_2cl"][i] == 0 else "darkmagenta" for i in range(len(clinical_data_nosham.ix[idx1,:]["GPA_2cl"])) ]
#    [t.set_color(i) for (i, t) in zip(color_labels_GPA, axGPA.get_yticklabels())]
#    axmatrix = fig.add_axes([0.3, 0.15, 0.6, 0.73])
#    metastasis_data_MAT_L1 = metastasis_data_MAT_L1[:, idx2]
#    metastasis_data_MAT_L1 = metastasis_data_MAT_L1[idx1, :]
#
#    liste_clusterisee_features = [L1_analysis['features_L1'].tolist()[i] for i in idx2]
#
#    FLAIR_patt = re.compile('\FLAIR_')
#    liste_clusterisee_couleurs_FLAIR_L1 = [re.search(FLAIR_patt, l) for l in liste_clusterisee_features]
#    liste_clusterisee_couleurs_FLAIR_L1 = map(str, liste_clusterisee_couleurs_FLAIR_L1)
#    FLAIR_indices_L1 = [i for i, s in enumerate(liste_clusterisee_couleurs_FLAIR_L1) if 'sre' in s]
#    color_labels = ['black'] * len(L1_analysis['features_L1'])
#    for i in range(len(L1_analysis['features_L1'])):
#        if i in FLAIR_indices_L1: color_labels[i] = "red"
#
#    im = axmatrix.matshow(metastasis_data_MAT_L1, aspect='auto', origin='lower', cmap=cmap, vmin=-1, vmax=1)
#    axmatrix.set_xticks(range(len(L1_analysis['features_L1'])))
#    axmatrix.set_xticklabels(liste_clusterisee_features, minor=False)
#    axmatrix.xaxis.set_label_position('bottom')
#    axmatrix.xaxis.tick_bottom()
#    [t.set_color(i) for (i, t) in zip(color_labels, axmatrix.xaxis.get_ticklabels())]
#    plt.xticks(rotation=-90, fontsize=6)
#    axmatrix.set_yticks(range(metastasis_data_MAT_L1.shape[0]))
#    axmatrix.set_yticklabels(clinical_data_nosham.ix[idx1,:]["patient"], fontsize=6)
#    axcolor = fig.add_axes([0.91, 0.15, 0.02, 0.73])
#    plt.colorbar(im, cax=axcolor)
#    plt.title('T1', fontsize = 10)
#    #plt.sshow()
#    plt.savefig(os.path.join(rootsave, rootsavefig, '35_predict_gpabin_plot3.png'))
#
#metastasis_data_DF_L1 = metastasis_data_DF_nosham[L1_analysis['features_L1'].tolist()]
#metastasis_data_MAT_L1 = metastasis_data_DF_L1.as_matrix()
#with plt.style.context('seaborn-pastel'):
#    fig = plt.figure(figsize=(12, 12))
#    ax1 = fig.add_axes([0.17, 0.15, 0.1, 0.73])
#    Y = sch.linkage(metastasis_data_MAT_L1, method='ward')
#    Z1 = sch.dendrogram(Y, orientation = 'left')
#    ax2 = fig.add_axes([0.3, 0.89, 0.6, 0.1])
#    Y = sch.linkage(metastasis_data_MAT_L1.transpose(), method='ward')
#    Z2 = sch.dendrogram(Y)
#    idx2 = Z2['leaves']
#    idx1 = Z1['leaves']
#    ax2.set_xticks([])
#    ax2.set_yticks([])
#    ax1.set_xticks([])
#    ax1.set_yticks([])
#    axGPA = fig.add_axes([0.285, 0.15, 0, 0.73])
#    axGPA.set_yticks(range(metastasis_data_MAT_L1.shape[0]))
#    axGPA.set_yticklabels(clinical_data_nosham.ix[idx1,:]["GPA_2cl"], fontsize=6)
#    color_labels_GPA = [ "darkgoldenrod" if clinical_data_nosham.ix[idx1,:]["GPA_2cl"][i] == 0 else "darkmagenta" for i in range(len(clinical_data_nosham.ix[idx1,:]["GPA_2cl"])) ]
#    [t.set_color(i) for (i, t) in zip(color_labels_GPA, axGPA.get_yticklabels())]
#    axGPA.set_xticks([])
#    axmatrix = fig.add_axes([0.3, 0.15, 0.6, 0.73])
#    metastasis_data_MAT_L1 = metastasis_data_MAT_L1[:, idx2]
#    metastasis_data_MAT_L1 = metastasis_data_MAT_L1[idx1, :]
#    liste_clusterisee_features = [L1_analysis['features_L1'].tolist()[i] for i in idx2]
#
#    firstorder_patt = re.compile('\_firstorder_')
#    liste_clusterisee_couleurs_firstorder_L1 = [re.search(firstorder_patt, l) for l in liste_clusterisee_features]
#    liste_clusterisee_couleurs_firstorder_L1 = map(str, liste_clusterisee_couleurs_firstorder_L1)
#    firstorder_indices_L1 = [i for i, s in enumerate(liste_clusterisee_couleurs_firstorder_L1) if 'sre' in s]
#    shape_patt = re.compile('\_shape_')
#    liste_clusterisee_couleurs_shape_L1 = [re.search(shape_patt, l) for l in liste_clusterisee_features]
#    liste_clusterisee_couleurs_shape_L1 = map(str, liste_clusterisee_couleurs_shape_L1)
#    shape_indices_L1 = [i for i, s in enumerate(liste_clusterisee_couleurs_shape_L1) if 'sre' in s]
#    glrlm_patt = re.compile('\_glrlm_')
#    liste_clusterisee_couleurs_glrlm_L1 = [re.search(glrlm_patt, l) for l in liste_clusterisee_features]
#    liste_clusterisee_couleurs_glrlm_L1 = map(str, liste_clusterisee_couleurs_glrlm_L1)
#    glrlm_indices_L1 = [i for i, s in enumerate(liste_clusterisee_couleurs_glrlm_L1) if 'sre' in s]
#    glszm_patt = re.compile('\_glszm_')
#    liste_clusterisee_couleurs_glszm_L1 = [re.search(glszm_patt, l) for l in liste_clusterisee_features]
#    liste_clusterisee_couleurs_glszm_L1 = map(str, liste_clusterisee_couleurs_glszm_L1)
#    glszm_indices_L1 = [i for i, s in enumerate(liste_clusterisee_couleurs_glszm_L1) if 'sre' in s]
#    glcm_patt = re.compile('\_glcm_')
#    liste_clusterisee_couleurs_glcm_L1 = [re.search(glcm_patt, l) for l in liste_clusterisee_features]
#    liste_clusterisee_couleurs_glcm_L1 = map(str, liste_clusterisee_couleurs_glcm_L1)
#    glcm_indices_L1 = [i for i, s in enumerate(liste_clusterisee_couleurs_glcm_L1) if 'sre' in s]
#    color_labels = ['None'] * len(L1_analysis['features_L1'])
#    for i in range(len(L1_analysis['features_L1'])):
#        if i in shape_indices_L1: color_labels[i] = "blue"
#        if i in glrlm_indices_L1: color_labels[i] = "red"
#        if i in glszm_indices_L1: color_labels[i] = "green"
#        if i in glcm_indices_L1: color_labels[i] = "black"
#        if i in firstorder_indices_L1: color_labels[i] = "darkgoldenrod"
#
#    im = axmatrix.matshow(metastasis_data_MAT_L1, aspect='auto', origin='lower', cmap=cmap, vmin=-1, vmax=1)
#    axmatrix.set_xticks(range(len(L1_analysis['features_L1'])))
#    axmatrix.set_xticklabels(liste_clusterisee_features, minor=False, fontsize = 6)
#    axmatrix.xaxis.set_label_position('bottom')
#    axmatrix.xaxis.tick_bottom()
#    [t.set_color(i) for (i, t) in zip(color_labels, axmatrix.xaxis.get_ticklabels())]
#    plt.xticks(rotation=-90)
#    axmatrix.set_yticks(range(metastasis_data_MAT_L1.shape[0]))
#    axmatrix.set_yticklabels(clinical_data_nosham.ix[idx1,:]["patient"], fontsize=6)
#    axcolor = fig.add_axes([0.91, 0.15, 0.02, 0.73])
#    plt.colorbar(im, cax=axcolor)
#    plt.title('T1', fontsize = 10)
#    #plt.sshow()
#    plt.savefig(os.path.join(rootsave, rootsavefig, '36_predict_gpabin_plot4.png'))
