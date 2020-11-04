'''
*************************************************************************
Copyright (c) 2017, Rawan Olayan

>>> SOURCE LICENSE >>>
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation (www.fsf.org); either version 2 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License is available at
http://www.fsf.org/licensing/licenses

>>> END OF LICENSE >>>
*************************************************************************
'''
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from collections import Counter

def run_classification(data,labels,test_idx,trees,c):
	All_scores = []
	length = len(data[0])
	#print len(data)
	total_AUPR_training = 0
	total_AUPR_testing = 0
	folds_AUPR = []
	folds_AUC = []
	folds_precision = []
	folds_recall = []
	folds_f1 = []
	for fold_data,test_idx_fold in zip(data,test_idx):
		train_idx_fold = []
		for idx in range(length):
			if idx not in test_idx_fold:
				train_idx_fold.append(idx)



		fold_data = np.array(fold_data)
		test_idx_fold = np.array(test_idx_fold)
		train_idx_fold = np.array(train_idx_fold)

		
		X_train, X_test = fold_data[train_idx_fold,], fold_data[test_idx_fold,]
		y_train, y_test = np.array(train_idx_fold), np.array(test_idx_fold)


		max_abs_scaler = MaxAbsScaler()
		X_train_maxabs_fit = max_abs_scaler.fit(X_train) 

		X_train_maxabs_transform = max_abs_scaler.transform(X_train)

		X_test_maxabs_transform = max_abs_scaler.transform(X_test)
		rf = RandomForestClassifier(n_estimators=trees ,n_jobs=6,criterion = c,class_weight="balanced",random_state=1357)
		
		rf.fit(X_train_maxabs_transform, y_train)
		try:
			scores_training = rf.decision_function(X_train_maxabs_transform)
			scores_testing = rf.decision_function(X_test_maxabs_transform)
		except:
			scores_training = rf.predict_proba(X_train_maxabs_transform)[:, 1]
			scores_testing = rf.predict_proba(X_test_maxabs_transform)[:, 1]

		y_pred = rf.predict_proba(X_test_maxabs_transform)

		All_scores.append(scores_testing)

		rf_fpr, rf_tpr, rf_thr = roc_curve(y_test, scores_testing)

		auc_val = auc(rf_fpr, rf_tpr)
		print(y_test)

	return All_scores