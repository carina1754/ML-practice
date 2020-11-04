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

import sys,random,argparse
from Graph_utils import *
from SNF import *
from Classify import *


def split_unkown_interactions(DT,folds=10):

	row,col = DT.shape
	negative = []
	for i in range(row):
		for j in range(col):
			if DT[i][j] == 0:
				negative.append((i,j))

	random.shuffle(negative)

	testing = np.array_split(np.array(negative),folds)
	return [list(fold) for fold in testing]



def get_similarities(sim_file,dMap):

	sim = []

	for line in open(sim_file).readlines():
		edge_list = get_edge_list(line.strip())
		sim.append(make_sim_matrix(edge_list,dMap))
	return sim




def run_DDR(R_file, D_sim_file, T_sim_file, no_of_splits, K, K_SNF, T_SNF, no_trees, split_criteria, out_file):

	R_file = "nr_admat_dgc_mat_2_line.txt"
	D_sim_file="nr_D_similarities.txt"
	T_sim_file="nr_T_similarities.txt"
	out_file="nr_results.txt"


	# read interaction and similarity files
	(D,T,DT_signature,aAllPossiblePairs,dDs,dTs,diDs,diTs) = get_All_D_T_thier_Labels_Signatures(R_file)
	R = get_edge_list(R_file)
	DT = get_adj_matrix_from_relation(R,dDs,dTs)
	D_sim = get_similarities(D_sim_file,dDs)
	T_sim = get_similarities(T_sim_file,dTs)

	row,col = DT.shape

	#---------------------- Start DDR functionality ---------------------------------

	labels = mat2vec(DT)
	test_idx = []

	folds_features = []
	for fold in split_unkown_interactions(DT,no_of_splits):


		#-------- infer zero intactions for Drugs and targets---------------
		DT_impute_D = impute_zeros(DT,D_sim[0])
		DT_impute_T = impute_zeros(np.transpose(DT),T_sim[0])

		#-------- construct GIP similarity drugs and targegs ----------------

		GIP_D = Get_GIP_profile(np.transpose(DT_impute_D),"d")
		GIP_T = Get_GIP_profile(DT_impute_T,"t")


		#-------- Perform SNF ----------------------------------------------

		WD = []
		WT = []

		for s in D_sim:
			WD.append(s)
		WD.append(GIP_D)

		for s in T_sim:
			WT.append(s)
		WT.append(GIP_T)

		D_SNF = SNF(WD,K_SNF,T_SNF)
		t_SNF = SNF(WT,K_SNF,T_SNF)

		#--------- Get neigborhood for drugs and target --------------------

		DS_D = FindDominantSet(D_SNF,5)
		DS_T = FindDominantSet(t_SNF,5)

		np.fill_diagonal(DS_D,0)
		np.fill_diagonal(DS_T,0)



		#--------- extract features -----------------------------------------

		features = get_features_per_fold(DT,DS_D,DS_T, True)
		folds_features.append(list(zip(*features)))
		test_idx.append([i*col+j for (i,j) in fold])


	#---------------- Get DDR predictions ------------------------------------

	predictions = run_classification(folds_features,labels,test_idx,no_trees,split_criteria)

	novel_predictions = []
	for pred,idxs in zip(predictions,test_idx):
		data = [(score,idx) for score,idx in zip(pred,idxs)]
		data.sort(key=lambda x:x[0],reverse=True)
		for s, i in data:
                     d = i/col
                     t = i%col
		if diDs in novel_predictions:
			novel_predictions.append((diDs[(int(d) + 1)], diTs[(int(t) + 1)], s))
		else:
			print("1")

	novel_predictions.sort(key=lambda x:x[2], reverse=True)

	oF = open(out_file,"w")
	for D,T,S in novel_predictions:
		oF.write("%s\t%s\t%f\n"%(D,T,S))
	oF.close()


	return 


if __name__ == '__main__':

#	run_DDR('nr_admat_dgc_mat_2_line.txt', "--DSimilarity=nr_D_similarities.txt", "--TSimilarity=nr_T_similarities.txt",  "--no_of_splits=10", "--K=5", "--K_SNF=3"," --T_SNF=10", "--N=100", "--s=gini", "--outfile=nr_results.txt")
	parser = argparse.ArgumentParser(description='DDR a method to predict drug target interactions')
	requiredArguments = parser.add_argument_group('required named arguments')
	requiredArguments.add_argument("--interaction",action="store",dest="R_file",help="Name of the file containg drug target interaction tuples",required=False)
	requiredArguments.add_argument("--DSimilarity",action="store",dest="D_sim_file",help="Name of the file containg drug similarties file names",required=False)
	requiredArguments.add_argument("--TSimilarity",action="store",dest="T_sim_file",help="Name of the file containg target similarties file names",required=False)
	requiredArguments.add_argument("--outfile",action="store",dest="out_file",help="Output file to write predictions", type=str,required=False)

	parser.add_argument("--no_of_splits",action="store",dest="no_of_splits",help="Number of parts to split unkown interactions. Default: 10",type=int,default=10)
	parser.add_argument("--K",action="store",dest="K",help="Number of nearest neighbors for drugs and targets neigborhood. Default: 5", type=int,default=5)
	parser.add_argument("--K_SNF",action="store",dest="K_SNF",help="Number of neighbors similarity fusion. Default: 3", type=int,default=3)
	parser.add_argument("--T_SNF",action="store",dest="T_SNF",help="Number of iteration for similarity fusion. Default: 10", type=int,default=10)
	parser.add_argument("--N",action="store",dest="no_of_trees",help="Number trees for random forest. Default: 100", type=int,default=100)
	parser.add_argument("--s",action="store",dest="split",help="Split critera for random forest trees. Default: gini", type=str,default="gini")

	oArgs = parser.parse_args()
	run_DDR(oArgs.R_file,oArgs.D_sim_file,oArgs.T_sim_file,oArgs.no_of_splits,oArgs.K,oArgs.K_SNF,oArgs.T_SNF,oArgs.no_of_trees,oArgs.split,oArgs.out_file)

