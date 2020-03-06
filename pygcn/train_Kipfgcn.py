#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_Kipfgcn.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/03/05 12:05:01
'''

# here put the import lib
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from models import KipfGCN

import numpy as np
import argparse
import logging
import time

logger = logging.getLogger(__name__)

# Set seed
def set_seed(args):
    np.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def load_data(args):
	"""
	Reads the data from pickle file
	Parameters
	----------
	args: params object
	args.data: The type of the dataset to be loaded
	Returns
	-------
	data: the dataset
	num_classes: the num of dataset classes
	"""
	logger.info("loading data")
	root_dir	= os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', args.data)
	dataset		= Planetoid(root_dir, args.data, T.NormalizeFeatures())
	num_classes  = dataset.num_classes
	data	= dataset[0]
	return data, num_classes

def train(args, data, model):
	# prepare optimizer
	if args.opt == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
	else:
		optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
	
	# set best metric
	best_val = 0.0
	best_test = 0.0
	
	# training here
	logger.info("begin training...")
	for epoch in range(args.max_epochs):
		model.train()
		optimizer.zero_grad()
		output = model(data.x, data.edge_index)
		logits = output[data.train_mask]
		train_loss = F.nll_loss(logits, data.y[data.train_mask])
		train_loss.backward()
		optimizer.step()

		# eval for every epoch
		model.eval()
		output = model(data.x, data.edge_index)
		train_acc = get_acc(output[data.train_mask], data.y, data.train_mask)
		val_acc = get_acc(output[data.val_mask], data.y, data.val_mask)

		if val_acc > best_val:
			best_val = val_acc
			best_test = get_acc(output[data.test_mask], data.y, data.test_mask)
		logger.info("Epoch: {}, train_loss: {:.5f}, train_acc: {:.5f}, vall_acc: {:.5f}".format(
			epoch+1, train_loss, train_acc, val_acc))
	logger.info("training done!")
	logger.info("Best Valid: {}, best Test: {}".format(best_val, best_test))


def get_acc(logits, y_actual, mask):
	"""
	Calculates accuracy
	Parameters
	----------
	logits:		Output of the model
	y_actual: 	Ground truth label of nodes
	mask: 		Indicates the nodes to be considered for evaluation
	Returns
	-------
	accuracy:	Classification accuracy for labeled nodes
	"""
	y_pred = torch.max(logits, dim=1)[1]
	return y_pred.eq(y_actual[mask]).sum().item() / mask.sum().item()


if __name__== "__main__":
	parser = argparse.ArgumentParser(description='Kipf GCN')
	# data type
	parser.add_argument('--data',     	dest="data",    	default='cora', 		help='Dataset to use')

	# train params
	parser.add_argument('--lr',       	dest="lr",             	default=0.01,   type=float,     help='Learning rate')
	parser.add_argument('--epoch',    	dest="max_epochs",     	default=200,    type=int,       help='Max epochs')
	parser.add_argument('--l2',       	dest="l2",             	default=5e-4,   type=float,     help='L2 regularization')
	parser.add_argument('--opt',      	dest="opt",            	default='adam',             	help='Optimizer to use for training')
	parser.add_argument('--seed',     	dest="seed",           	default=1234,   type=int,       help='Seed for randomization')

	# GCN-related params
	parser.add_argument('--gcn_dim',  	dest="gcn_dim",     	default=16,     type=int,       help='GCN hidden dimension')
	parser.add_argument('--drop',     	dest="dropout",        	default=0.5,    type=float,     help='Dropout for full connected layer')

	args = parser.parse_args()

	# set seed
	set_seed(args)

	# set logger
	logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
	
	# set cuda
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.device = device

	# load data
	data, num_classes = load_data(args)
	data.to(args.device)

	# train the model
	model = KipfGCN(data, num_classes, args)
	model.to(args.device)
	train(args, data, model)




