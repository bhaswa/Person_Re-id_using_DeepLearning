import os,math
import random
import shutil, sys

for t in os.listdir('./m'):
	for p in os.listdir('./m/'+t):
		files1 = os.listdir('./m/'+t+'/'+p)
		random.shuffle(files1)
		
		train_files1 = files1[:int(math.ceil(len(files1)/2))]
		val_files1 = files1[int(math.ceil(len(files1)/2)):]
		os.makedirs('./CNNTrainRes/train/'+p)
		os.makedirs('./CNNTrainRes/val/'+p)
		for f in train_files1:
		  shutil.copyfile('./m/'+t+'/'+p+'/'+f, './CNNTrainRes/train/'+p+'/'+f)
		for f in val_files1:
		  shutil.copyfile('./m/'+t+'/'+p+'/'+f, './CNNTrainRes/val/'+p+'/'+f)
