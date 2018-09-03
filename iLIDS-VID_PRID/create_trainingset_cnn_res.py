import os,math
import random
import shutil, sys

for p in os.listdir('./i-LIDS-VID/images/cam1'):
    files1 = os.listdir('./i-LIDS-VID/sequences/cam1/'+p)
    files2 = os.listdir('./i-LIDS-VID/sequences/cam2/'+p)
    random.shuffle(files1)
    random.shuffle(files2)
    train_files1 = files1[:int(math.ceil(len(files1)/2))]
    val_files1 = files1[int(math.ceil(len(files1)/2)):]
    train_files2 = files2[:int(math.ceil(len(files2)/2))]
    val_files2 = files2[int(math.ceil(len(files2)/2)):]
    os.makedirs('./CNNTrainRes/train/'+p)
    os.makedirs('./CNNTrainRes/val/'+p)
    for f in train_files1:
      shutil.copyfile('./i-LIDS-VID/sequences/cam1/'+p+'/'+f, './CNNTrainRes/train/'+p+'/'+f)
    for f in val_files1:
      shutil.copyfile('./i-LIDS-VID/sequences/cam1/'+p+'/'+f, './CNNTrainRes/val/'+p+'/'+f)
    for f in train_files2:
      shutil.copyfile('./i-LIDS-VID/sequences/cam2/'+p+'/'+f, './CNNTrainRes/train/'+p+'/'+f)
    for f in val_files2:
      shutil.copyfile('./i-LIDS-VID/sequences/cam2/'+p+'/'+f, './CNNTrainRes/val/'+p+'/'+f)
