import pandas as pd
import cv2
import numpy as np
import sys
import os
import ast
import random
import shutil
final_df = pd.DataFrame(columns=['plant_count'])
test_df = pd.DataFrame(columns=['plant_count'])
validate_df = pd.DataFrame(columns=['plant_count'])

if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('test'):
    os.makedirs('test')
if not os.path.exists('validate'):
    os.makedirs('validate')

dirs = [i for i in range(1, 18)]
dirs.pop(11)

filecounter = 0
for dirnum in dirs:
    dirname = 'dataset' + str(dirnum).zfill(2)
    
    fd = open(os.path.join(dirname,'gt.txt'))

    data = []
    for line in fd:
        line = line.strip()
        imgnum = line.split(' ')[1]
        x = line.split(' ')[2]
        if (x == 'X'):
            continue
        y = line.split(' ')[3]

        imagename = imgnum.zfill(10)+'.png'
        if not os.path.exists(os.path.join(dirname,imagename)):
            continue
        image = cv2.imread(os.path.join(dirname,imagename))

        h = image.shape[0]
        x = int(x)/2
        y = h - int(y)/2
        data.append([imagename, y, x])

        #print(imagename)
        #print(x, y)

    random.shuffle(data)
    for i in range(len(data)):
        item = data[i]
        imagename = item[0]
        y = item[1]
        x = item[2]

        newname = str(filecounter).zfill(10) + '.png'
        filecounter += 1
        df = pd.DataFrame(data=[[1, [[y, x]]]],
        index=[newname],
                          columns=['plant_count', 'plant_locations'])
        if (i < len(data)*0.8):
            shutil.move(os.path.join(dirname,imagename), 'train/'+newname)
            final_df = final_df.append(df)
        elif (i < len(data)*0.9):
            shutil.move(os.path.join(dirname,imagename), 'test/'+newname)
            test_df = test_df.append(df)
        else:
            shutil.move(os.path.join(dirname,imagename), 'validate/'+newname)
            validate_df = validate_df.append(df)

final_df.to_csv('train.csv')
shutil.move('train.csv', 'train')
test_df.to_csv('test.csv')
shutil.move('test.csv', 'test')
validate_df.to_csv('validate.csv')
shutil.move('validate.csv', 'validate')

