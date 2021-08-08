import os
import pandas as pd 

BASE_DIR='/d/project2/Input/'
train_folder=BASE_DIR+'train/'
test_folder=BASE_DIR+'test/'
val_folder=BASE_DIR+'val/'
train_annotation1=train_folder+'with_mask/'
train_annotation2=train_folder+'without_mask/'
test_annotation1=test_folder+'with_mask/'
test_annotation2=test_folder+'without_mask/'
val_annotation1=val_folder+'with_mask/'
val_annotation2=val_folder+'without_mask/'

files_in_train_annotated1 = sorted(os.listdir(train_annotation1))
files_in_train_annotated2 = sorted(os.listdir(train_annotation2))
files_in_test_annotated1 = sorted(os.listdir(test_annotation1))
files_in_test_annotated2 = sorted(os.listdir(test_annotation2))
files_in_val_annotated1 = sorted(os.listdir(val_annotation1))
files_in_val_annotated2 = sorted(os.listdir(val_annotation2))

#images_train1=[i for i in files_in_train_annotated1]

df_train_1 = pd.DataFrame()
df_train_1['images']=[train_annotation1 + str(x) for x in files_in_train_annotated1]
df_train_1['labels']='with_mask'

##print(df_train_1['images'])

#images_train2=[i for i in files_in_train_annotated2]
df_train_2 = pd.DataFrame()
df_train_2['images']=[train_annotation2 + str(x) for x in files_in_train_annotated2]
df_train_2['labels']='without_mask'
df_train=pd.concat([df_train_1,df_train_2])
df_train.to_csv('train.csv', header=None,index=None)

#images_test1=[i for i in files_in_test_annotated1]
df_test_1 = pd.DataFrame()
df_test_1['images']=[test_annotation1 + str(x) for x in files_in_test_annotated1]
df_test_1['labels']='with_mask'
#df_test_1.to_csv('val1.csv', header=None)

#images_test2=[i for i in files_in_test_annotated2]
df_test_2 = pd.DataFrame()
df_test_2['images']=[test_annotation2 + str(x) for x in files_in_test_annotated2]
df_test_2['labels']='without_mask'
df_test=pd.concat([df_test_1 , df_test_2])
df_test.to_csv('test.csv', header=None,index=None)

#images_val1=[i for i in files_in_val_annotated1]
df_val_1 = pd.DataFrame()
df_val_1['images']=[val_annotation1 + str(x) for x in files_in_val_annotated1]
df_val_1['labels']='with_mask'

#images_val2=[i for i in files_in_val_annotated2]
df_val_2 = pd.DataFrame()
df_val_2['images']=[val_annotation2 + str(x) for x in files_in_val_annotated2]
df_val_2['labels']='without_mask'
df_val=pd.concat([df_val_1 , df_val_2])
df_val.to_csv('val.csv', header=None,index=None)    