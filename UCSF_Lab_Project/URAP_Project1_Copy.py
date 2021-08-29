#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import os
import numpy as np
os.getcwd()


# In[3]:


ls


# In[4]:


cd Mammogram_Training_DICOM_files


# In[5]:


cd ..


# In[6]:


project_dir = os.getcwd()


# In[7]:


path_to_files = []

counter = 0
for root, dirs, files in os.walk(project_dir):
    for file in files:
        if file.endswith(".dcm"):
            counter += 1
            path_to_files.append(os.path.join(root, file))


# In[8]:


ls


# In[9]:


import shutil
shutil.move(path_to_files[0], "/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/unpacked_training_files/"+path_to_files[0].split('/')[-1])


# In[10]:


for i in range(1, len(path_to_files)):
    new_file_name = str(i)+path_to_files[i].split('/')[-1]
    unpack_dir = "/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/unpacked_training_files/"
    shutil.move(path_to_files[i], unpack_dir+new_file_name)
    


# In[11]:


unpack_test_dir = "/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/unpacked_testing_files/"


# In[12]:


get_ipython().system('ls')


# In[13]:


os.getcwd()


# In[14]:


current_path = '/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/unpacked_training_files/'
files = os.listdir(current_path)

for file in files:
    d = len(file)-10
    os.rename(os.path.join(current_path, file), os.path.join(current_path, 'train_'+file[:d]+'.dcm'))
    


# In[15]:


paths_to_test_files = []
testing_dir = '/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/testing_data_DICOM_folders/'

counter = 0
for root, dirs, files in os.walk(testing_dir):
    for file in files:
        if file.endswith(".dcm"):
            counter += 1
            paths_to_test_files.append(os.path.join(root, file))


# In[16]:


for i in range(0, len(paths_to_test_files)):
    new_file_name = 'train_'+str(i)+'.dcm'
    unpack_test_dir = "/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/unpacked_testing_files/"
    shutil.move(paths_to_test_files[i], unpack_test_dir+new_file_name)


# In[17]:


current_path = '/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/unpacked_testing_files/'
files = os.listdir(current_path)

for file in files:
    os.rename(os.path.join(current_path, file), os.path.join(current_path, 'test_'+file[5:]))


# In[133]:


training_file_list = os.listdir('/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/unpacked_training_files/')
os.chdir('/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1')


# In[217]:


# NOTE: This cell was created AFTER the initial running of the entire script. 
# This cell was necessay because in my second run, I ran into problems importing packages,
# mainly because I wasn't using the Anaconda version of Jupyter Notebook (which caused file mis-location)

# Rename the files 
# I am doing this due to a Jupyter Notebook error made it unable to import packages I had installed
# The files are existing but with the wrong names

# This part was necessary to re-order the list

project_folder = '/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/'
for file in os.listdir('/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/unpacked_training_files/'):
    if any(ch.isdigit() for ch in file):
        label = int(''.join(list(filter(str.isdigit, file))))
        os.rename(project_folder+'unpacked_training_files/'+file, project_folder+'unpacked_training_files/'+'train'+str(label)+'.dcm')
    

# Finally, reorder the files
final_training_files = []
for file in os.listdir(project_folder+'unpacked_training_files/'):
    if file.split('.')[0].isalnum():
        final_training_files.append(file)
        
final_training_files = sorted(final_training_files, key=lambda x:int(''.join(list(filter(str.isdigit, x)))))


# In[304]:


import numpy as np
import pydicom

# Test display image
ex_image = final_training_files[67]
img_ex_data = pydicom.dcmread(training_folder+ex_image)
plt.imshow(img_ex_data.pixel_array)


# In[283]:


import numpy as np

# library to open .png file
from PIL import Image 

import pydicom

from skimage import data, color
from skimage.transform import rescale, resize


# Part 4) function below:
# This function will check the extension of a file (either DICOM or PNG), 
# read the file, and then convert it into a NumpyZ compressed file 

def image_file_to_numpyz(filename, label_category, parent_folder, destination_name, imgsize):
    # Check file extension
    extension = filename.split('.')[-1]
    
    if extension == 'dcm':
        # Read data
        data = pydicom.dcmread(parent_folder+filename)
        dicom_array = data.pixel_array
        
        # Resize the Numpy array
        resized_array = resize(dicom_array, (imgsize[0], imgsize[1]),
                       anti_aliasing=True)
        
        # save NumpZ file
        np.savez_compressed(destination_name, a=resized_array, b=label_category)
        
    elif extension == 'png':
        # Read Data
        data = Image.open(parent_folder+filename)
        png_array = np.asarray(data)
        
        # Resize the Numpy array
        resized_array = resize(png_array, (imgsize[0], imgsize[1]),
                       anti_aliasing=True)
        
        # save NumpZ file
        np.savez_compressed(destination_name, a=resized_array, b=label_category)
        
    elif extension == 'jpg':
        # Read Data
        data = Image.open(parent_folder+filename)
        jpg_array = np.asarray(data)
        
        # Resize the Numpy array
        resized_array = resize(jpg_array, (imgsize[0], imgsize[1]),
                       anti_aliasing=True)
        
        # save NumpZ file
        np.savez_compressed(destination_name, a=resized_array, b=label_category)


# In[233]:


# Let's read in the csv description file for Mass_Training!
project_folder = '/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/'

import pandas as pd
training_dframe = pd.read_csv(project_folder+'mass_case_description_train_set.csv', header='infer')


# In[234]:


training_dframe.iloc[0:,:]


# In[235]:


sample_training = training_dframe.iloc[0:201,:]
sample_training


# In[236]:


binary_sample_training = sample_training.copy()


# In[237]:


mapping = {1:'NOT_DENSE', 2:'NOT_DENSE', 3:'DENSE', 4:'DENSE'}


# In[238]:


binary_sample_training['breast_density'] = binary_sample_training['breast_density'].map(mapping)


# In[239]:


binary_sample_training.head()


# In[240]:


os.mkdir('Training_npy_images')


# In[243]:


os.chdir(project_folder+'Training_npy_images')
os.getcwd()


# In[252]:


project_folder


# In[258]:


density_labels = binary_sample_training['breast_density']
density_labels[0]


# In[282]:


final_training_files[0].split('.')[0]+'NPZ'


# In[284]:


# This piece of code basically goes through each image .dcm file, 
# extracts the image matrix, reshapes it to (299x299), then deposits 
# the matrix into a list

Size = (299,299)
location_to_save = project_folder+'Training_npy_images/'

for file, label in zip(final_training_files, density_labels):
    try:
        new_name = file.split('.')[0]+'NPZ'
        image_file_to_numpyz(file, label, project_folder+'unpacked_training_files/', location_to_save+new_name, Size)
        print('Image file'+ file[5:-4] +' saved as NumpyZ!')
        
    except ValueError as e:
        print('Image {} Data not readable!'.format(file[5:-4]))
        continue


# In[287]:


os.chdir('/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/')
os.getcwd()


# In[288]:


os.mkdir('Testing_npy_images')


# In[295]:


# Let's read in the csv description file for Mass_Testing!
project_folder = '/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/'

# get the test data dataframe
import pandas as pd
testing_dframe = pd.read_csv(project_folder+'mass_case_description_test_set.csv', header='infer')
sample_testing = testing_dframe.iloc[0:177,:]


# Create the binary-label (for density) dataframe
# Rename the column items to "DENSE/NOT_DENSE" binary labels
binary_sample_testing = sample_testing.copy()
mapping = {1:'NOT_DENSE', 2:'NOT_DENSE', 3:'DENSE', 4:'DENSE'}
binary_sample_testing['breast_density'] = binary_sample_testing['breast_density'].map(mapping)


#Isolate the labels series
test_density_labels = binary_sample_testing['breast_density']

test_density_labels.head(6)


# In[293]:


# This cell is for renaming the testing image files

# NOTE: This cell was created AFTER the initial running of the entire script. 
# This cell was necessay because in my second run, I ran into problems importing packages,
# mainly because I wasn't using the Anaconda version of Jupyter Notebook (which caused file mis-location)

# Rename the files 
# I am doing this due to a Jupyter Notebook error made it unable to import packages I had installed
# The files are existing but with the wrong names

# This part was necessary to re-order the list

project_folder = '/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/'
for file in os.listdir('/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/unpacked_testing_files/'):
    if any(ch.isdigit() for ch in file):
        label = int(''.join(list(filter(str.isdigit, file))))
        os.rename(project_folder+'unpacked_testing_files/'+file, project_folder+'unpacked_testing_files/'+'test'+str(label)+'.dcm')
    

# Finally, reorder the files
final_testing_files = []
for file in os.listdir(project_folder+'unpacked_testing_files/'):
    if file.split('.')[0].isalnum():
        final_testing_files.append(file)
        
final_testing_files = sorted(final_testing_files, key=lambda x:int(''.join(list(filter(str.isdigit, x)))))


# In[300]:


# This code is the same as saving the .npz training files - I'm just applying it to test-files now

# This piece of code basically goes through each image .dcm file, 
# extracts the image matrix, reshapes it to (299x299), then deposits 
# the matrix into a list

Size = (299,299)
location_to_save = project_folder+'Testing_npy_images/'

for file, label in zip(final_testing_files, test_density_labels):
    try:
        new_name = file.split('.')[0]+'NPZ'
        image_file_to_numpyz(file, label, project_folder+'unpacked_testing_files/', location_to_save+new_name, Size)
        print('Image file'+ file[4:-4] +' saved as NumpyZ!')
        
    except ValueError as e:
        print('Image {} Data not readable!'.format(file[4:-4]))
        continue


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


# Alright, time to build a Neural Network!

# Before we start building the "ResNet architecture", let's start learning
# by buildinga the 'simple version' of a convolutional neural network (2D CNN)

# It works by taking the image data, using a filter on the image (called 'convolving')
# in order to detect edges, dark and bright spots, and other image characteristics
# Then these pieces of data are combined, pooled (meaning finding the max. value in
# a filter window), and then put through a dense connected layer.

# This system of Conv. -> Max Pooling -> Dense Layer is repeated 2-3 times,
# and finally we will have a Dropout layer (to correct for overfitting).

# First, we need to import Tensorflow and Keras
# Then import all the layers needed for the ConvNet 


# In[3]:


import os
project_folder = '/Users/rabeya/URAP/UCSF_Lab/Practice_Project_1/'
os.listdir(project_folder+'Training_npy_images/')


# In[4]:



# Let's first work with our training set (and do a validation holdout set on it)
# Load the NumpyZ files
loaded_training_images_list = []
loaded_training_labels = []
training_file_folder = os.listdir(project_folder+'Training_npy_images/')
for file in training_file_folder:
    loaded = np.load(project_folder+'Training_npy_images/'+file)
    X = loaded['a']
    y_label = loaded['b']
    loaded_training_images_list.append(X)
    loaded_training_labels.append(y_label)


# In[5]:


import pandas as pd
labels_series = pd.Series([label.item(0) for label in loaded_training_labels])
labels_series.head()


# In[6]:


# Now, we need to convert the y_labels into a numeric array
# We can code "DENSE" as 1, and "NOT_DENSE" as 0
label_mapping = {'DENSE':1, 'NOT_DENSE':0}
y_numeric = np.array(labels_series.map(label_mapping))


# In[7]:


y_numeric[:10]


# In[8]:


loaded_training_images_list[0].shape


# In[9]:


plt.imshow(loaded_training_images_list[0])


# In[10]:


# Convert the iamge list and label list to numpy arrays
X = np.array(loaded_training_images_list)
y = y_numeric


# In[11]:


len(X)


# In[12]:


X[34].shape


# In[13]:


img0_copies = [X[0], X[0], X[0]]
np.stack(img0_copies, axis=-1).shape


# In[14]:


X_stacked = np.array([np.stack([img, img, img], axis=-1) for img in X])


# In[15]:


X_stacked[0].shape


# In[16]:


len(X_stacked)


# In[17]:


# Let's make the training and validation subsets (from the training data itself!)
X_train, X_valid = X_stacked[:10][:5], X_stacked[:10][5:]
y_train, y_valid = y[:10][:5], y[:10][5:]


# In[18]:


X_valid.shape


# In[19]:


# Now that we have a numerical encoding of our breast density
# labels, let's start building the 2D ConvNet model!
# First, import the packages

from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.applications.resnet50 import preprocess_input, decode_predictions


# In[21]:


# Then, build the model!

model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(Dense(1, activation='sigmoid'))
model.layers[0].trainable = False
model.summary()


# In[ ]:


y_train


# In[ ]:


# Compile the CNN model
from tensorflow.python.keras import optimizers

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)


# In[ ]:


preds_proba = model.predict_proba(X_valid)
prds = model.predic


# In[ ]:





# In[ ]:





# In[ ]:


# Let's get some results and visualization through confusion matrix!
# Edits of my own code from the past

score_realdata =


# In[256]:


# Display the corresponding confusion matrix for the cancer data (on the test set)

labels_cancer = LR_real.classes_
y_predictions = LR_real.predict(real_X.values)
cm_cancer = confusion_matrix(y_predictions, real_y)

print('Classification accuracy: {}%'.format(int(100*score_realdata)))
print("False Negative Rate: {}%".format(round(100*2/35, 2)))

plt.figure(figsize=(10,7))
plt.title('2 = Benign Tumor, 4 = Malignant Tumor')
sns.heatmap(cm_cancer, annot=True, xticklabels=labels_cancer, yticklabels=labels_cancer)
plt.xlabel('Logit C.F. Matrix Prediction')
plt.ylabel('Truth')

