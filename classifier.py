import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
from skimage.feature import hog

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics.classification import *

import time
import pickle
import os.path as path

# use cv2.imread(),matplotlib.image.imread() return (0~1) when 'png',but
# return (0~255),when 'jpg'
def get_color_hist(img, color_space='RGB',nbins=32, bins_range=(0, 255)):

    if color_space != 'RGB':
        if color_space == 'BGR':
            feature_img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        elif color_space == 'HLS':
            feature_img = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
        elif color_space == 'LUV':
            feature_img = cv2.cvtColor(img,cv2.COLOR_RGB2LUV)
    else:
        feature_img = np.copy(img)

    hist_1 = np.histogram(feature_img[:,:,0],bins=nbins,range=bins_range)
    hist_2 = np.histogram(feature_img[:,:,1],bins=nbins,range=bins_range)
    hist_3 = np.histogram(feature_img[:,:,2],bins=nbins,range=bins_range)

    bin_centers = (hist_1[1][0:(len(hist_2[1])-1)] + hist_3[1][1:])/2
    hist_features = np.concatenate((hist_1[0],hist_2[0],hist_3[0]))

    return hist_features,hist_1, hist_2, hist_3, bin_centers


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):

    if vis == True:
        features, hog_image = hog(img, orient, (pix_per_cell, pix_per_cell),
                                  (cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orient, (pix_per_cell, pix_per_cell), (cell_per_block, cell_per_block), visualise=vis,
                       feature_vector=feature_vec)
        return features

def get_spatial_color_feature(img, size=(32, 32)):
    resized = cv2.resize(img,size)
    return np.ravel(resized)

def get_spatial_gray_feature(img, size=(32, 32)):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray,size)
    return np.ravel(resized)

def extract_feature(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    rgb_feature = get_color_hist(img)
    _,_,l_feature,_,_ = get_color_hist(img, 'HLS')
    spatial_color = get_spatial_color_feature(img, (32, 32))
    spatial_gray = get_spatial_gray_feature(img, (16, 16))
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    hog_r_feature = get_hog_features(hls[:,:,0], orient, pix_per_cell, cell_per_block).ravel()
    hog_g_feature = get_hog_features(hls[:,:,1], orient, pix_per_cell, cell_per_block).ravel()
    hog_b_feature = get_hog_features(hls[:,:,2], orient, pix_per_cell, cell_per_block).ravel()
    combined = np.concatenate((rgb_feature[0],l_feature[0],spatial_gray,hog_r_feature,hog_g_feature,hog_b_feature))

    return combined


color_space = 'HLS'
orient = 15
pix_per_cell =12
cell_per_block = 2
train_img_shape = (64,64)

svm_fname = 'svm.p'


if __name__ == '__main__':

    v_fnames = glob.glob('viehcle/*/*')
    non_v_fnames = glob.glob('non-viehcle/*/*')

    print('viehcle data count:',len(v_fnames))
    print('non viehcle data count:',len(non_v_fnames))
    train_img_shape = mpimg.imread(v_fnames[0]).shape

    print('data image shape: ', train_img_shape)

    sample_count_per_class = 1000

    sampled_v_fname = shuffle(v_fnames)[0:sample_count_per_class]
    sampled_nonv_fname = shuffle(non_v_fnames)[0:sample_count_per_class]

    data = np.concatenate((sampled_v_fname,sampled_nonv_fname))
    label = np.zeros(2*sample_count_per_class,np.int32)
    label[0:sample_count_per_class] = 1


    data ,label = shuffle(data,label)
    feature = []
    for fname in data:
        img = cv2.imread(fname)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        feature.append(extract_feature(img).tolist())

    print('feature count:',len(feature[0]))
    feature = np.array(feature)
    label = np.array(label)
    feature_scaler = MinMaxScaler().fit(feature)


    normalized_feature = feature_scaler.transform(feature)

    train_feature,test_feature,train_label,test_label = train_test_split(normalized_feature,label,test_size=0.3)

    from sklearn.model_selection import GridSearchCV
    svm = SVC()
    params = [{'kernel':['linear'],'C':[0.01,0.05,0.1]}]
    clf = GridSearchCV(svm,params,'precision')

    start = time.time()
    # svm.fit(train_feature, train_label)
    clf.fit(train_feature,train_label)
    print('Training time: ', time.time() - start)

    print('GridSearch best score:' , clf.best_score_)
    print('GridSearch best params:' , clf.best_params_)
    start = time
    predict = clf.predict(test_feature)


    f1 = f1_score(test_label,predict)
    accuracy = accuracy_score(test_label,predict)
    precision = precision_score(test_label,predict)
    recall = recall_score(test_label,predict)
    print('accuracy on test data is ',accuracy)
    print('f1 on test data is ',f1)
    print('precision on test data is ',precision)
    print('recall on test data is ',recall)

    with open(svm_fname,'wb') as file:
        pickle.dump({'model':clf,'scaler':feature_scaler},file)
else:
    if path.exists(svm_fname):
        with open(svm_fname, 'rb') as  file:
            dict = pickle.load(file)
            svm = dict['model']
            feature_scaler = dict['scaler']
    else:
        raise FileNotFoundError('Need to train svm')

