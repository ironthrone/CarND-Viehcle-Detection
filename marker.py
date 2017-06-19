import classifier
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import time


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, tuple(bbox[0]), tuple(bbox[1]), color, thick)
    return imcopy


def search_and_classify(img, x_range=[None, None], y_range=[None, None],
                      window_size=(64, 64), stride=(16, 16),debug=False):
    # check search zone border
    x_start = 0 if x_range[0] == None else x_range[0]
    x_stop = img.shape[1] if x_range[1] == None else x_range[1]
    y_start = 0 if y_range[0] == None else y_range[0]
    y_stop = img.shape[0] if y_range[1] == None else y_range[1]

    x_stride= stride[0]
    y_stride = stride[1]

    x_window_count = (x_stop - x_start - window_size[0]) // x_stride + 1
    y_window_count = (y_stop - y_start - window_size[1]) // y_stride + 1

    windows = []
    test_features = []
    start = time.time()
    for i in range(x_window_count):
        for j in range(y_window_count):
            x = x_start + i * x_stride
            y = y_start + j * y_stride

            window = ((x,y),(x+window_size[0],y+window_size[1]))
            windows.append(window)

            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]],
                                  classifier.train_img_shape[0:2])
            features = classifier.extract_feature(test_img)

            test_features.append( features)
    if debug:
        print('slide time:',time.time()-start)

    start = time.time()
    normolized_test_features = classifier.feature_scaler.transform(np.array(test_features))
    prediction = classifier.svm.predict(normolized_test_features)
    if debug:
        print('predict time:',time.time()-start)
    windows = np.array(windows)[prediction==1].tolist()
    return windows



def heat_and_thresh(img,windows,thresh,debug=False):
    heat = np.zeros(img.shape[0:2])
    for window in windows:
        heat[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1

    heat = cv2.GaussianBlur(heat,(5,5),0)

    heat[heat < thresh] = 0

    labels = label(heat)

    if debug:
        plt.imshow(labels[0], cmap='gray')
        plt.savefig('output_images/labeled_heat.png')
        plt.show()

    heat_windows = []
    for car_number in range(1, labels[1] + 1):
        car_zone = (labels[0] == car_number).nonzero()
        car_zone_x = car_zone[1]
        car_zone_y = car_zone[0]
        heat_windows.append(((np.min(car_zone_x), np.min(car_zone_y)), (np.max(car_zone_x), np.max(car_zone_y))))
    return heat_windows,heat,labels[0]

    # if np.max(car_zone_x) - np.min(car_zone_x) < 30 or np.max(car_zone_y) - np.min(car_zone_y) < 30:
    #     continue

def apply_heat(img,debug=False):
    hot_windows = []
    hot_windows.extend(search_and_classify(img, [400, None], [400, 550], (96, 64), (24, 16),debug=debug))
    hot_windows.extend(search_and_classify(img, [400, None], [400, 600], (128, 96), (36, 24),debug=debug))

    box_img = draw_boxes(img,hot_windows)
    ret = heat_and_thresh(img, hot_windows, 2)
    return ret[0],box_img,ret[1],ret[2]



if __name__ == '__main__':


    for i in range(6):

        image = cv2.imread('test_images/test{}.jpg'.format(i+1))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        start = time.time()
        ret = apply_heat(image)
        for window in ret[0]:
            cv2.rectangle(image, window[0], window[1], (0, 0, 255), 6)

        plt.imsave('output_images/recognized{}.png'.format(i+1), ret[1])
        plt.imsave('output_images/heatmap{}.png'.format(i+1), ret[2])
        plt.imsave('output_images/labeled{}.png'.format(i+1), ret[3])
        expand_2 = np.dstack((ret[2], ret[2], ret[2]))
        expand_2 = (expand_2 * 255 / np.max(expand_2)).astype(np.uint8)

        expand_3 = np.dstack((ret[3], ret[3], ret[3]))
        expand_3 = (expand_3 * 255 / np.max(expand_3)).astype(np.uint8)
        row2 = np.hstack((expand_2, expand_3))

        row1 = np.hstack((image, ret[1]))
        result = np.vstack((row1, row2))
        result =  cv2.resize(result, image.shape[::-1][1:3])
        end = time.time()
        print('Process one image time:',end-start)
        plt.subplot(2,3,i+1)
        plt.savefig('output_images/search_test_images.png')
        plt.imshow(result)

    plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05)
    plt.show()
