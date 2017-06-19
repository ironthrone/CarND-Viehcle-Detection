import glob
import matplotlib.pyplot as plt
import classifier
import cv2


def explore():
    car_fnames = glob.glob('test_images/car*')
    noncar_fnames = glob.glob('test_images/nocar*')

    orient = 9
    pix_per_cell = 8
    cell_per_block = 2


    for i in range(min(len(car_fnames),len(noncar_fnames))):

        car = cv2.imread(car_fnames[i])
        noncar = cv2.imread(noncar_fnames[i])
        # noncar = cv2.imread('test_images/project_video_screen.png')
        car = cv2.cvtColor(car,cv2.COLOR_BGR2RGB)
        noncar = cv2.cvtColor(noncar,cv2.COLOR_BGR2RGB)

        car_gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
        noncar_gray = cv2.cvtColor(noncar, cv2.COLOR_BGR2GRAY)

        car_hls = cv2.cvtColor(car, cv2.COLOR_RGB2HLS)
        noncar_hls = cv2.cvtColor(noncar, cv2.COLOR_RGB2HLS)

        car_col_rgb = classifier.get_color_hist(car)
        noncar_col_rgb = classifier.get_color_hist(noncar)

        car_col_hls = classifier.get_color_hist(car,'HLS')
        noncar_col_hls = classifier.get_color_hist(noncar,'HLS')

        # color feature
        figure,axes = plt.subplots(2,3,figsize=(15,9))
        figure.suptitle('Color')
        axes[0, 0].imshow(car)
        axes[0, 0].set_ylabel('Car')
        axes[0,1].set_title('rgb')
        axes[0,1].plot(car_col_rgb[0])
        axes[0,2].set_title('hls')
        axes[0,2].plot(car_col_hls[0])
        axes[1, 0].set_ylabel('Not Car')
        axes[1, 0].imshow(noncar)
        axes[1,1].plot(noncar_col_rgb[0])
        axes[1,2].plot(noncar_col_hls[0])
        plt.tight_layout(0.5,0.3,0.3)
        plt.subplots_adjust(top=0.9)
        plt.savefig('output_images/explorer_color.png')

        size = (16,16)
        car_spatial = classifier.get_spatial_color_feature(car,size)
        noncar_spatial = classifier.get_spatial_color_feature(noncar,size)

        car_spatial_gray = classifier.get_spatial_color_feature(car_gray,size)
        noncar_spatial_gray = classifier.get_spatial_color_feature(noncar_gray,size)

        # spatial feature
        _, axes = plt.subplots(2, 3, figsize=(15, 9))
        plt.suptitle('Spatial')

        axes[0, 0].imshow(car)
        axes[0,0].set_ylabel('Car')
        axes[0, 1].set_title('RGB spatial')
        axes[0, 1].plot(car_spatial)
        axes[0, 2].set_title('Gray spatial')
        axes[0, 2].plot(car_spatial_gray)
        axes[1,0].set_ylabel('Not Car')
        axes[1, 0].imshow(noncar)
        axes[1, 1].plot(noncar_spatial)
        axes[1, 2].plot(noncar_spatial_gray)
        plt.tight_layout(0.5, 0.3, 0.3)
        plt.subplots_adjust(top=0.9)
        plt.savefig('output_images/explorer_spatial.png')

        _,car_r_hog = classifier.get_hog_features(car[:, :, 0], orient, pix_per_cell, cell_per_block, True)
        _,car_g_hog = classifier.get_hog_features(car[:, :, 1], orient, pix_per_cell, cell_per_block, True)
        _,car_b_hog = classifier.get_hog_features(car[:, :, 2], orient, pix_per_cell, cell_per_block, True)

        _,noncar_r_hog = classifier.get_hog_features(noncar[:, :, 0], orient, pix_per_cell, cell_per_block, True)
        _,noncar_g_hog = classifier.get_hog_features(noncar[:, :, 1], orient, pix_per_cell, cell_per_block, True)
        _,noncar_b_hog = classifier.get_hog_features(noncar[:, :, 2], orient, pix_per_cell, cell_per_block, True)


        _,car_h_hog = classifier.get_hog_features(car_hls[:, :, 0], orient, pix_per_cell, cell_per_block, True)
        _,car_l_hog = classifier.get_hog_features(car_hls[:, :, 1], orient, pix_per_cell, cell_per_block, True)
        _,car_s_hog = classifier.get_hog_features(car_hls[:, :, 2], orient, pix_per_cell, cell_per_block, True)

        _,noncar_h_hog = classifier.get_hog_features(noncar_hls[:, :, 0], orient, pix_per_cell, cell_per_block, True)
        _,noncar_l_hog = classifier.get_hog_features(noncar_hls[:, :, 1], orient, pix_per_cell, cell_per_block, True)
        _,noncar_s_hog = classifier.get_hog_features(noncar_hls[:, :, 2], orient, pix_per_cell, cell_per_block, True)


        _, car_gray_hog = classifier.get_hog_features(car_gray, orient, pix_per_cell, cell_per_block, True)

        _, noncar_gray_hog = classifier.get_hog_features(noncar_gray, orient, pix_per_cell, cell_per_block, True)

        # hog feature
        _,axes = plt.subplots(4,4)
        plt.suptitle('Hog')
        axes[0, 0].imshow(car)
        axes[0,0].set_ylabel('car')

        axes[0,1].set_title('r channel')
        axes[0,1].imshow(car_r_hog,cmap='gray')


        axes[0,2].set_title('g channel')
        axes[0,2].imshow(car_g_hog,cmap='gray')

        axes[0,3].set_title('b channel')
        axes[0,3].imshow(car_b_hog,cmap='gray')

        axes[1, 0].imshow(noncar)
        axes[1,0].set_ylabel('not car')
        axes[1,1].imshow(noncar_r_hog,cmap='gray')
        axes[1,2].imshow(noncar_g_hog,cmap='gray')
        axes[1,3].imshow(noncar_b_hog,cmap='gray')


        axes[2,0].set_title('h channel')
        axes[2,0].imshow(car_h_hog,cmap='gray')
        axes[2,1].set_title('l channel')
        axes[2,1].imshow(car_l_hog,cmap='gray')

        axes[2,2].set_title('s channel')
        axes[2,2].imshow(car_s_hog,cmap='gray')
        axes[2,3].set_title('gray channel')
        axes[2,3].imshow(car_gray_hog,cmap='gray')

        axes[3,0].imshow(noncar_h_hog,cmap='gray')
        axes[3,1].imshow(noncar_l_hog,cmap='gray')
        axes[3,2].imshow(noncar_s_hog,cmap='gray')
        axes[3, 3].imshow(noncar_gray_hog,cmap='gray')
        plt.tight_layout(1,0.5)
        plt.subplots_adjust(top=0.9)
        plt.savefig('output_images/explorer_hog.png')
        plt.show()


# import marker
# import numpy as np
# img = mpimg.imread('test_images/test4.jpg')
#
# hot_windows = marker.search_and_classify(img)
# draw_img = marker.draw_boxes(img, hot_windows)
# plt.imshow(draw_img)
# plt.show()

explore()