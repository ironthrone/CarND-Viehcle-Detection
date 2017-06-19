from moviepy.editor import VideoFileClip
import marker
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
class Tracker(object):
    def __init__(self,hist_count=5):
        self.last_frames = []
        self.hist_max_count = hist_count
        self.current_frame =0

    def track(self,img,debug=False):
        self.current_frame +=1
        ret = marker.apply_heat(img)
        windows = ret[0]

        labeled = ret[3]
        if len(windows) > 0:
            self.add_to_hist(Frame(windows,ret[1],ret[2]))
        if len(self.last_frames) >0:
            all_window_array = np.concatenate([frame.windows for frame in self.last_frames]).reshape(-1,2,2).astype(np.int32)

            thresh = min(len(self.last_frames),self.hist_max_count//2)

            after_hist_windows = marker.heat_and_thresh(img,all_window_array,thresh)
            labeled = after_hist_windows[2]
            for window in after_hist_windows[0]:
                cv2.rectangle(img,window[0],window[1],(0,0,255),6)
        result = img

        if debug:
            row1 = np.hstack((img, ret[1]))

            heatmap = np.dstack((ret[2], ret[2], ret[2]))
            heatmap = (heatmap * 255 / np.max(heatmap)).astype(np.uint8)

            labeled = np.dstack((labeled, labeled, labeled))
            labeled = (labeled * 255 / np.max(labeled)).astype(np.uint8)
            row2 = np.hstack((heatmap, labeled))

            result = np.vstack((row1, row2))
            result = cv2.resize(result,img.shape[::-1][1:3])
            if len(self.last_frames)>6:
                _,axes = plt.subplots(6,2,figsize=(8,12))
                for i in range(6):

                    j = i +2
                    axes[i, 0].imshow(self.last_frames[-j].boxed)
                    axes[i, 1].imshow(self.last_frames[-j].heatmap,cmap='gray')
                plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05)
                plt.savefig('output_images/hist_frame_boxed_heatmap.png')
                plt.imsave('output_images/current_frame_labeled.png',labeled)
                plt.imsave('output_images/current_frame_boxed.png',img)

        return result



    def add_to_hist(self,frame):
        if len(frame.windows) > 0:
            self.last_frames.append(frame)
            if len(self.last_frames) > self.hist_max_count:
                self.last_frames = self.last_frames[-self.hist_max_count:-1]

class Frame():
    def __init__(self,windows,boxed,heatmap):
        self.boxed = boxed
        self.heatmap = heatmap
        self.windows = windows


src_fname = 'project_video.mp4'
out_fname='{}_marked_{}.mp4'.format(src_fname.split('.')[0],int(time.time()))

tracker = Tracker(10)
clip = VideoFileClip(src_fname)\
    # .subclip(0,10)
start = time.time()

marked = clip.fl_image(tracker.track)

marked.write_videofile(out_fname,audio=False)

end = time.time()
print('Cost time: {}'.format(end-start))