import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import  *
import time

data_dirs = ['Adirondack']
def stereoMatchSAD(left_img, right_img, directory):
    # Load in both images, assumed to be RGBA 8bit per channel images
    left_img = Image.open(left_img)
    left = np.asarray(left_img)
    right_img = Image.open(right_img)
    right = np.asarray(right_img)

    # Initial squared differences
    w, h = left_img.size  # assume that both images are same size
    sd = np.empty((w, h), np.uint8)
    sd.shape = h, w

    # SSD support window (kernel)
    win_ssd = np.empty((w, h), np.uint16)
    win_ssd.shape = h, w
    
    # Depth (or disparity) map
    depth = np.empty((w, h), np.uint8)
    depth.shape = h, w

    # Minimum ssd difference between both images
    min_ssd = np.empty((w, h), np.uint16)
    min_ssd.shape = h, w
    min_ssd.fill(65535)
    
    max_offset = 20
    offset_adjust = 255 / max_offset 

    # Create ranges now instead of per loop
    y_range = range(h)
    x_range = range(w)
    x_range_ssd = range(w)

    # u and v support window
    window_range = range(-3, 4) # 6x6
    start = time.time()
    # Main loop....
    for offset in tqdm(range(max_offset)):
        # Create initial image of squared differences between left and right image at the current offset
        for y in y_range:
            for x in x_range_ssd:
                if x - offset > 0:
                    diff = abs(int(left[y, x, 0]) - int(right[y, x - offset, 0]))
                    sd[y, x] = diff

    # Sum the absolute differences over a support window at this offset
        for y in y_range:
            for x in x_range:
                sum_sd = 0
                for i in window_range:
                    for j in window_range:
                        # TODO: replace this expensive check by surrounding image with buffer / padding
                        if (-1 < y + i < h) and (-1 < x + j < w):
                            sum_sd += sd[y + i, x + j]

                # Store the sum in the window SSD image
                win_ssd[y, x] = sum_sd

        # Update the min ssd diff image with this new data
        for y in y_range:
            for x in x_range:
                # Is this new windowed SSD pixel a better match?
                if win_ssd[y, x] < min_ssd[y, x]:
                    # If so, store it and add to the depth map      
                    min_ssd[y, x] = win_ssd[y, x]
                    depth[y, x] = offset * offset_adjust
    # print time.time()-start
    # Convert to PIL and save it
    Image.fromarray(depth).save('Data/trainingQ/' + directory + '/depth_SAD.png')
def stereoMatchSSD(left_img, right_img, directory):
    # Load in both images, assumed to be RGBA 8bit per channel images
    left_img = Image.open(left_img)
    left = np.asarray(left_img)
    right_img = Image.open(right_img)
    right = np.asarray(right_img)

    # Initial squared differences
    w, h = left_img.size  # assume that both images are same size
    sd = np.empty((w, h), np.uint8)
    sd.shape = h, w

    # SSD support window (kernel)
    win_ssd = np.empty((w, h), np.uint16)
    win_ssd.shape = h, w
    
    # Depth (or disparity) map
    depth = np.empty((w, h), np.uint8)
    depth.shape = h, w

    # Minimum ssd difference between both images
    min_ssd = np.empty((w, h), np.uint16)
    min_ssd.shape = h, w
    min_ssd.fill(65535)
    
    max_offset = 20
    offset_adjust = 255 / max_offset 

    # Create ranges now instead of per loop
    y_range = range(h)
    x_range = range(w)
    x_range_ssd = range(w)

    # u and v support window
    window_range = range(-3, 4) # 6x6
    start = time.time()
    # Main loop....
    for offset in tqdm(range(max_offset)):
        # Create initial image of squared differences between left and right image at the current offset
        for y in y_range:
            for x in x_range_ssd:
                if x - offset > 0:
                    diff = int(left[y, x, 0]) - int(right[y, x - offset, 0])
                    sd[y, x] = diff * diff

    # Sum the squared differences over a support window at this offset
        for y in y_range:
            for x in x_range:
                sum_sd = 0
                for i in window_range:
                    for j in window_range:
                        # TODO: replace this expensive check by surrounding image with buffer / padding
                        if (-1 < y + i < h) and (-1 < x + j < w):
                            sum_sd += sd[y + i, x + j]

                # Store the sum in the window SSD image
                win_ssd[y, x] = sum_sd

        # Update the min ssd diff image with this new data
        for y in y_range:
            for x in x_range:
                # Is this new windowed SSD pixel a better match?
                if win_ssd[y, x] < min_ssd[y, x]:
                    # If so, store it and add to the depth map      
                    min_ssd[y, x] = win_ssd[y, x]
                    depth[y, x] = offset * offset_adjust
    # print time.time()-start
    # Convert to PIL and save it
    Image.fromarray(depth).save('Data/trainingQ/' + directory + '/depth_SSD.png')

# def cencus_transform()

if __name__ == '__main__':
    for directory in data_dirs:
        imageleft = 'Data/trainingQ/' + directory + '/im0.png'
        imageright = 'Data/trainingQ/' + directory + '/im1.png'
        stereoMatchSSD(imageleft, imageright, directory)
