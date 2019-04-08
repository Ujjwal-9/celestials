import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import  *
import time
import cv2
data_dirs = ['Pipes', 'Piano', 'Adirondack', 'Jadeplant', 'Playroom', 'Vintage', 'Recycle', 'PlaytableP', 'PianoL', 'MotorcycleE', 'Motorcycle', 'Shelves', 'Teddy', 'Playtable', 'ArtL']
# data_dirs = ['Adirondack']
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

def rms_blockmatching(imageleft, imageright, directory,size):

  searchrange = 50

  im0 = plt.imread(imageleft)
  im1 = plt.imread(imageright)

  disparity_map = np.zeros((im1.shape[0],im1.shape[1]))
  
  for i in range(im1.shape[0]):
    for j in range(im1.shape[1]):
      right_temp = im1[int(i-size/2):int(i+size/2+1),int(j-size/2):int(j+size/2+1)]
      
      min = 1000000
      for p in range(searchrange):
        match = im0[int(i-size/2):int(i+size/2+1),int(j-size/2-p):int(j+size/2-p+1)]

        if right_temp.shape == match.shape:

          temp = rms_diff(right_temp, match)
          
          if temp<min:
            min = temp 
            minx = i - p
        else:
          break
        
    
      distance = np.abs(i - minx)
      disparity_map[i][j] = distance
  
  cv2.imwrite('output_rms_abs.png', disparity_map)
  return disparity_map



def abs_blockmatching(imageleft, imageright, directory, size):

  searchrange = 50

  im0 = plt.imread(imageleft)
  im1 = plt.imread(imageright)

  disparity_map = np.zeros((im1.shape[0],im1.shape[1]))
  
  for i in range(im1.shape[0]):
    for j in range(im1.shape[1]):
      right_temp = im1[int(i-size/2):int(i+size/2+1),int(j-size/2):int(j+size/2+1)]
      
      min = 1000000
      for p in range(searchrange):
        match = im0[int(i-size/2):int(i+size/2+1),int(j-size/2-p):int(j+size/2-p+1)]

        if right_temp.shape == match.shape:

          temp = abs_diff(right_temp, match)
          
          if temp<min:
            min = temp 
            minx = i - p
        else:
          break
        
    
      distance = np.abs(i - minx)
      disparity_map[i][j] = distance
  
  cv2.imwrite('output_rms_abs.png', disparity_map)
  return disparity_map

def abs_diff(I1, I2):
  return np.sum((np.absolute(I1-I2)))

def rms_diff(I1, I2):
    return np.sqrt(np.sum((I1 - I2)**2))

def rms_abs_blockmatching(imageleft, imageright, directory,size):

  searchrange = 50

  im0 = plt.imread(imageleft)
  im1 = plt.imread(imageright)

  disparity_map = np.zeros((im1.shape[0],im1.shape[1]))
  
  for i in range(im1.shape[0]):
    for j in range(im1.shape[1]):
      right_temp = im1[int(i-size/2):int(i+size/2+1),int(j-size/2):int(j+size/2+1)]
      
      min = 1000000
      for p in range(searchrange):
        match = im0[int(i-size/2):int(i+size/2+1),int(j-size/2-p):int(j+size/2-p+1)]

        if right_temp.shape == match.shape:

          temp = (rms_diff(right_temp, match) + abs_diff(right_temp, match))/2
          
          if temp<min:
            min = temp 
            minx = i - p
        else:
          break
        
    
      distance = np.abs(i - minx)
      disparity_map[i][j] = distance

  cv2.imwrite('Data/trainingQ/' + directory+'/output_rms_abs.png', disparity_map)
  return disparity_map

def census_blockmatching(imageleft, imageright, directory, size):
  

  def transform(image, window_size=3):
      half_window_size = window_size // 2

      image = cv2.copyMakeBorder(image, top=half_window_size, left=half_window_size, right=half_window_size, bottom=half_window_size, borderType=cv2.BORDER_CONSTANT, value=0)
      rows, cols = image.shape
      census = np.zeros((rows - half_window_size * 2, cols - half_window_size * 2), dtype=np.uint8)
      center_pixels = image[half_window_size:rows - half_window_size, half_window_size:cols - half_window_size]

      offsets = [(row, col) for row in range(half_window_size) for col in range(half_window_size) if not row == half_window_size + 1 == col]
      for (row, col) in offsets:
          census = (census << 1) | (image[row:row + rows - half_window_size * 2, col:col + cols - half_window_size * 2] >= center_pixels)
      return census

  def column_cost(left_col, right_col):
      """
      Column-wise Hamming edit distance
      """
      return np.sum(np.unpackbits(np.bitwise_xor(left_col, right_col), axis=1), axis=1).reshape(left_col.shape[0], left_col.shape[1])

  def cost(left, right, window_size=3, disparity=0):
      """
      Compute cost difference between left and right grayscale images. Disparity value can be used to assist with evaluating stereo
      correspondence.
      """
      ct_left = transform(left, window_size=window_size)
      ct_right = transform(right, window_size=window_size)
      rows, cols = ct_left.shape
      C = np.full(shape=(rows, cols), fill_value=0)
      for col in range(disparity, cols):
          C[:, col] = column_cost(
              ct_left[:, col:col + 1],
              ct_right[:, col - disparity:col - disparity + 1]
          ).reshape(ct_left.shape[0])
      return C

  def norm(image):
      return cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)


  resize_pct = 0.5
  ndisp = 330
  ndisp *= resize_pct
  left = cv2.imread(imageleft, 0)
  right = cv2.imread(imageright, 0)
  left = cv2.resize(left, dsize=(0,0), fx=resize_pct, fy=resize_pct)
  right = cv2.resize(right, dsize=(0, 0), fx=resize_pct, fy=resize_pct)

  window_size = size
  ct_left = norm(transform(left, window_size))
  ct_right = norm(transform(right, window_size))

  ct_costs = []
  for exponent in range(0, 6):
      import math
      disparity = int(ndisp / math.pow(2, exponent))
      print(math.pow(2, exponent), disparity)
      ct_costs.append(norm(cost(left, right, window_size, disparity)))

  cv2.imwrite('Data/trainingQ/' + directory+ '/output_census.png', np.vstack(np.hstack([ct_left, ct_right])))
  return ct_left

if __name__ == '__main__':
    for directory in data_dirs:
        imageleft = 'Data/trainingQ/' + directory + '/im0.png'
        imageright = 'Data/trainingQ/' + directory + '/im1.png'
        path = 'Data/trainingQ/' + directory
        # stereoSSD(imageleft, imageright, directory)
        census_blockmatching(imageleft, imageright, directory, 7)
        
