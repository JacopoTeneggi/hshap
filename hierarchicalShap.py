import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def display_cropped_images(images, score):
  mean = np.array([0.5, 0.5, 0.5])
  sd = np.array([0.5, 0.5, 0.5])
  fig, axs = plt.subplots(4,4, figsize=(15,15))
  for i in range(4):
    for j in range(4): 
      sample_image = images[4*i + j].numpy().transpose(1,2,0)
      im = sample_image*sd + mean
      axs[i,j].imshow(im)
      axs[i,j].set_title("#%d score:%f " %(4*i + j, score[4*i + j]))

def construct_subsets(im, start = (0,0), region_size = (None, None), background = None):
  if (region_size[0] == None or region_size[1] == None):
    region_size = im.numpy().shape[1:3]

  middle = (start[0] + int(0.5*region_size[0]), start[1] + int(0.5*region_size[1]))
  end = (start[0] + region_size[0], start[1] + region_size[1])
  
  subsets_size = [16]
  image_size = []
  for dim in im.shape: 
    subsets_size.append(dim)
    image_size.append(dim)
  subsets = torch.zeros(subsets_size)

  quadrant_size = [image_size[0], int(region_size[0]/2), int(region_size[1]/2)]
  if (background == None): 
    background = torch.ones(image_size)


  # removing 0 features 
  im1234 = background.clone()
  im1234[:,start[0]:end[0],start[1]:end[1]] = im[:,start[0]:end[0],start[1]:end[1]]
  # removing 1 feature
  im234 = im1234.clone()
  im234[:,start[0]:middle[0],start[1]:middle[1]] = background[:,start[0]:middle[0],start[1]:middle[1]]
  im134 = im1234.clone()
  im134[:,start[0]:middle[0],middle[1]:end[1]] = background[:,start[0]:middle[0],middle[1]:end[1]]
  im124 = im1234.clone()
  im124[:,middle[0]:end[0],start[1]:middle[1]] = background[:,middle[0]:end[0],start[1]:middle[1]]
  im123 = im1234.clone()
  im123[:,middle[0]:end[0],middle[1]:end[1]] = background[:,middle[0]:end[0],middle[1]:end[1]]
  # removing 2 features
  im34 = im234.clone()
  im34[:,start[0]:middle[0],middle[1]:end[1]] = background[:,start[0]:middle[0],middle[1]:end[1]]
  im24 = im234.clone()
  im24[:,middle[0]:end[0],start[1]:middle[1]] = background[:,middle[0]:end[0],start[1]:middle[1]]
  im23 = im234.clone()
  im23[:,middle[0]:end[0],middle[1]:end[1]] = background[:,middle[0]:end[0],middle[1]:end[1]]
  im14 = im134.clone()
  im14[:,middle[0]:end[0],start[1]:middle[1]] = background[:,middle[0]:end[0],start[1]:middle[1]]
  im13 = im134.clone()
  im13[:,middle[0]:end[0],middle[1]:end[1]] = background[:,middle[0]:end[0],middle[1]:end[1]]
  im12 = im123.clone()
  im12[:,middle[0]:end[0],start[1]:middle[1]] = background[:,middle[0]:end[0],start[1]:middle[1]]
  # remoing 3 features 
  im4 = im34.clone()
  im4[:,middle[0]:end[0],start[1]:middle[1]] = background[:,middle[0]:end[0],start[1]:middle[1]]
  im3 = im34.clone()
  im3[:,middle[0]:end[0],middle[1]:end[1]] = background[:,middle[0]:end[0],middle[1]:end[1]]
  im2 = im24.clone()
  im2[:,middle[0]:end[0],middle[1]:end[1]] = background[:,middle[0]:end[0],middle[1]:end[1]]
  im1 = im14.clone()
  im1[:,middle[0]:end[0],middle[1]:end[1]] = background[:,middle[0]:end[0],middle[1]:end[1]]
  # removing 4 
  im_ = background
  
  subsets[0] = im1234
  subsets[1] = im234
  subsets[2] = im134
  subsets[3] = im124
  subsets[4] = im123
  subsets[5] = im34
  subsets[6] = im24
  subsets[7] = im23
  subsets[8] = im14
  subsets[9] = im13
  subsets[10] = im12
  subsets[11] = im4
  subsets[12] = im3
  subsets[13] = im2
  subsets[14] = im1
  subsets[15] = im_

  return subsets

def subsetScores(net, sub, label):
  outputs = net(sub)

  score = np.zeros(16)
  score[0] = outputs[0, label]
  score[1] = outputs[1, label]
  score[2] = outputs[2, label]
  score[3] = outputs[3, label]
  score[4] = outputs[4, label]
  score[5] = outputs[5, label]
  score[6] = outputs[6, label]
  score[7] = outputs[7, label]
  score[8] = outputs[8, label]
  score[9] = outputs[9, label]
  score[10] = outputs[10, label]
  score[11] = outputs[11, label]
  score[12] = outputs[12, label]
  score[13] = outputs[13, label]
  score[14] = outputs[14, label]
  score[15] = outputs[15, label]
  return score

def constructShapMap(score):
  phi1 = (score[14]-score[15])/4 + (score[8]-score[11] + score[9]-score[12] + score[10]-score[13])/12 + (score[2]-score[5] + score[3]-score[6] + score[4]-score[7])/12 + (score[0]-score[1])/4
  phi2 = (score[13]-score[15])/4 + (score[6]-score[11] + score[7]-score[12] + score[10]-score[14])/12 + (score[1]-score[5] + score[3]-score[8] + score[4]-score[9])/12 + (score[0]-score[2])/4
  phi3 = (score[12]-score[15])/4 + (score[5]-score[11] + score[9]-score[12] + score[7]-score[13])/12 + (score[2]-score[8] + score[1]-score[6] + score[4]-score[10])/12 + (score[0]-score[3])/4
  phi4 = (score[11]-score[15])/4 + (score[6]-score[14] + score[5]-score[12] + score[6]-score[13])/12 + (score[1]-score[9] + score[3]-score[10] + score[1]-score[9])/12 + (score[0]-score[4])/4
  shap_map = np.array([[phi1, phi2], [phi3, phi4]])
  return shap_map

def get_salient_regions(image, shap_map,  start, quadrant_size): 
  srs = []
  for i in range(len(shap_map)): 
    for j in range(len(shap_map)): 
      if (shap_map[i,j] > 5): 
        x = start[0] + i*quadrant_size[0]
        y = start[1] + j*quadrant_size[1]
        srs.append( ((x,y),(quadrant_size)) )
  return srs

def display_salient(im, srs):
  plt.figure()
  sample_image = im.numpy().transpose(1,2,0)
  mean = np.array([0.5, 0.5, 0.5])
  sd = np.array([0.5, 0.5, 0.5])
  image = sample_image*sd + mean
  N_points = 101
  plt.imshow(image)
  for sr in srs:
    start = sr[0]
    quadrant_size = sr[1]
    plt.plot(np.linspace(start[1], start[1]+quadrant_size[1], N_points), start[0]*np.ones(N_points), 'r')
    plt.plot(np.linspace(start[1], start[1]+quadrant_size[1], N_points), (start[0]+quadrant_size[0])*np.ones(N_points), 'r')
    plt.plot(start[1]*np.ones(N_points), np.linspace(start[0], start[0]+quadrant_size[0], N_points), 'r')
    plt.plot((start[1]+quadrant_size[1])*np.ones(N_points), np.linspace(start[0], start[0]+quadrant_size[0], N_points), 'r')

def do_all(net, im, label, strt, rgs, debug = False):
  images_final = construct_subsets(im, strt, rgs)
  score = subsetScores(net, images_final, label)
  sm = constructShapMap(score)

  if (debug): 
    display_cropped_images(images_final, score) 
    f = plt.figure()
    sns.heatmap(sm)
    plt.title("Shap values of each quadrant");
  
  quad = (int(rgs[0]/2), int(rgs[1]/2))
  srs = get_salient_regions(im, sm, strt, quad)
  return srs
