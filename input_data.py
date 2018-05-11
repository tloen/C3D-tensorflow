# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
from sortnet import embedding_generator

def get_frames_data(filename, num_frames_per_clip=16, uniform_sampling):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  for parent, dirnames, filenames in os.walk(filename):
    if(len(filenames)<num_frames_per_clip):
      return [], s_index
    filenames = sorted(filenames)
    if uniform_sampling:
      files = random.sample(range(len(filenames)), num_frames_per_clip)
      files = sorted(files)
    else:
      s_index = random.randint(0, len(filenames) - num_frames_per_clip)
      files = range(s_index, s_index + num_frames_per_clip)
    for i in files:
      image_name = str(filename) + '/' + str(filenames[i])
      img = Image.open(image_name)
      img_data = np.array(img)
      ret_arr.append(img_data)
  return ret_arr, s_index

def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False, embeddings=False, uniform_sampling):
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
  embed = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
  np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
  big_mean = np.repeat(np_mean, 2, 1)
  big_mean = np.repeat(big_mean, 2, 2)
  '''
  if random_embeddings:
    # fix the random embeddings
    np.random.seed(1729)
    # embeddings = np.random.randn(len(lines), 16, 10) # MOVE EMBEDDING_DIM TO THIS FILE
    embeddings = np.ones((len(lines), 16, 10)) # MOVE EMBEDDING_DIM TO THIS FILE
  else:
    print('NOTIMPLEMENTED')
  '''
  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = list(range(len(lines)))
    random.seed(time.time())
    random.shuffle(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
  for index in video_indices:
    if(batch_index>=batch_size):
      next_batch_start = index
      break
    line = lines[index].strip('\n').split()
    dirname = line[0]
    tmp_label = line[1]
    if not shuffle:
      pass
      # print("Loading a video clip from {}...".format(dirname))
    tmp_data, s_index = get_frames_data(dirname, num_frames_per_clip, uniform_sampling=uniform_sampling)
    # video is at [s_index, s_index + num_frames_per_clip)
    img_datas = [];
    embed_imgs = []
    if(len(tmp_data)!=0):
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if(img.width>img.height):
          scale = float(crop_size)/float(img.height)
          img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
          scale = float(crop_size)/float(img.width)
          img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
        img_datas.append(img)
        
        # generate embeddings
        if embeddings:
          img = Image.fromarray(tmp_data[j].astype(np.uint8))
          if(img.width>img.height):
            scale = float(224)/float(img.height)
            img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), 224))).astype(np.float32)
          else:
            scale = float(224)/float(img.width)
            img = np.array(cv2.resize(np.array(img),(224, int(img.height * scale + 1)))).astype(np.float32)
          crop_x = int((img.shape[0] - 224)/2)
          crop_y = int((img.shape[1] - 224)/2)
          embed_img = img[crop_x:crop_x+224, crop_y:crop_y+224,:] - big_mean[j]
          embed_imgs.append(embed_img)
      data.append(img_datas)
      if embeddings:
        embed.append(embedding_generator.get_embeddings(np.array(embed_imgs)))
      else:
        embed.append(np.ones((num_frames_per_clip, 10)))
      label.append(int(tmp_label))
      # embed.append(embeddings[index, :, :])
      batch_index = batch_index + 1
      read_dirnames.append(dirname)

  # pad (duplicate) data/label if less than batch_size
  valid_len = len(data)
  pad_len = batch_size - valid_len
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      label.append(int(tmp_label))
      embed.append(embeddings[video_indices[-1], :, :])
  np_arr_data = np.array(data).astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)
  np_arr_embed = np.array(embed).astype(np.float32)
  return np_arr_data, np_arr_label, np_arr_embed, next_batch_start, read_dirnames, valid_len

