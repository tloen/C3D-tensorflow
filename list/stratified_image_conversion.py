with open('./all.list', 'r') as all_file:
  filenames = [[] for _ in range(101)]
  for line in all_file:
    v, l = line.split(' ')
    l = int(l)
    filenames[l].append(v)

TRAIN = .16
DEV = .04
TEST = .8

split_id = '%d_%d_%d' % (TRAIN * 100, DEV * 100, TEST * 100)
print(split_id)

with open('./s_train_%s.list' % split_id, 'w') as train_file, \
    open('./s_dev_%s.list' % split_id, 'w') as dev_file, \
    open('./s_test_%s.list' % split_id, 'w') as test_file, \
    open('./s_sortnet_train_%s.list' % split_id, 'w') as sortnet_train_file:
  for l in range(101):
    videos = filenames[l]
    num_videos = len(videos)
    
    b = [0, DEV * num_videos, (TRAIN + DEV) * num_videos, num_videos]
    b = [round(boundary) for boundary in b]
    dev = videos[b[0]:b[1]]
    train = videos[b[1]:b[2]]
    test = videos[b[2]:b[3]]
    for vid in train:
      print(vid, l, file=train_file)
      print(vid, l, file=sortnet_train_file)
    for vid in dev:
      print(vid, l, file=dev_file)
    for vid in test:
      print(vid, l, file=test_file)
      print(vid, l, file=sortnet_train_file)

     
  
  

  
