

with open('./ucfTrainTestlist/trainlist01.txt') as trainfile:
  for line in trainfile:
    p, n = line.split(' ')
    p = p[:-4]
    p = '/sailhome/ejwang/UCF101_c3d/' + p
    n = int(n) - 1
  print()
