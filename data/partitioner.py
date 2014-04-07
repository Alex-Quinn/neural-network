import sys
import random

k = 10

orig_data = open(sys.argv[1]).readlines()
data_len = len(orig_data)

shuffled_data = orig_data[:]
random.shuffle(shuffled_data)

partition_size = data_len/k

for i in range(k):
    test_file = open('./data/test_%d' % (i+1), 'w+')
    train_file = open('./data/train_%d' % (i+1), 'w+')
    partition_start = partition_size * i
    for x in range(data_len):
        if x >= partition_start and x < partition_start + partition_size:
            test_file.write(shuffled_data[x])
        else:
            train_file.write(shuffled_data[x])