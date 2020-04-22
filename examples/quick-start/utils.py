import os
import sys
import time
import numpy as np
import paddle.dataset.mnist

gpu_id = int(os.getenv("FLAGS_selected_gpus", "0"))
np.random.seed(gpu_id)

def gen_data_fake():
    return {"x": np.random.random(size=(128, 784)).astype('float32'),
            "y": np.random.randint(2, size=(128, 1)).astype('int64')}


def gen_data_bak():
    return {"x": np.random.random(size=(128, 32)).astype('float32'),
            "y": np.random.randint(2, size=(128, 1)).astype('int64')}

def gen_datai_2():
    x = np.random.random(size=(128, 32)).astype('float32')
    return {"x": x,
            "y": (np.sum(x[:,0:16], axis=1)>10).astype('int64')}

def data_iter(reader):
    bs = 64
    count = 1
    data = []
    label = []
    for l in reader():
        if count % bs ==0:
            # print(np.array(data).astype("float32"))
            x = np.array(data).astype("float32").reshape((-1,784))
            y = np.array(label).astype('int64').reshape(-1,1)
            yield {"x":x, "y": y}
            data = []
            label = []
        #print(type(l[0]), l[0].shape)
        data += list(l[0])
        label.append(l[1])
        count += 1


reader_train = paddle.dataset.mnist.train()
reader_test = paddle.dataset.mnist.test()

iters = data_iter(reader_train)
iters_test = data_iter(reader_test)

def iters_shuffle():
    all_data =  []
    for data in iters:
        all_data.append(data)
    print(len(all_data))
    idx =  np.random.permutation(len(all_data))
    print(idx[0:20])
    global run_index
    run_index = 0
    def get():
        global run_index
        while True:
            run_index +=1
            yield all_data[idx[run_index]]
    return get

shuff = ( iters_shuffle() )()

def gen_data():
    return shuff.__next__()

def gen_test():
    for data in iters_test:
        yield data
