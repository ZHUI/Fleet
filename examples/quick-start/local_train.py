# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle.fluid as fluid
from nets import mlp
from utils import gen_data, gen_test


input_x = fluid.layers.data(name="x", shape=[784], dtype='float32')
input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

cost, pre = mlp(input_x, input_y)
test_program = fluid.default_main_program().clone(for_test=True)
optimizer = fluid.optimizer.SGD(learning_rate=0.01)
optimizer.minimize(cost)
place = fluid.CPUPlace()

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
step = 301
loss = 1.0
for i in range(step):
    cost_val = exe.run(feed=gen_data(),
                       fetch_list=[cost.name])
    loss = 0.9*loss + 0.1 * cost_val[0]
    print("step%d\tsmmoth%f\tcost=%f" % (i, loss, cost_val[0]))

test_loss = 0.0
acc = 0
i = 0
for data in gen_test():
    i += len(data["y"])
    cost_val, pre_val = exe.run(test_program, feed=data,
                       fetch_list=[cost.name, pre.name])
    test_loss += cost_val
    #print(np.argmax(pre_val, axis=1).reshape(-1))
    #print(data["y"].reshape(-1))
    acc += np.sum(np.argmax(pre_val, axis=1).reshape(-1) == data["y"].reshape(-1))
    print("step%d\tall_loss%f\tall_ccc=%f\trate%f" % (i, test_loss, acc, acc/i))



