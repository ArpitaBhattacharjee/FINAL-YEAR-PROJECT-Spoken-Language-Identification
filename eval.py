import mxnet as mx
from mxboard import SummaryWriter


frequency = 44100
# 44100 random samples between -1 and 1
data = mx.random.uniform(low=-1, high=1, shape=(frequency,))
max_abs_val = data.abs().max()
# rescale the data to the range [-1, 1]
data = Data/test/test
with SummaryWriter(logdir='./logs') as sw:
    sw.add_audio(tag='uniform_audio', audio=data, global_step=0)