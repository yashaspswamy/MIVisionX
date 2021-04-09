import numpy as np
import cv2
from enum import Enum
from amd.rali.plugin.pytorch import RALIClassificationIterator
from amd.rali.plugin.pytorch import RALI_iterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import sys
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, rali_cpu = True):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)
        world_size = 1
        local_rank = 0
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder()
        self.contrast = ops.Contrast(min_contrast = 15, max_contrast = 40, preserve = True)
        self.brightness = ops.Brightness(alpha = 0.5, beta= 15, preserve = True)
        self.colorTemp = ops.ColorTemperature(adjustment_value = 70, preserve = True)
        self.exposure = ops.Exposure(exposure=0.5, preserve = True)
        self.noise = ops.SnPNoise(snpNoise=0.05, preserve = True)
        self.gamma = ops.GammaCorrection(gamma=0.7, preserve = True)
        self.rotate = ops.Rotate(angle=30, preserve = True)
        self.resize = ops.Resize( resize_x=crop, resize_y=crop, preserve = True)
        # self.cmnp = ops.CropMirrorNormalize(device="cpu",
        #                                     output_dtype=types.FLOAT,
        #                                     output_layout=types.NCHW,
        #                                     crop=(crop, crop),
        #                                     image_type=types.RGB,
        #                                     mirror = 0,
        #                                     mean=[0, 0, 0],
        #                                     std=[1,1,1])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.resize(images)
        images = self.contrast(images)
        images = self.brightness(images)
        images = self.colorTemp(images)
        images = self.exposure(images)
        images = self.noise(images)
        images = self.gamma(images)
        output = self.rotate(images)
        # output = self.cmnp(images, mirror=rng)
        return [output, self.labels]

def main():
    if  len(sys.argv) < 4:
        print ('Please pass image_folder cpu/gpu batch_size')
        exit(0)
    _image_path = sys.argv[1]
    if(sys.argv[2] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    bs = int(sys.argv[3])
    nt = 1
    di = 0
    crop_size = 224
    pipe = HybridTrainPipe(batch_size=bs, num_threads=nt, device_id=di, data_dir=_image_path, crop=crop_size, rali_cpu=_rali_cpu)
    pipe.build()
    world_size=1
    imageIterator = RALI_iterator(pipe)

    for i, (image_batch, image_tensor) in enumerate(imageIterator, 0):
        #cv2.imshow('image_batch', cv2.cvtColor(image_batch, cv2.COLOR_RGB2BGR))
        img = cv2.cvtColor(image_batch, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(i)+'output_img.png', img)
        # cv2.waitKey(10)

if __name__ == '__main__':
    main()
