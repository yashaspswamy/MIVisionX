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
        #  Params for decoder
        self.decode_width = 500
        self.decode_height = 500
        self.shuffle = True
        self.shard_id = 0
        self.num_shards = 1
        self.path = data_dir

        # Initializing the parameters
        self.set_seed(0)
        self.aug_strength = 0
        #params for contrast
        self.min_param = self.create_int_param(0)
        self.max_param = self.create_int_param(255)
        #param for brightness
        self.alpha_param = self.create_float_param(1)
        self.beta_param = self.create_float_param(10)
        #param for colorTemp
        self.adjustment_param = self.create_int_param(0)
        #param for exposure
        self.shift_param = self.create_float_param(0.0)
        #param for SnPNoise
        self.sdev_param = self.create_float_param(0.0)
        #param for gamma
        self.gamma_shift_param = self.create_float_param(0.0)
        #param for rotate
        self.degree_param = self.create_float_param(0.0)
        #param for lens correction
        self.strength = self.create_float_param(0.0)
        self.zoom = self.create_float_param(1.0)

        self.decode = ops.ImageDecoder()
        self.contrast = ops.Contrast(min_contrast = self.min_param, max_contrast = self.max_param)
        self.brightness = ops.Brightness(alpha =self.alpha_param, beta= self.beta_param)
        self.colorTemp = ops.ColorTemperature(adjustment_value = self.adjustment_param)
        self.exposure = ops.Exposure(exposure=self.shift_param)
        self.noise = ops.SnPNoise(snpNoise=self.sdev_param)
        self.gamma = ops.GammaCorrection(gamma=self.gamma_shift_param)
        self.rotate = ops.Rotate(angle=self.degree_param)
        self.resize = ops.Resize( resize_x=crop, resize_y=crop)
        self.lensCorrection = ops.LensCorrection(strength = self.strength, zoom = self.zoom)
        self.warpaffine = ops.WarpAffine(matrix=[-0.35, 0.35, 0.65, 1.35, -10, 10])
        self.nop = ops.Nop()
        self.copy = ops.Copy()
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.decode.output = self.decode.rali_c_func_call(self._handle,self.path,self.decode_width,self.decode_height,self.shuffle,self.shard_id,self.num_shards,False)
        self.resize.output = self.resize.rali_c_func_call(self._handle,self.decode.output,False)
        self.brightness.output = self.brightness.rali_c_func_call(self._handle,self.decode.output,False)
        self.copy.output = self.copy.rali_c_func_call(self._handle,self.resize.output,True)
        self.exposure.output = self.exposure.rali_c_func_call(self._handle,self.copy.output, False)
        self.contrast.output = self.contrast.rali_c_func_call(self._handle,self.exposure.output,True)
        self.brightness.output = self.brightness.rali_c_func_call(self._handle,self.contrast.output,True)
        self.exposure.output = self.exposure.rali_c_func_call(self._handle,self.contrast.output,True)
        self.colorTemp.output = self.colorTemp.rali_c_func_call(self._handle,self.exposure.output, True)
        self.noise.output = self.noise.rali_c_func_call(self._handle,self.exposure.output,True)
        self.gamma.output = self.gamma.rali_c_func_call(self._handle,self.colorTemp.output,True)
        self.lensCorrection.output = self.lensCorrection.rali_c_func_call(self._handle,self.gamma.output,True)
        self.warpaffine.output = self.warpaffine.rali_c_func_call(self._handle,self.lensCorrection.output,True)
        self.nop.output = self.nop.rali_c_func_call(self._handle, self.gamma.output,True)
        self.rotate.output = self.rotate.rali_c_func_call(self._handle, self.brightness.output,True)

    def updateAugmentationParameter(self, augmentation):
        #values for contrast
        self.aug_strength = augmentation
        min = int(augmentation*100)
        max = 150 + int((1-augmentation)*100)
        self.update_int_param(min, self.min_param)
        self.update_int_param(max, self.max_param)

        #values for brightness
        alpha = augmentation*1.95
        self.update_float_param(alpha, self.alpha_param)

        #values for colorTemp
        adjustment = (augmentation*99) if ((int(augmentation*100)) % 2 == 0) else (-1*augmentation*99)
        adjustment = int(adjustment)
        self.update_int_param(adjustment, self.adjustment_param)

        #values for exposure
        shift = augmentation*0.95
        self.update_float_param(shift, self.shift_param)

        #values for SnPNoise
        sdev = augmentation*0.7
        sdev = 0.06
        self.update_float_param(sdev, self.sdev_param)

        #values for gamma
        gamma_shift = augmentation*5.0
        self.update_float_param(gamma_shift, self.gamma_shift_param)

    def renew_parameters(self):
        curr_degree = self.get_float_value(self.degree_param)
        #values for rotation change
        degree = self.aug_strength * 100
        self.update_float_param(curr_degree+degree, self.degree_param)


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
    crop_size = 400
    pipe = HybridTrainPipe(batch_size=bs, num_threads=nt, device_id=di, data_dir=_image_path, crop=crop_size, rali_cpu=_rali_cpu)
    # pipe.build()
    pipe.verify_graph()
    world_size=1
    imageIterator = RALI_iterator(pipe)

    for i, (image_batch) in enumerate(imageIterator, 0):
        #cv2.imshow('image_batch', cv2.cvtColor(image_batch, cv2.COLOR_RGB2BGR))
        print("Processing image:: ",i)
        pipe.updateAugmentationParameter(0.5+(i/0.1))
        pipe.renew_parameters()
        img = cv2.cvtColor(image_batch, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(i)+'output_img.png', img)
        # cv2.waitKey(10)

if __name__ == '__main__':
    main()
