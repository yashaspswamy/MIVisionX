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
        self.shuffle = False
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
        self.alpha_param = self.create_float_param(0.5)
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
        #param for snow
        self.snow = self.create_float_param(0.1)
        #param for rain
        self.rain = self.create_float_param(0.1)
        self.rain_width = self.create_int_param(2)
        self.rain_height = self.create_int_param(15)
        self.rain_transparency = self.create_float_param(0.25)
        #param for blur
        self.blur = self.create_int_param(5)
        #param for jitter
        self.kernel_size = self.create_int_param(3)
        #param for hue
        self.hue = self.create_float_param(1.0)
        #param for saturation
        self.saturation = self.create_float_param(1.5)
        #param for warp affine
        self.affine_matrix = [0.35,0.25,0.75,1,1,1]
        #param for fog
        self.fog = self.create_float_param(0.35)
        #param for vignette
        self.vignette = self.create_float_param(50)
        #param for flip
        self.flip_axis = self.create_int_param(0)
        #param for blend
        self.blend = self.create_float_param(0.5)

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
        self.warpaffine = ops.WarpAffine(matrix=self.affine_matrix)
        self.slice = ops.Slice(crop_h=250,crop_w=250,crop_pos_x = 0,crop_pos_y=0)
        self.snow = ops.Snow(snow=self.snow)
        self.rain = ops.Rain(rain=self.rain, rain_width = self.rain_width, rain_height = self.rain_height, rain_transparency =self.rain_transparency)
        self.blur = ops.Blur(blur = self.blur)
        self.jitter =ops.Jitter(kernel_size = self.kernel_size)
        self.hue = ops.Hue(hue=self.hue)
        self.saturation = ops.Saturation(saturation = self.saturation)
        self.fisheye = ops.FishEye()
        self.vignette = ops.Vignette(vignette = self.vignette)
        self.fog = ops.Fog(fog=self.fog)
        self.pixelate = ops.Pixelate()
        self.flip = ops.Flip(flip=self.flip_axis)
        self.blend = ops.Blend(blend = self.blend)
        self.nop = ops.Nop()
        self.copy = ops.Copy()
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.decode.output = self.decode.rali_c_func_call(self._handle,self.path,self.decode_width,self.decode_height,self.shuffle,self.shard_id,self.num_shards,False)
        self.slice.output = self.slice.rali_c_func_call(self._handle, self.decode.output, False)
        self.resize.output = self.resize.rali_c_func_call(self._handle,self.decode.output,True)
        self.brightness.output = self.brightness.rali_c_func_call(self._handle,self.resize.output,False)
        # self.fog.output = self.fog.rali_c_func_call(self._handle,self.resize.output,True)
        # self.fisheye.output = self.fisheye.rali_c_func_call(self._handle,self.resize.output,True)
        # self.vignette.output = self.vignette.rali_c_func_call(self._handle,self.resize.output,True)
        # self.pixelate.output = self.pixelate.rali_c_func_call(self._handle,self.resize.output,True)
        self.flip.output = self.flip.rali_c_func_call(self._handle,self.resize.output,False)
        self.blend = self.blend.rali_c_func_call(self._handle,self.resize.output,self.flip.output,True)
        # self.blur.output = self.blur.rali_c_func_call(self._handle,self.resize.output,True)
        # self.jitter.output = self.jitter.rali_c_func_call(self._handle,self.resize.output,True)
        # self.hue.output = self.hue.rali_c_func_call(self._handle,self.resize.output,True)
        # self.saturation.output = self.saturation.rali_c_func_call(self._handle,self.resize.output,True)
        # self.snow.output = self.snow.rali_c_func_call(self._handle,self.brightness.output,True)
        # self.gamma.output = self.gamma.rali_c_func_call(self._handle,self.brightness.output,True)
        # self.rain.output = self.rain.rali_c_func_call(self._handle,self.brightness.output,True)
        # self.copy.output = self.copy.rali_c_func_call(self._handle,self.resize.output,True)
        # self.exposure.output = self.exposure.rali_c_func_call(self._handle,self.copy.output, False)
        # self.contrast.output = self.contrast.rali_c_func_call(self._handle,self.exposure.output,True)
        # self.brightness.output = self.brightness.rali_c_func_call(self._handle,self.contrast.output,True)
        # self.exposure.output = self.exposure.rali_c_func_call(self._handle,self.contrast.output,True)
        # self.colorTemp.output = self.colorTemp.rali_c_func_call(self._handle,self.exposure.output, True)
        # self.noise.output = self.noise.rali_c_func_call(self._handle,self.exposure.output,True)
        # self.lensCorrection.output = self.lensCorrection.rali_c_func_call(self._handle,self.gamma.output,True)
        # self.warpaffine.output = self.warpaffine.rali_c_func_call(self._handle,self.resize.output,True)
        # self.nop.output = self.nop.rali_c_func_call(self._handle, self.gamma.output,True)
        # self.rotate.output = self.rotate.rali_c_func_call(self._handle, self.brightness.output,True)

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
        pipe.updateAugmentationParameter(0.5+(i/0.01))
        pipe.renew_parameters()
        img = cv2.cvtColor(image_batch, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(i)+'output_img.png', img)
        # cv2.waitKey(10)

if __name__ == '__main__':
    main()
