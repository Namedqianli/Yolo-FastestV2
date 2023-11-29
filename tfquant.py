import random
from glob import glob

import numpy as np
import tensorflow as tf
import os

from PIL import Image

from tensorflow.python.framework import ops
ops.reset_default_graph()

val_dir='E:\\dataset\\license_plate_detection\\train'
saved_model_dir = '.\saved_model'

class RepresentativeDataset:
    def __init__(self, val_dir, img_size=(640,480), sample_size=200):
        self.val_dir = val_dir
        self.img_size = img_size
        self.sample_size = sample_size
    
    def __call__(self):
        representative_list = random.sample(glob(os.path.join(self.val_dir, "*.jpg")), self.sample_size)
        for image_path in representative_list:
            input_data = Image.open(image_path).resize(self.img_size)
            # input_data = np.expand_dims(input_data, axis=-1)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = input_data.astype('float32')
            input_data /= 255.0  # normalize input data range from 0 to 1.0
            yield [input_data]
representative_dataset_gen = RepresentativeDataset(val_dir)
# 装载预训练模型
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.float32 # or tf.uint8
tflite_quant_model = converter.convert()

with open('tiny_lpd_mo.tflite', 'wb') as w:
    w.write(tflite_quant_model)
