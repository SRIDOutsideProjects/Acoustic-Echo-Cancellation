"""
Script to covert a .h5 weights file of the DTLN model to tf lite.

Example call:
    $python convert_weights_to_tf_light.py -m /name/of/the/model.h5 \
                                              -t name_target 
                              

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 30.06.2020

This code is licensed under the terms of the MIT-license.
"""

from DTLN_model import DTLN_model
import argparse
from pkg_resources import parse_version
import tensorflow as tf
quantization = True
weights_file='weights/DTLN_model.h5'
target_folder='weights/model'
converter = DTLN_model()
converter.create_tf_lite_model(weights_file, 
                                  target_folder,
                                  norm_stft=True,
                                  use_dynamic_range_quant=bool(quantization))
