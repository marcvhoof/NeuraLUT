#  This file is part of NeuraLUT.
#
#  NeuraLUT is a derivative work based on LogicNets,
#  which is licensed under the Apache License 2.0.
#
#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import math
import time
import torch
import torch.nn as nn
from torch import Tensor

from brevitas.core.quant import QuantType
from brevitas.core.quant import RescalingIntQuant, ClampedBinaryQuant
from brevitas.core.scaling import ScalingImplType
import brevitas.nn as bnn

#########################################
# Free Function Caching Implementations #
#########################################

# Module-level caches for free functions
_get_int_state_space_cache = {}
_get_float_state_space_cache = {}

# TODO: Put this inside an abstract base class
def get_int_state_space(bits: int, signed: bool, narrow_range: bool, is_cuda: bool):
    start = int(0 if not signed else (-(2 ** (bits - 1)) + int(narrow_range)))  # calculate the minimum value in the range
    end = int(start + 2 ** (bits) - int(narrow_range))  # calculate the maximum of the range
    state_space = torch.as_tensor(range(start, end))
    if is_cuda:
        return state_space.cuda()
    return state_space

def get_int_state_space_with_cache(bits: int, signed: bool, narrow_range: bool, is_cuda: bool, debug=False):
    key = (bits, signed, narrow_range, is_cuda)
    if key in _get_int_state_space_cache:
        if debug:
            print("DEBUG: get_int_state_space_with_cache cache hit for key:", key)
        return _get_int_state_space_cache[key]
    if debug:
        start_time = time.time()
    result = get_int_state_space(bits, signed, narrow_range, is_cuda)
    _get_int_state_space_cache[key] = result
    if debug:
        elapsed_time = time.time() - start_time
        print("DEBUG: get_int_state_space_with_cache computed new value for key:", key,
              "in {:.6f} seconds".format(elapsed_time))
    return result

# TODO: Put this inside an abstract base class
def get_float_state_space(
    bits: int,
    scale_factor: float,
    signed: bool,
    narrow_range: bool,
    quant_type: QuantType,
    is_cuda: bool,
):
    if quant_type == QuantType.INT:
        bin_state_space = get_int_state_space(bits, signed, narrow_range, is_cuda)
    elif quant_type == QuantType.BINARY:
        bin_state_space = torch.as_tensor([-1.0, 1.0])
    if is_cuda:
        bin_state_space = bin_state_space.cuda()
    state_space = scale_factor * bin_state_space
    return state_space

def get_float_state_space_with_cache(
    bits: int,
    scale_factor: float,
    signed: bool,
    narrow_range: bool,
    quant_type: QuantType,
    is_cuda: bool,
    debug=False,
):
    key = (bits, scale_factor, signed, narrow_range, quant_type, is_cuda)
    if key in _get_float_state_space_cache:
        if debug:
            print("DEBUG: get_float_state_space_with_cache cache hit for key:", key)
        return _get_float_state_space_cache[key]
    if debug:
        start_time = time.time()
    result = get_float_state_space(bits, scale_factor, signed, narrow_range, quant_type, is_cuda)
    _get_float_state_space_cache[key] = result
    if debug:
        elapsed_time = time.time() - start_time
        print("DEBUG: get_float_state_space_with_cache computed new value for key:", key,
              "in {:.6f} seconds".format(elapsed_time))
    return result

#############################################
# Class Based Caching Implementations Below #
#############################################

# TODO: Add an abstract class with a specific interface which all brevitas-based classes inherit from?
class QuantBrevitasActivation(nn.Module):
    def __init__(self, brevitas_module, pre_transforms: list = [], post_transforms: list = []):
        super(QuantBrevitasActivation, self).__init__()
        self.brevitas_module = brevitas_module
        self.pre_transforms = nn.ModuleList(pre_transforms)
        self.post_transforms = nn.ModuleList(post_transforms)
        self.is_bin_output = False
        # Initialize caches for memoization of methods
        self._bin_str_from_int_cache = {}
        self._state_space_cache = {}
        self._bin_state_space_cache = {}
        self._bin_str_from_float_cache = {}

    # TODO: Move to a base class
    # TODO: Move the string templates to verilog.py
    def get_bin_str_from_float(self, x, is_cuda):
        quant_type = self.get_quant_type()
        _, bits = self.get_scale_factor_bits()
        if quant_type == QuantType.INT:
            tensor_quant = self.brevitas_module.act_quant_proxy.fused_activation_quant_proxy.tensor_quant
            narrow_range = tensor_quant.int_quant.narrow_range
            signed = tensor_quant.int_quant.signed
            offset = 2 ** (bits - 1) - int(narrow_range) if signed else 0
            for idx, value in enumerate(self.get_state_space(is_cuda)):
                if math.isclose(self.get_state_space(is_cuda)[idx], x, rel_tol=1e-03):
                    return f"{int(self.get_bin_state_space(is_cuda)[idx] + offset):0{int(bits)}b}"
            raise Exception("Value not found in state space")
        elif quant_type == QuantType.BINARY:
            return f"{int(x):0{int(bits)}b}"
        else:
            raise Exception("Unknown quantization type: {}".format(quant_type))

    def get_bin_str_from_float_with_cache(self, x, is_cuda, debug=False):
        """
        Cached version of get_bin_str_from_float.
        Caches the result based on (x, is_cuda, scale_factor, bits, quant_type).
        """
        scale_factor, bits = self.get_scale_factor_bits()
        quant_type = self.get_quant_type()
        key = (float(x), is_cuda, scale_factor, bits, quant_type)
        if key in self._bin_str_from_float_cache:
            if debug:
                print("DEBUG: get_bin_str_from_float_with_cache cache hit for key:", key)
            return self._bin_str_from_float_cache[key]
        if debug:
            start_time = time.time()
        result = self.get_bin_str_from_float(x, is_cuda)
        self._bin_str_from_float_cache[key] = result
        if debug:
            elapsed_time = time.time() - start_time
            print("DEBUG: get_bin_str_from_float_with_cache computed new value for key:", key,
                  "in {:.6f} seconds".format(elapsed_time))
        return result

    def get_bin_str_from_int(self, x, is_cuda, debug=False):
        # Only measure timing if debug is enabled
        if debug:
            start_time = time.time()
            print("DEBUG: Entering get_bin_str_from_int")
            print("    Input value:", x)
            print("    Input type:", type(x))
            if isinstance(x, torch.Tensor):
                try:
                    print("    Tensor length:", x.numel())
                except Exception as e:
                    print("    Could not determine tensor length:", e)

        quant_type = self.get_quant_type()
        scale_factor, bits = self.get_scale_factor_bits()
        if quant_type == QuantType.INT:
            tensor_quant = self.brevitas_module.act_quant_proxy.fused_activation_quant_proxy.tensor_quant
            narrow_range = tensor_quant.int_quant.narrow_range
            signed = tensor_quant.int_quant.signed
            offset = 2 ** (bits - 1) - int(narrow_range) if signed else 0
            if int(x) - x != 0:
                raise Exception("Value is not an integer, either run lut_inference first or change function to get_bin_str_from_float")
            result = f"{int(x + offset):0{int(bits)}b}"
        elif quant_type == QuantType.BINARY:
            result = f"{int(x):0{int(bits)}b}"
        else:
            raise Exception("Unknown quantization type: {}".format(quant_type))

        if debug:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("DEBUG: Exiting get_bin_str_from_int")
            print("    Output binary string:", result)
            print("    Output type:", type(result))
            print("    Output string length:", len(result))
            print("    Time taken in get_bin_str_from_int: {:.6f} seconds".format(elapsed_time))
            
        return result

    def get_bin_str_from_int_with_cache(self, x, is_cuda, debug=False):
        """
        Cached version of get_bin_str_from_int.
        Checks if the input (x, is_cuda) has been processed before.
        Also reports timing if debug is enabled.
        """
        if debug:
            start_time = time.time()
        key = (int(x), is_cuda)
        if key in self._bin_str_from_int_cache:
            result = self._bin_str_from_int_cache[key]
            if debug:
                end_time = time.time()
                print("DEBUG: get_bin_str_from_int_with_cache cache hit for key:", key)
                print("    Time taken for cached retrieval: {:.6f} seconds".format(end_time - start_time))
            return result

        result = self.get_bin_str_from_int(x, is_cuda, debug)
        self._bin_str_from_int_cache[key] = result
        if debug:
            end_time = time.time()
            print("DEBUG: get_bin_str_from_int_with_cache cache miss for key:", key)
            print("    Time taken for computation: {:.6f} seconds".format(end_time - start_time))
        return result

    def get_state_space(self, is_cuda):
        quant_type = self.get_quant_type()
        scale_factor, bits = self.get_scale_factor_bits()
        if quant_type == QuantType.INT:
            tensor_quant = self.brevitas_module.act_quant_proxy.fused_activation_quant_proxy.tensor_quant
            narrow_range = tensor_quant.int_quant.narrow_range
            signed = tensor_quant.int_quant.signed
            state_space = get_float_state_space(bits, scale_factor, signed, narrow_range, quant_type, is_cuda)
        elif quant_type == QuantType.BINARY:
            state_space = scale_factor * torch.tensor([-1, 1])
        else:
            raise Exception("Unknown quantization type: {}".format(quant_type))
        return self.apply_post_transforms(state_space)

    def get_state_space_with_cache(self, is_cuda, debug=False):
        """
        Cached version of get_state_space.
        Cache key uses (is_cuda, scale_factor, bits, quant_type).
        """
        scale_factor, bits = self.get_scale_factor_bits()
        quant_type = self.get_quant_type()
        key = (is_cuda, scale_factor, bits, quant_type)
        if key in self._state_space_cache:
            if debug:
                print("DEBUG: get_state_space_with_cache cache hit for key:", key)
            return self._state_space_cache[key]
        if debug:
            start_time = time.time()
        result = self.get_state_space(is_cuda)
        self._state_space_cache[key] = result
        if debug:
            elapsed_time = time.time() - start_time
            print("DEBUG: get_state_space_with_cache computed new value for key:", key,
                  "in {:.6f} seconds".format(elapsed_time))
        return result

    def get_bin_state_space(self, is_cuda):
        quant_type = self.get_quant_type()
        _, bits = self.get_scale_factor_bits()
        if quant_type == QuantType.INT:
            tensor_quant = self.brevitas_module.act_quant_proxy.fused_activation_quant_proxy.tensor_quant
            narrow_range = tensor_quant.int_quant.narrow_range
            signed = tensor_quant.int_quant.signed
            state_space = get_int_state_space(bits, signed, narrow_range, is_cuda)
        elif quant_type == QuantType.BINARY:
            state_space = torch.tensor([0, 1])
        else:
            raise Exception("Unknown quantization type: {}".format(quant_type))
        return state_space

    def get_bin_state_space_with_cache(self, is_cuda, debug=False):
        """
        Cached version of get_bin_state_space.
        Cache key uses (is_cuda, quant_type, bits, [narrow_range, signed] for INT).
        For simplicity, we use (is_cuda, self.get_scale_factor_bits(), self.get_quant_type()) as key.
        """
        scale_factor, bits = self.get_scale_factor_bits()
        quant_type = self.get_quant_type()
        key = (is_cuda, scale_factor, bits, quant_type)
        if key in self._bin_state_space_cache:
            if debug:
                print("DEBUG: get_bin_state_space_with_cache cache hit for key:", key)
            return self._bin_state_space_cache[key]
        if debug:
            start_time = time.time()
        result = self.get_bin_state_space(is_cuda)
        self._bin_state_space_cache[key] = result
        if debug:
            elapsed_time = time.time() - start_time
            print("DEBUG: get_bin_state_space_with_cache computed new value for key:", key,
                  "in {:.6f} seconds".format(elapsed_time))
        return result

    def bin_output(self):
        self.is_bin_output = True

    def float_output(self):
        self.is_bin_output = False

    def get_quant_type(self):
        brevitas_module_type = type(
            self.brevitas_module.act_quant_proxy.fused_activation_quant_proxy.tensor_quant
        )
        if brevitas_module_type == RescalingIntQuant:
            return QuantType.INT
        elif brevitas_module_type == ClampedBinaryQuant:
            return QuantType.BINARY
        else:
            raise Exception(
                "Unknown quantization type for tensor_quant: {}".format(brevitas_module_type)
            )

    def get_scale_factor_bits(self):
        quant_proxy = self.brevitas_module.act_quant_proxy
        current_status = quant_proxy.training
        quant_proxy.eval()
        _, scale_factor, bits = quant_proxy(quant_proxy.zero_hw_sentinel)
        quant_proxy.training = current_status
        return scale_factor, bits

    def apply_pre_transforms(self, x):
        for i in range(len(self.pre_transforms)):
            x = self.pre_transforms[i](x)
        return x

    def apply_post_transforms(self, x):
        for i in range(len(self.post_transforms)):
            x = self.post_transforms[i](x)
        return x

    def forward(self, x):
        if self.is_bin_output:
            s, _ = self.get_scale_factor_bits()
            x = self.apply_pre_transforms(x)
            x = self.brevitas_module(x)
            x = torch.round(x / s).type(torch.int64)
        else:
            x = self.apply_pre_transforms(x)
            x = self.brevitas_module(x)
            x = self.apply_post_transforms(x)
        return x
