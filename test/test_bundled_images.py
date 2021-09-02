#!/usr/bin/env python3
import torch
import torch.utils.bundled_inputs
import io
import cv2
from torch.testing._internal.common_utils import TestCase

torch.ops.load_library("//caffe2/torch/fb/operators:decode_bundled_image")

def model_size(sm):
    buffer = io.BytesIO()
    torch.jit.save(sm, buffer)
    return len(buffer.getvalue())

def save_and_load(sm):
    buffer = io.BytesIO()
    torch.jit.save(sm, buffer)
    buffer.seek(0)
    return torch.jit.load(buffer)

"""Return an InflatableArg that contains a tensor of the compressed image and the way to decode it

    keyword arguments:
    img_tensor -- the raw image tensor in HWC or NCHW with pixel value of type unsigned int
                  if in NCHW format, N should be 1
    quality -- the quality needed to compress the image
"""
def bundle_jpeg_image(img_tensor, quality):
    # turn NCHW to HWC
    if img_tensor.dim() == 4:
        assert(img_tensor.size(0) == 1)
        img_tensor = img_tensor[0].permute(1, 2, 0)
    pixels = img_tensor.numpy()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc_img = cv2.imencode(".JPEG", pixels, encode_param)
    enc_img_tensor = torch.from_numpy(enc_img)
    enc_img_tensor = torch.flatten(enc_img_tensor).byte()
    obj = torch.utils.bundled_inputs.InflatableArg(enc_img_tensor, "torch.ops.fb.decode_bundled_image({})")
    return obj

class TestBundledInputs(TestCase):
    def test_single_tensors(self):
        class SingleTensorModel(torch.nn.Module):
            def forward(self, arg):
                return arg
        im = cv2.imread("caffe2/test/test_img/p1.jpg")
        tensor = torch.from_numpy(im)
        inflatable_arg = bundle_jpeg_image(tensor, 90)
        input = [(inflatable_arg,)]
        sm = torch.jit.script(SingleTensorModel())
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(sm, input)
        loaded = save_and_load(sm)
        inflated = loaded.get_all_bundled_inputs()
        decoded_data = inflated[0][0]
        # raw image
        raw_data = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        raw_data = torch.from_numpy(raw_data).float()
        raw_data = raw_data.permute(2, 0, 1)
        raw_data = torch.div(raw_data, 255).unsqueeze(0)
        self.assertEqual(len(inflated), 1)
        self.assertEqual(len(inflated[0]), 1)
        self.assertEqual(raw_data.shape, decoded_data.shape)
        self.assertTrue(torch.allclose(raw_data, decoded_data, atol=0.1, rtol=1e-01))
