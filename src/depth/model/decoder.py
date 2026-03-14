from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d

class Decoder:
    def __init__(self):
        self.conv_1_a = Conv2d(512, 256, kernel_size=3, stride=1)
        self.conv_1_b = Conv2d(512, 256, kernel_size=3, stride=1)

        self.conv_2_a = Conv2d(256, 128, kernel_size=3, stride=1)
        self.conv_2_b = Conv2d(256, 128, kernel_size=3, stride=1)
        self.conv_head_2 = Conv2d(128, 1, kernel_size=3, stride=1)

        self.conv_3_a = Conv2d(128, 64, kernel_size=3, stride=1)
        self.conv_3_b = Conv2d(128, 64, kernel_size=3, stride=1)
        self.conv_head_3 = Conv2d(64, 1, kernel_size=3, stride=1)

        self.conv_4_a = Conv2d(64, 32, kernel_size=3, stride=1)
        self.conv_4_b = Conv2d(96, 32, kernel_size=3, stride=1)
        self.conv_head_4 = Conv2d(32, 1, kernel_size=3, stride=1)

        self.conv_5_a = Conv2d(32, 16, kernel_size=3, stride=1)
        self.conv_5_b = Conv2d(16, 16, kernel_size=3, stride=1)
        self.conv_head_5 = Conv2d(16, 1, kernel_size=3, stride=1)


    def __call__(self, x: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        layer_0, layer_1, layer_2, layer_3, layer_4 = x

        layer_3_conv = self.conv_1_a(layer_4.pad((1, 1, 1, 1), mode="reflect")).elu()
        bb, cc, hh, ww = [int(s) for s in layer_3_conv.shape]
        layer_3_upsampled = layer_3_conv.interpolate((bb, cc, hh * 2, ww * 2), mode='nearest')
        layer_3_out = self.conv_1_b(layer_3_upsampled.cat(layer_3, dim=1).pad((1, 1, 1, 1), mode="reflect")).elu()

        layer_2_conv = self.conv_2_a(layer_3_out.pad((1, 1, 1, 1), mode="reflect")).elu()
        bb, cc, hh, ww = [int(s) for s in layer_2_conv.shape]
        layer_2_upsampled = layer_2_conv.interpolate((bb, cc, hh * 2, ww * 2), mode='nearest')
        layer_2_out = self.conv_2_b(layer_2_upsampled.cat(layer_2, dim=1).pad((1, 1, 1, 1), mode="reflect")).elu()
        layer_2_head = self.conv_head_2(layer_2_out.pad((1, 1, 1, 1), mode="reflect")).sigmoid()

        layer_1_conv = self.conv_3_a(layer_2_out.pad((1, 1, 1, 1), mode="reflect")).elu()
        bb, cc, hh, ww = [int(s) for s in layer_1_conv.shape]
        layer_1_upsampled = layer_1_conv.interpolate((bb, cc, hh * 2, ww * 2), mode='nearest')
        layer_1_out = self.conv_3_b(layer_1_upsampled.cat(layer_1, dim=1).pad((1, 1, 1, 1), mode="reflect")).elu()
        layer_1_head = self.conv_head_3(layer_1_out.pad((1, 1, 1, 1), mode="reflect")).sigmoid()

        stem_conv = self.conv_4_a(layer_1_out.pad((1, 1, 1, 1), mode="reflect")).elu()
        bb, cc, hh, ww = [int(s) for s in stem_conv.shape]
        stem_upsampled = stem_conv.interpolate((bb, cc, hh * 2, ww * 2), mode='nearest')
        stem_out = self.conv_4_b(stem_upsampled.cat(layer_0, dim=1).pad((1, 1, 1, 1), mode="reflect")).elu()
        stem_head = self.conv_head_4(stem_out.pad((1, 1, 1, 1), mode="reflect")).sigmoid()

        final_conv = self.conv_5_a(stem_out.pad((1, 1, 1, 1), mode="reflect")).elu()
        bb, cc, hh, ww = [int(s) for s in final_conv.shape]
        final_upsampled = final_conv.interpolate((bb, cc, hh * 2, ww * 2), mode='nearest')
        final_out = self.conv_5_b(final_upsampled.pad((1, 1, 1, 1), mode="reflect")).elu()
        final_head = self.conv_head_5(final_out.pad((1, 1, 1, 1), mode="reflect")).sigmoid()

        return final_head, stem_head, layer_1_head, layer_2_head
