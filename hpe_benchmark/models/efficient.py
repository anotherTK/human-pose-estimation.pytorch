import re
import math
import collections
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from hpe_benchmark.layers import JointsL2Loss

GlobalParams = collections.namedtuple('GlobalParams', ['batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor', 'min_depth', 'drop_connect_rate'])

BlockArgs = collections.namedtuple('BlockArgs', ['kernel_size', 'num_repeat', 'input_filters', 'output_filters', 'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

def relu_fn(x):
    return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(
        filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class Conv2dSamePadding(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels,
                         kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [
            self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] +
                    (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] +
                    (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w //
                          2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def efficientnet_params(model_name):
    params_dict = {
        # coefficients: width, depth, res, dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }

    return params_dict[model_name]


class BlockDecoder(object):

    @staticmethod
    def _encode_block_string(block):
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]

        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ration)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def _decode_block_string(block_string):
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        assert (('s' in options and len(options['s']) == 1) or (
            len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])]
        )

    @staticmethod
    def encode(block_args):
        block_strings = []
        for block in block_args:
            block_strings.append(
                BlockDecoder._encode_block_string(block)
            )
        return block_strings

    @staticmethod
    def decode(string_list):
        assert isinstance(string_list, list)
        block_args = []
        for block_string in string_list:
            block_args.append(
                BlockDecoder._decode_block_string(block_string)
            )
        return block_args


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2, drop_connect_rate=0.2):
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    if model_name.startswith('efficientnet'):
        # width, depth, res, dropout
        w, d, _, p = efficientnet_params(model_name)
        block_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p)
    else:
        raise NotImplementedError(
            'model name is not pre-defined: %s' % model_name)

    if override_params:
        global_params = global_params._replace(**override_params)

    return block_args, global_params


url_map = {
    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth',
    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth',
    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth',
    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth',
    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth',
    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth',
}


def load_pretrained_weights(model, model_name):
    state_dict = model_zoo.load_url(url_map[model_name])
    model.load_state_dict(state_dict)
    print('Loaded pretrained weights for {}'.format(model_name))


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block"""

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio

        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2dSamePadding(
                in_channels=inp,
                out_channels=oup,
                kernel_size=1,
                bias=False,
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup,
                momentum=self._bn_mom,
                eps=self._bn_eps,
            )

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2dSamePadding(
            in_channels=oup,
            out_channels=oup,
            groups=oup,
            kernel_size=k,
            stride=s,
            bias=False,
        )

        self._bn1 = nn.BatchNorm2d(
            num_features=oup,
            momentum=self._bn_mom,
            eps=self._bn_eps,
        )

        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio))

            self._se_reduce = Conv2dSamePadding(
                in_channels=oup,
                out_channels=num_squeezed_channels,
                kernel_size=1,
            )

            self._se_expand = Conv2dSamePadding(
                in_channels=num_squeezed_channels,
                out_channels=oup,
                kernel_size=1,
            )

        final_oup = self._block_args.output_filters
        self._project_conv = Conv2dSamePadding(
            in_channels=oup,
            out_channels=final_oup,
            kernel_size=1,
            bias=False,
        )

        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup,
            momentum=self._bn_mom,
            eps=self._bn_eps,
        )

    def forward(self, inputs, drop_connect_rate=None):

        x = inputs
        # Expansion
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))

        # Depthwise
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # SE
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            # reduce-expand
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            # integral/ re-weight
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate,
                                 training=self.training)
            x = x + inputs
        return x


class PEFFN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        blocks_args, global_params = get_model_params(cfg.MODEL.EFFNET.NAME, None)
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.ohkm = cfg.LOSS.OHKM
        self.topk = cfg.LOSS.TOPK
        self.ctf = cfg.LOSS.COARSE_TO_FINE

        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2dSamePadding(
            in_channels, out_channels,
            kernel_size=3,
            stride=2,
            bias=False,
        )
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=bn_mom,
            eps=bn_eps,
        )

        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params),
                output_filters=round_filters(
                    block_args.output_filters, self._global_params),
                num_repeat=round_repeats(
                    block_args.num_repeat, self._global_params)
            )

            self._blocks.append(MBConvBlock(block_args, self._global_params))

            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(
                    block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2dSamePadding(
            in_channels, out_channels,
            kernel_size=1,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=bn_mom,
            eps=bn_eps
        )

        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        self._loss = nn.CrossEntropyLoss()

        self.down_out_idx = [5, 9, 21, 31]
        self.upsample_layers = self._make_upsample_layers([448, 160, 56, 32])
        self.skip_connect = cfg.MODEL.EFFNET.SKIP_CONNECT
        self.out_channels = cfg.KEYPOINT.NUM
        self.final_layers = self._make_final_layers([448, 160, 56, 32])

        self.init_weights()
        # load pretrained


    def _make_upsample_layers(self, num_channles):
        layers = []
        for i in range(len(num_channles)):
            layers.append(nn.Sequential(
                Conv2dSamePadding(num_channles[i], num_channles[i], 3),
                nn.BatchNorm2d(num_channles[i]),
            ))
            if i != len(num_channles) - 1:
                layers.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    Conv2dSamePadding(num_channles[i], num_channles[i + 1], 3),
                    nn.BatchNorm2d(num_channles[i + 1]),
                ))

        return nn.ModuleList(layers)

    def _make_final_layers(self, num_channles):
        layers = []
        scale_factor = [8, 4, 2, 1]
        for i in range(len(num_channles)):
            if i == len(num_channles) - 1:
                layers.append(Conv2dSamePadding(
                    num_channles[i], self.out_channels, 1))
            else:
                layers.append(nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor[i], mode='bilinear'),
                    Conv2dSamePadding(num_channles[i], self.out_channels, 1),
                ))

        return nn.ModuleList(layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def _calculate_loss(self, outputs, valids, labels):
        loss1 = JointsL2Loss()
        if self.ohkm:
            loss2 = JointsL2Loss(has_ohkm=self.ohkm, topk=self.topk)

        loss = 0
        for j in range(4):
            ind = j
            if self.ctf:
                ind += 1
            tmp_labels = labels[:, ind, :, :, :]

            if j == 3 and self.ohkm:
                tmp_loss = loss2(outputs[j], valids, tmp_labels)
            else:
                tmp_loss = loss1(outputs[j], valids, tmp_labels)

            if j < 3:
                tmp_loss = tmp_loss / 4

            loss += tmp_loss

        return dict(total_loss=loss)

    def forward(self, imgs, valids=None, labels=None):
        down_outputs = []
        x = relu_fn(self._bn0(self._conv_stem(imgs)))
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x)
            # print("{}: {}".format(idx,x.shape))
            if idx in self.down_out_idx:
                down_outputs.append(x)
        
        # upsample
        stage_output = []
        pre_up = down_outputs[-1]
        for i in range(len(down_outputs)):
            if i != 0 and self.skip_connect:
                x = relu_fn(self.upsample_layers[2*i](down_outputs[-i-1] + pre_up))
            else:
                x = relu_fn(self.upsample_layers[2*i](pre_up))
            stage_output.append(x)
            if i != len(down_outputs) - 1:
                pre_up = relu_fn(self.upsample_layers[2*i + 1](x))

        # make stage output to final feature map size
        outputs = []
        for i, s_out in enumerate(stage_output):
            outputs.append(self.final_layers[i](s_out))

        if valids is None and labels is None:
            return outputs[-1]
        else:
            return self._calculate_loss(outputs, valids, labels)


    @classmethod
    def from_pretrained(cls, model_name):
        model = EfficientNet.from_name(model_name)
        load_pretrained_weights(model, model_name)
        return model
