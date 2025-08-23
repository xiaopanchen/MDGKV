import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from meta_layers import *
from torch.autograd import Function
from kmm import KMM, BasicConv2d
from KinRelation import *


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('MetaLinear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        print(x.size())
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.maxpool(output)
        output = self.bn2(output)

        return output


# 基于度量学习和典型相关分析的亲缘关系识别网络——控制理论与决策文章
class TSKRM(nn.Module):
    def __init__(self):
        super(TSKRM, self).__init__()
        self.layer1 = nn.Sequential(
            BasicBlock(3, 32, stride=1))

        self.layer2 = nn.Sequential(
            BasicBlock(32, 64, stride=1))

        self.layer3 = nn.Sequential(
            BasicBlock(64, 128, stride=1))

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(size=(8, 8))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # print(out.size())
        return out


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(MetaModule):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = MetaConv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = MetaBatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.conv2 = MetaConv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = MetaBatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.extra = nn.Sequential(
            MetaConv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            MetaBatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet6(nn.Module):
    def __init__(self):
        super(RestNet6, self).__init__()
        self.layer1 = nn.Sequential(RestNetDownBlock(3, 32, [2, 1]),
                                    )

        self.layer2 = nn.Sequential(RestNetDownBlock(32, 64, [2, 1]),
                                    )

        self.layer3 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class RestNet12(nn.Module):
    def __init__(self):
        super(RestNet12, self).__init__()
        self.layer1 = nn.Sequential(RestNetDownBlock(3, 32, [2, 1]),
                                    RestNetBasicBlock(32, 32, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(32, 64, [2, 1]),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class RestNet10(nn.Module):
    def __init__(self):
        super(RestNet10, self).__init__()
        self.layer1 = nn.Sequential(RestNetDownBlock(3, 32, [2, 1]),
                                    RestNetBasicBlock(32, 32, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(32, 64, [2, 1]),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class GeM(nn.Module):
    """
    Generalized-mean pooling: GeM pooling incorporates GAP and
max pooling methods, but keeps embeddings in d-dimensional space.
GAP and max pooling are special cases of GeM pooling,
i.e., when p = 1, GeM turns into GAP, and turns into max pooling when p → ∞.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


def init_param(model):
    # when you add the convolution and batch norm, below will be useful
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class RestNet6_attention(MetaModule):
    def __init__(self):
        super(RestNet6_attention, self).__init__()
        self.layer1 = RestNetDownBlock(3, 32, [2, 1])

        self.layer2 = RestNetDownBlock(32, 64, [2, 1])

        self.layer3 = RestNetDownBlock(64, 128, [2, 1])

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(size=(8, 8))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # print(out.size())
        x1 = self.maxpool(out)
        x2 = self.upsample(x1)
        x3 = self.sigmoid(x2)
        out = (out * x3 + out)

        return out


class RestNet6_attention_MetaBN(MetaModule):
    def __init__(self):
        super(RestNet6_attention_MetaBN, self).__init__()
        self.layer1 = RestNetDownBlock(3, 32, [2, 1])

        self.layer2 = RestNetDownBlock(32, 64, [2, 1])

        self.layer3 = RestNetDownBlock(64, 128, [2, 1])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(size=(8, 8))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # print(out.size())
        x1 = self.maxpool(out)
        x2 = self.upsample(x1)
        x3 = self.sigmoid(x2)
        out = (out * x3 + out)
        # out = self.gap(out)

        return out


class train_Net_zl_MetaBN(MetaModule):
    def __init__(self):
        super(train_Net_zl_MetaBN, self).__init__()
        self.base = RestNet6_attention_MetaBN()

        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.last1 = nn.Sequential(
            MetaLinear(32768, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            MetaLinear(2048, 1)
        )
        self.last1.apply(weights_init_classifier)

        self.feat_bn_p = MixUpBatchNorm1d(8192)
        init.constant_(self.feat_bn_p.weight, 1)
        init.constant_(self.feat_bn_p.bias, 0)

        self.feat_bn_c = MixUpBatchNorm1d(8192)
        init.constant_(self.feat_bn_c.weight, 1)
        init.constant_(self.feat_bn_c.bias, 0)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2, MTE='', save_index=0):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        # print(x1.size())
        feature1 = self.base(x1)
        feature2 = self.base(x2)
        # 待测试
        feature1 = feature1.view(feature1.size(0), -1)
        # print(feature1.shape)
        feature1 = self.feat_bn_p(feature1, MTE, save_index)
        feature2 = feature2.view(feature2.size(0), -1)
        # print(feature2.shape)
        feature2 = self.feat_bn_c(feature2, MTE, save_index)

        x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        x2 = torch.pow(feature1 - feature2, 2)
        x3 = feature1 * feature2
        x4 = feature1 + feature2

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.last1(x)

        return x, feature1, feature2


class RestNet12_attention(nn.Module):
    def __init__(self):
        super(RestNet12_attention, self).__init__()
        self.layer1 = nn.Sequential(RestNetDownBlock(3, 32, [2, 1]),
                                    RestNetBasicBlock(32, 32, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(32, 64, [2, 1]),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(size=(8, 8))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # print(out.size())
        x1 = self.maxpool(out)
        x2 = self.upsample(x1)
        x3 = self.sigmoid(x2)
        out = (out * x3 + out)  # attention

        return out


class train_Net_zl(MetaModule):
    def __init__(self):
        super(train_Net_zl, self).__init__()
        self.model = RestNet6_attention()
        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.last1 = nn.Sequential(
            MetaLinear(32768, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            MetaLinear(2048, 1)
        )
        self.last1.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        # print(x1.size())
        feature1 = self.model(x1)
        feature2 = self.model(x2)
        x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        x2 = torch.pow(feature1 - feature2, 2)
        x3 = feature1 * feature2
        x4 = feature1 + feature2

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.last1(x)

        return x, feature1, feature2


class GradReverse(Function):
    # 声明静态方法：在使用这个方法的时候，类不需要实例化
    @staticmethod
    # ctx can be used to store tensors that can be then retrieved during the backward pass.
    # 从输入向量计算输出向量
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    # 接受输出向量对于某个标量的梯度，然后计算输入向量相对于这个标量的梯度。
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        # Use it by calling the apply method:
        return GradReverse.apply(x, constant)


class NetD(MetaModule):
    """
    AlexNet Discriminator for Meta-Learning Domain Generalization(MLADG) on PACS
    """

    def __init__(self, discriminate=1):
        # why discriminate is 1? xpchen
        super(NetD, self).__init__()

        self.layer = nn.Sequential(
            MetaLinear(8192, 1024),
            nn.ReLU(),
            MetaLinear(1024, discriminate),
            nn.Sigmoid(),
        )

    def forward(self, input):
        """
        Forward the discriminator with a backward reverse layer
        """
        # gradient revesal layer(GRL)
        input = GradReverse.grad_reverse(input, constant=1)
        out = self.layer(input)
        return out


class train_Net_zl_jig_D(MetaModule):
    def __init__(self, jigsaw_classes=31):
        super(train_Net_zl_jig_D, self).__init__()
        self.model = RestNet6_attention()
        self.netD = NetD()
        # self.model.apply(init_param)  # 2022.03.14
        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.last1 = nn.Sequential(
            MetaLinear(32768, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            MetaLinear(2048, 1)
        )
        self.last1.apply(weights_init_classifier)
        self.jigsaw_classifier = MetaLinear(8192, jigsaw_classes)
        self.jigsaw_classifier.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        # print(x1.size())
        feature1 = self.model(x1)
        feature2 = self.model(x2)

        x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        x2 = torch.pow(feature1 - feature2, 2)
        x3 = feature1 * feature2
        x4 = feature1 + feature2

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.last1(x)
        feature1 = feature1.view(feature1.size(0), -1)
        feature2 = feature2.view(feature2.size(0), -1)
        fea = torch.cat((feature1, feature2), dim=0)
        predict = self.netD(fea)
        return x, feature1, feature2, self.jigsaw_classifier(feature1), self.jigsaw_classifier(feature2), predict


class train_Net_zl_jig0(MetaModule):
    def __init__(self, jigsaw_classes=31):
        super(train_Net_zl_jig0, self).__init__()

        self.model = RestNet6_attention()
        # self.model.apply(init_param)  # 2022.03.14
        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.last1 = nn.Sequential(
            MetaLinear(32768, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            MetaLinear(2048, 1)
        )
        self.last1.apply(weights_init_classifier)
        self.jigsaw_classifier = nn.Sequential(
            MetaLinear(8192, 1024),
            nn.ReLU(),
            MetaLinear(1024, 512),
            nn.ReLU(),
            MetaLinear(512, jigsaw_classes),
        )
        self.jigsaw_classifier.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2, x10, x20):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        # print(x1.size())
        feature1 = self.model(x1)
        feature2 = self.model(x2)
        feature10 = self.model(x10)
        feature20 = self.model(x20)

        x1 = torch.pow(feature10, 2) - torch.pow(feature20, 2)
        x2 = torch.pow(feature10 - feature20, 2)
        x3 = feature10 * feature20
        x4 = feature10 + feature20

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.last1(x)
        feature1 = feature1.view(feature1.size(0), -1)
        feature2 = feature2.view(feature2.size(0), -1)
        return x, feature10, feature20, self.jigsaw_classifier(feature1), self.jigsaw_classifier(feature2)


class train_Net_zl_jig(MetaModule):
    def __init__(self, jigsaw_classes=31):
        super(train_Net_zl_jig, self).__init__()

        self.model = RestNet6_attention()
        # self.model.apply(init_param)  # 2022.03.14
        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.last1 = nn.Sequential(
            MetaLinear(32768, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            MetaLinear(2048, 1)
        )
        self.last1.apply(weights_init_classifier)
        self.jigsaw_classifier = MetaLinear(8192, jigsaw_classes)
        self.jigsaw_classifier.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        # print(x1.size())
        feature1 = self.model(x1)
        feature2 = self.model(x2)

        x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        x2 = torch.pow(feature1 - feature2, 2)
        x3 = feature1 * feature2
        x4 = feature1 + feature2

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.last1(x)
        feature1 = feature1.view(feature1.size(0), -1)
        feature2 = feature2.view(feature2.size(0), -1)
        return x, feature1, feature2, self.jigsaw_classifier(feature1), self.jigsaw_classifier(feature2)


class train_Net_zl_jigF(MetaModule):
    # 不打乱的特征用来亲属关系鉴定，打乱的特征约束学习到的特征
    def __init__(self, jigsaw_classes=31):
        super(train_Net_zl_jigF, self).__init__()

        self.model = RestNet6_attention()
        # self.model.apply(init_param)  # 2022.03.14
        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.last1 = nn.Sequential(
            MetaLinear(32768, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            MetaLinear(2048, 1),
            nn.Sigmoid()
        )
        self.last1.apply(weights_init_classifier)
        self.jigsaw_classifier = nn.Sequential(MetaLinear(8192, 2048),
                                               nn.ReLU(),
                                               MetaLinear(2048, jigsaw_classes))
        self.jigsaw_classifier.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2, x1_0, x2_0):  # x.shape: [batchsize, 6, 64, 64]
        if self.training:
            feature1 = self.model(x1)
            feature2 = self.model(x2)
            # for rorate images
            feature1_0 = self.model(x1_0)
            feature2_0 = self.model(x2_0)

            x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
            x2 = torch.pow(feature1 - feature2, 2)
            x3 = feature1 * feature2
            x4 = feature1 + feature2

            x = torch.cat((x1, x2, x3, x4), dim=1)
            x = x.view(x.size(0), -1)
            # print(x.size())
            x = self.last1(x)
            feature1_0 = feature1_0.view(feature1_0.size(0), -1)
            feature2_0 = feature2_0.view(feature2_0.size(0), -1)
            return x, feature1, feature2, self.jigsaw_classifier(feature1_0), self.jigsaw_classifier(feature2_0)
        else:
            feature1 = self.model(x1)
            feature2 = self.model(x2)

            x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
            x2 = torch.pow(feature1 - feature2, 2)
            x3 = feature1 * feature2
            x4 = feature1 + feature2

            x = torch.cat((x1, x2, x3, x4), dim=1)
            x = x.view(x.size(0), -1)
            # print(x.size())
            x = self.last1(x)
            feature1 = feature1.view(feature1.size(0), -1)
            feature2 = feature2.view(feature2.size(0), -1)
            return x, feature1, feature2, self.jigsaw_classifier(feature1), self.jigsaw_classifier(feature2)


class train_Net_zl_Rotate(MetaModule):
    def __init__(self, angle_classes=31):
        super(train_Net_zl_Rotate, self).__init__()

        self.model = RestNet6_attention()
        # self.model.apply(init_param)  # 2022.03.14
        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.last1 = nn.Sequential(
            MetaLinear(32768, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            MetaLinear(2048, 1024),
            MetaLinear(1024, 1)
        )
        self.last1.apply(weights_init_classifier)
        self.rotate_classifier = nn.Sequential(MetaLinear(8192, 2048),
                                               nn.ReLU(),
                                               MetaLinear(2048, angle_classes))
        self.rotate_classifier.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        # print(x1.size())
        feature1 = self.model(x1)
        feature2 = self.model(x2)

        x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        x2 = torch.pow(feature1 - feature2, 2)
        x3 = feature1 * feature2
        x4 = feature1 + feature2

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.last1(x)
        feature1 = feature1.view(feature1.size(0), -1)
        feature2 = feature2.view(feature2.size(0), -1)
        return x, feature1, feature2, self.rotate_classifier(feature1), self.rotate_classifier(feature2)


class train_Net_zl_RotateF(MetaModule):
    def __init__(self, angle_classes=4):
        super(train_Net_zl_RotateF, self).__init__()

        self.model = RestNet6_attention()
        # self.model.apply(init_param)  # 2022.03.14
        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.last1 = nn.Sequential(
            MetaLinear(32768, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            MetaLinear(2048, 1024),
            MetaLinear(1024, 1),
            nn.Sigmoid()  # 2022.5.6 add
        )
        self.last1.apply(weights_init_classifier)
        self.rotate_classifier = nn.Sequential(MetaLinear(8192, 2048),
                                               nn.ReLU(),
                                               MetaLinear(2048, angle_classes))
        self.rotate_classifier.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2, x1_0, x2_0):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        # print(x1.size())
        if self.training:
            feature1 = self.model(x1)
            feature2 = self.model(x2)
            # for rorate images
            feature1_0 = self.model(x1_0)
            feature2_0 = self.model(x2_0)
            # 如果只用下面的x1和x2试试?20230424
            x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
            x2 = torch.pow(feature1 - feature2, 2)
            x3 = feature1 * feature2
            x4 = feature1 + feature2

            x = torch.cat((x1, x2, x3, x4), dim=1)
            x = x.view(x.size(0), -1)
            # print(x.size())
            x = self.last1(x)
            feature1_0 = feature1_0.view(feature1_0.size(0), -1)
            feature2_0 = feature2_0.view(feature2_0.size(0), -1)
            return x, feature1, feature2, self.rotate_classifier(feature1_0), self.rotate_classifier(feature2_0)
        else:
            feature1 = self.model(x1)
            feature2 = self.model(x2)

            x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
            x2 = torch.pow(feature1 - feature2, 2)
            x3 = feature1 * feature2
            x4 = feature1 + feature2

            x = torch.cat((x1, x2, x3, x4), dim=1)
            x = x.view(x.size(0), -1)
            # print(x.size())
            x = self.last1(x)
            feature1 = feature1.view(feature1.size(0), -1)
            feature2 = feature2.view(feature2.size(0), -1)
            return x, feature1, feature2, self.rotate_classifier(feature1), self.rotate_classifier(feature2)


class train_Net_zl_Baseline(MetaModule):
    def __init__(self, angle_classes=4):
        super(train_Net_zl_Baseline, self).__init__()

        self.model = RestNet6_attention()
        # self.model.apply(init_param)  # 2022.03.14
        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.last1 = nn.Sequential(
            MetaLinear(32768, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            MetaLinear(2048, 1024),
            MetaLinear(1024, 1),
            nn.Sigmoid()  # 2022.5.6 add
        )
        self.last1.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        # print(x1.size())

        feature1 = self.model(x1)
        feature2 = self.model(x2)
        x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        x2 = torch.pow(feature1 - feature2, 2)
        x3 = feature1 * feature2
        x4 = feature1 + feature2

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.last1(x)

        return x, feature1, feature2


class train_Net_resnet18_jig(MetaModule):
    def __init__(self, jigsaw_classes=31):
        super(train_Net_resnet18_jig, self).__init__()

        self.model = resnet18(pretrained=True)
        # self.model.apply(init_param)  # 2022.03.14
        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.last1 = nn.Sequential(
            MetaLinear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            MetaLinear(1024, 1)
        )
        self.last1.apply(weights_init_classifier)
        self.jigsaw_classifier = MetaLinear(512, jigsaw_classes)
        self.jigsaw_classifier.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        # print(x1.size())
        feature1 = self.model(x1)
        feature2 = self.model(x2)

        x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        x2 = torch.pow(feature1 - feature2, 2)
        x3 = feature1 * feature2
        x4 = feature1 + feature2

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.last1(x)
        feature1 = feature1.view(feature1.size(0), -1)
        feature2 = feature2.view(feature2.size(0), -1)
        return x, feature1, feature2, self.jigsaw_classifier(feature1), self.jigsaw_classifier(feature2)


class train_Net_Li_jig(MetaModule):
    def __init__(self, jigsaw_classes=31):
        super(train_Net_Li_jig, self).__init__()
        feature = resnet18(pretrained=True)
        self.kin_model = KinRelation_jiG(feature, 512)
        # self.model.apply(init_param)  # 2022.03.14
        # self.classifier = ClassBlock(10368, 533, 0.5)
        # self.last1 = nn.Sequential(
        #     MetaLinear(32768, 2048),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     MetaLinear(2048, 1)
        # )
        # self.last1.apply(weights_init_classifier)
        self.jigsaw_classifier = MetaLinear(512, jigsaw_classes)
        self.jigsaw_classifier.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2, x10, x20):  # x.shape: [batchsize, 6, 64, 64]
        if self.training:
            x, feature1, feature2, feature10, feature20 = self.kin_model(x1, x2, x10, x20)
            feature1 = feature1.view(feature1.size(0), -1)
            feature2 = feature2.view(feature2.size(0), -1)
            v1, v2 = self.jigsaw_classifier(feature1), self.jigsaw_classifier(feature2)
            # 分块前的
            feature10 = feature10.view(feature10.size(0), -1)
            feature20 = feature20.view(feature20.size(0), -1)
            return x, feature10, feature20, v1, v2
        else:
            x, feature10, feature20 = self.kin_model(x1, x2, x10, x20)
            feature10 = feature10.view(feature10.size(0), -1)
            feature20 = feature20.view(feature20.size(0), -1)
            return x, feature10, feature20

        # x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        # x2 = torch.pow(feature1 - feature2, 2)
        # x3 = feature1 * feature2
        # x4 = feature1 + feature2
        #
        # x = torch.cat((x1, x2, x3, x4), dim=1)
        # x = x.view(x.size(0), -1)
        # # print(x.size())
        # x = self.last1(x)
        # 分块后的

        feature1 = feature1.view(feature1.size(0), -1)
        feature2 = feature2.view(feature2.size(0), -1)
        v1, v2 = self.jigsaw_classifier(feature1), self.jigsaw_classifier(feature2)
        # 分块前的
        feature10 = feature10.view(feature10.size(0), -1)
        feature20 = feature20.view(feature20.size(0), -1)
        if self.training:
            return x, feature10, feature20, v1, v2
        else:
            return x, feature10, feature20


class train_Net_zl_jig_KMM(MetaModule):
    def __init__(self, jigsaw_classes=31):
        super(train_Net_zl_jig_KMM, self).__init__()
        self.model = RestNet6_attention()
        # self.model.apply(init_param)  # 2022.03.14
        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.last1 = nn.Sequential(
            MetaLinear(32768, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            MetaLinear(2048, 1)
        )
        self.KMM = KMM(128)
        # self.KMM2 = KMM(128)
        # 1×1卷积
        self.cn_f1 = BasicConv2d(128, 128, 1, 1)
        self.cn_f2 = BasicConv2d(128, 128, 1, 1)
        self.last1.apply(weights_init_classifier)
        self.jigsaw_classifier = MetaLinear(8192, jigsaw_classes)
        self.jigsaw_classifier.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        # print(x1.size())
        feature1 = self.model(x1)
        feature2 = self.model(x2)
        # KMM
        k_p = self.cn_f1(feature1)
        k_c = self.cn_f2(feature2)
        k_p = self.KMM(k_p)
        k_c = self.KMM(k_c)

        x1 = torch.pow(k_p, 2) - torch.pow(k_c, 2)
        x2 = torch.pow(k_p - k_c, 2)
        x3 = k_p * k_c
        x4 = k_p + k_c

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.last1(x)
        feature1 = feature1.view(feature1.size(0), -1)
        feature2 = feature2.view(feature2.size(0), -1)
        return x, feature1, feature2, self.jigsaw_classifier(feature1), self.jigsaw_classifier(feature2)


class rel_Net_zl(MetaModule):
    def __init__(self):
        super(rel_Net_zl, self).__init__()
        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.last1 = nn.Sequential(
            MetaLinear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            MetaLinear(512, 1)
        )
        self.last1.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        # print(x1.size())
        feature1 = x1
        feature2 = x2

        x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        x2 = torch.pow(feature1 - feature2, 2)
        x3 = feature1 * feature2
        x4 = feature1 + feature2

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.last1(x)

        return x


class train_Net_zl_xp(MetaModule):
    def __init__(self):
        super(train_Net_zl_xp, self).__init__()
        self.model = RestNet6_attention()
        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.last1 = nn.Sequential(
            MetaLinear(32768, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            MetaLinear(2048, 512),
            MetaLinear(512, 1)
        )

        self.last1.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        # print(x1.size())
        feature1 = self.model(x1)
        feature2 = self.model(x2)

        x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        x2 = torch.pow(feature1 - feature2, 2)
        x3 = feature1 * feature2
        x4 = feature1 + feature2
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.last1(x)

        return x, feature1, feature2


def main():
    input = torch.rand(16, 3, 64, 64)
    model = TSKRM()
    output = model.forward(input)
    print(output.shape)

    pass


if __name__ == '__main__':
    main()
