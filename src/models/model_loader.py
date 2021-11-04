import torchvision.models as models
from torch import nn

def model_loader(model_name='resnet18', in_features=3, num_class=10, pretrain=False):
    assert model_name in ['resnet18', 'resnet34', 'resnet50', 'alexnet', 'vgg16', 'squeezenet1_0', 'densenet161', 'inception_v3', 'googlenet',
                          'mobilenet_v2', 'mobilenet_v3_large', 'mnasnet'], "no model in 'models' module."
    model = getattr(models, model_name)(pretrained=pretrain)

    if 'vgg' in model_name:
        in_layer = model.features[0]
        out_layer = model.classifier[-1]
        if not in_features == 3:
            out_channels = in_layer.out_channels
            kernel_size = in_layer.kernel_size
            stride = in_layer.stride
            padding = in_layer.padding
            bias = in_layer.bias
            model.features[0] = nn.Conv2d(in_features, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)

        if not num_class == out_layer.out_features:
            input_feature_fc_layer = out_layer.in_features
            model.classifier[-1] = nn.Linear(input_feature_fc_layer, num_class, bias=False)
    elif 'densenet' in model_name:
        print('dense')
        in_layer = model.features.conv0
        out_layer = model.classifier
        if not in_features == 3:
            out_channels = in_layer.out_channels
            kernel_size = in_layer.kernel_size
            stride = in_layer.stride
            padding = in_layer.padding
            bias = in_layer.bias
            model.features.conv0 = nn.Conv2d(in_features, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)

        if not num_class == out_layer.out_features:
            input_feature_fc_layer = out_layer.in_features
            model.classifier = nn.Linear(input_feature_fc_layer, num_class, bias=False)
    else:
        in_layer = model.conv1
        out_layer = model.fc
        if not in_features == 3:
            out_channels = in_layer.out_channels
            kernel_size = in_layer.kernel_size
            stride = in_layer.stride
            padding = in_layer.padding
            bias = in_layer.bias
            model.conv1 = nn.Conv2d(in_features, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)

        if not num_class == out_layer.out_features:
            input_feature_fc_layer = out_layer.in_features
            model.fc = nn.Linear(input_feature_fc_layer, num_class, bias=False)


    return model

# resnet18 = models.resnet18()
# alexnet = models.alexnet()
# vgg16 = models.vgg16()
# squeezenet = models.squeezenet1_0()
# densenet = models.densenet161()
# inception = models.inception_v3()
# googlenet = models.googlenet()
# shufflenet = models.shufflenet_v2_x1_0()
# mobilenet_v2 = models.mobilenet_v2()
# mobilenet_v3_large = models.mobilenet_v3_large()
# mobilenet_v3_small = models.mobilenet_v3_small()
# resnext50_32x4d = models.resnext50_32x4d()
# wide_resnet50_2 = models.wide_resnet50_2()
# mnasnet = models.mnasnet1_0()
