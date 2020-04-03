import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

###############################################################################
CONVERSION=[
 'block0.0.weight',	(64, 3, 7, 7),	 'conv1.weight',	(64, 3, 7, 7),
 'block0.1.weight',	(64,),	 'bn1.weight',	(64,),
 'block0.1.bias',	(64,),	 'bn1.bias',	(64,),
 'block0.1.running_mean',	(64,),	 'bn1.running_mean',	(64,),
 'block0.1.running_var',	(64,),	 'bn1.running_var',	(64,),
 'block1.1.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.0.conv1.weight',	(64, 64, 3, 3),
 'block1.1.conv_bn1.bn.weight',	(64,),	 'layer1.0.bn1.weight',	(64,),
 'block1.1.conv_bn1.bn.bias',	(64,),	 'layer1.0.bn1.bias',	(64,),
 'block1.1.conv_bn1.bn.running_mean',	(64,),	 'layer1.0.bn1.running_mean',	(64,),
 'block1.1.conv_bn1.bn.running_var',	(64,),	 'layer1.0.bn1.running_var',	(64,),
 'block1.1.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.0.conv2.weight',	(64, 64, 3, 3),
 'block1.1.conv_bn2.bn.weight',	(64,),	 'layer1.0.bn2.weight',	(64,),
 'block1.1.conv_bn2.bn.bias',	(64,),	 'layer1.0.bn2.bias',	(64,),
 'block1.1.conv_bn2.bn.running_mean',	(64,),	 'layer1.0.bn2.running_mean',	(64,),
 'block1.1.conv_bn2.bn.running_var',	(64,),	 'layer1.0.bn2.running_var',	(64,),
 'block1.2.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.1.conv1.weight',	(64, 64, 3, 3),
 'block1.2.conv_bn1.bn.weight',	(64,),	 'layer1.1.bn1.weight',	(64,),
 'block1.2.conv_bn1.bn.bias',	(64,),	 'layer1.1.bn1.bias',	(64,),
 'block1.2.conv_bn1.bn.running_mean',	(64,),	 'layer1.1.bn1.running_mean',	(64,),
 'block1.2.conv_bn1.bn.running_var',	(64,),	 'layer1.1.bn1.running_var',	(64,),
 'block1.2.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.1.conv2.weight',	(64, 64, 3, 3),
 'block1.2.conv_bn2.bn.weight',	(64,),	 'layer1.1.bn2.weight',	(64,),
 'block1.2.conv_bn2.bn.bias',	(64,),	 'layer1.1.bn2.bias',	(64,),
 'block1.2.conv_bn2.bn.running_mean',	(64,),	 'layer1.1.bn2.running_mean',	(64,),
 'block1.2.conv_bn2.bn.running_var',	(64,),	 'layer1.1.bn2.running_var',	(64,),
 'block2.0.conv_bn1.conv.weight',	(128, 64, 3, 3),	 'layer2.0.conv1.weight',	(128, 64, 3, 3),
 'block2.0.conv_bn1.bn.weight',	(128,),	 'layer2.0.bn1.weight',	(128,),
 'block2.0.conv_bn1.bn.bias',	(128,),	 'layer2.0.bn1.bias',	(128,),
 'block2.0.conv_bn1.bn.running_mean',	(128,),	 'layer2.0.bn1.running_mean',	(128,),
 'block2.0.conv_bn1.bn.running_var',	(128,),	 'layer2.0.bn1.running_var',	(128,),
 'block2.0.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.0.conv2.weight',	(128, 128, 3, 3),
 'block2.0.conv_bn2.bn.weight',	(128,),	 'layer2.0.bn2.weight',	(128,),
 'block2.0.conv_bn2.bn.bias',	(128,),	 'layer2.0.bn2.bias',	(128,),
 'block2.0.conv_bn2.bn.running_mean',	(128,),	 'layer2.0.bn2.running_mean',	(128,),
 'block2.0.conv_bn2.bn.running_var',	(128,),	 'layer2.0.bn2.running_var',	(128,),
 'block2.0.shortcut.conv.weight',	(128, 64, 1, 1),	 'layer2.0.downsample.0.weight',	(128, 64, 1, 1),
 'block2.0.shortcut.bn.weight',	(128,),	 'layer2.0.downsample.1.weight',	(128,),
 'block2.0.shortcut.bn.bias',	(128,),	 'layer2.0.downsample.1.bias',	(128,),
 'block2.0.shortcut.bn.running_mean',	(128,),	 'layer2.0.downsample.1.running_mean',	(128,),
 'block2.0.shortcut.bn.running_var',	(128,),	 'layer2.0.downsample.1.running_var',	(128,),
 'block2.1.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.1.conv1.weight',	(128, 128, 3, 3),
 'block2.1.conv_bn1.bn.weight',	(128,),	 'layer2.1.bn1.weight',	(128,),
 'block2.1.conv_bn1.bn.bias',	(128,),	 'layer2.1.bn1.bias',	(128,),
 'block2.1.conv_bn1.bn.running_mean',	(128,),	 'layer2.1.bn1.running_mean',	(128,),
 'block2.1.conv_bn1.bn.running_var',	(128,),	 'layer2.1.bn1.running_var',	(128,),
 'block2.1.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.1.conv2.weight',	(128, 128, 3, 3),
 'block2.1.conv_bn2.bn.weight',	(128,),	 'layer2.1.bn2.weight',	(128,),
 'block2.1.conv_bn2.bn.bias',	(128,),	 'layer2.1.bn2.bias',	(128,),
 'block2.1.conv_bn2.bn.running_mean',	(128,),	 'layer2.1.bn2.running_mean',	(128,),
 'block2.1.conv_bn2.bn.running_var',	(128,),	 'layer2.1.bn2.running_var',	(128,),
 'block3.0.conv_bn1.conv.weight',	(256, 128, 3, 3),	 'layer3.0.conv1.weight',	(256, 128, 3, 3),
 'block3.0.conv_bn1.bn.weight',	(256,),	 'layer3.0.bn1.weight',	(256,),
 'block3.0.conv_bn1.bn.bias',	(256,),	 'layer3.0.bn1.bias',	(256,),
 'block3.0.conv_bn1.bn.running_mean',	(256,),	 'layer3.0.bn1.running_mean',	(256,),
 'block3.0.conv_bn1.bn.running_var',	(256,),	 'layer3.0.bn1.running_var',	(256,),
 'block3.0.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.0.conv2.weight',	(256, 256, 3, 3),
 'block3.0.conv_bn2.bn.weight',	(256,),	 'layer3.0.bn2.weight',	(256,),
 'block3.0.conv_bn2.bn.bias',	(256,),	 'layer3.0.bn2.bias',	(256,),
 'block3.0.conv_bn2.bn.running_mean',	(256,),	 'layer3.0.bn2.running_mean',	(256,),
 'block3.0.conv_bn2.bn.running_var',	(256,),	 'layer3.0.bn2.running_var',	(256,),
 'block3.0.shortcut.conv.weight',	(256, 128, 1, 1),	 'layer3.0.downsample.0.weight',	(256, 128, 1, 1),
 'block3.0.shortcut.bn.weight',	(256,),	 'layer3.0.downsample.1.weight',	(256,),
 'block3.0.shortcut.bn.bias',	(256,),	 'layer3.0.downsample.1.bias',	(256,),
 'block3.0.shortcut.bn.running_mean',	(256,),	 'layer3.0.downsample.1.running_mean',	(256,),
 'block3.0.shortcut.bn.running_var',	(256,),	 'layer3.0.downsample.1.running_var',	(256,),
 'block3.1.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.1.conv1.weight',	(256, 256, 3, 3),
 'block3.1.conv_bn1.bn.weight',	(256,),	 'layer3.1.bn1.weight',	(256,),
 'block3.1.conv_bn1.bn.bias',	(256,),	 'layer3.1.bn1.bias',	(256,),
 'block3.1.conv_bn1.bn.running_mean',	(256,),	 'layer3.1.bn1.running_mean',	(256,),
 'block3.1.conv_bn1.bn.running_var',	(256,),	 'layer3.1.bn1.running_var',	(256,),
 'block3.1.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.1.conv2.weight',	(256, 256, 3, 3),
 'block3.1.conv_bn2.bn.weight',	(256,),	 'layer3.1.bn2.weight',	(256,),
 'block3.1.conv_bn2.bn.bias',	(256,),	 'layer3.1.bn2.bias',	(256,),
 'block3.1.conv_bn2.bn.running_mean',	(256,),	 'layer3.1.bn2.running_mean',	(256,),
 'block3.1.conv_bn2.bn.running_var',	(256,),	 'layer3.1.bn2.running_var',	(256,),
 'block4.0.conv_bn1.conv.weight',	(512, 256, 3, 3),	 'layer4.0.conv1.weight',	(512, 256, 3, 3),
 'block4.0.conv_bn1.bn.weight',	(512,),	 'layer4.0.bn1.weight',	(512,),
 'block4.0.conv_bn1.bn.bias',	(512,),	 'layer4.0.bn1.bias',	(512,),
 'block4.0.conv_bn1.bn.running_mean',	(512,),	 'layer4.0.bn1.running_mean',	(512,),
 'block4.0.conv_bn1.bn.running_var',	(512,),	 'layer4.0.bn1.running_var',	(512,),
 'block4.0.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.0.conv2.weight',	(512, 512, 3, 3),
 'block4.0.conv_bn2.bn.weight',	(512,),	 'layer4.0.bn2.weight',	(512,),
 'block4.0.conv_bn2.bn.bias',	(512,),	 'layer4.0.bn2.bias',	(512,),
 'block4.0.conv_bn2.bn.running_mean',	(512,),	 'layer4.0.bn2.running_mean',	(512,),
 'block4.0.conv_bn2.bn.running_var',	(512,),	 'layer4.0.bn2.running_var',	(512,),
 'block4.0.shortcut.conv.weight',	(512, 256, 1, 1),	 'layer4.0.downsample.0.weight',	(512, 256, 1, 1),
 'block4.0.shortcut.bn.weight',	(512,),	 'layer4.0.downsample.1.weight',	(512,),
 'block4.0.shortcut.bn.bias',	(512,),	 'layer4.0.downsample.1.bias',	(512,),
 'block4.0.shortcut.bn.running_mean',	(512,),	 'layer4.0.downsample.1.running_mean',	(512,),
 'block4.0.shortcut.bn.running_var',	(512,),	 'layer4.0.downsample.1.running_var',	(512,),
 'block4.1.conv_bn1.conv.weight',	(512, 512, 3, 3),	 'layer4.1.conv1.weight',	(512, 512, 3, 3),
 'block4.1.conv_bn1.bn.weight',	(512,),	 'layer4.1.bn1.weight',	(512,),
 'block4.1.conv_bn1.bn.bias',	(512,),	 'layer4.1.bn1.bias',	(512,),
 'block4.1.conv_bn1.bn.running_mean',	(512,),	 'layer4.1.bn1.running_mean',	(512,),
 'block4.1.conv_bn1.bn.running_var',	(512,),	 'layer4.1.bn1.running_var',	(512,),
 'block4.1.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.1.conv2.weight',	(512, 512, 3, 3),
 'block4.1.conv_bn2.bn.weight',	(512,),	 'layer4.1.bn2.weight',	(512,),
 'block4.1.conv_bn2.bn.bias',	(512,),	 'layer4.1.bn2.bias',	(512,),
 'block4.1.conv_bn2.bn.running_mean',	(512,),	 'layer4.1.bn2.running_mean',	(512,),
 'block4.1.conv_bn2.bn.running_var',	(512,),	 'layer4.1.bn2.running_var',	(512,),
 'logit.weight',	(1000, 512),	 'fc.weight',	(1000, 512),
 'logit.bias',	(1000,),	 'fc.bias',	(1000,),

]

def load_pretrain(net, pretrain_file, skip=[], conversion=CONVERSION, is_print=True):

    #raise NotImplementedError
    print('\tload pretrain_file: %s'%pretrain_file)

    #pretrain_state_dict = torch.load(pretrain_file)
    pretrain_state_dict = torch.load(pretrain_file, map_location=lambda storage, loc: storage)
    state_dict = net.state_dict()

    i = 0
    conversion = np.array(conversion).reshape(-1,4)
    for key,_,pretrain_key,_ in conversion:
        if any(s in key for s in
            ['.num_batches_tracked',]+skip):
            continue

        #print('\t\t',key)
        if is_print:
            print('\t\t','%-48s  %-24s  <---  %-32s  %-24s'%(
                key, str(state_dict[key].shape),
                pretrain_key, str(pretrain_state_dict[pretrain_key].shape),
            ))
        i = i+1

        state_dict[key] = pretrain_state_dict[pretrain_key]

    net.load_state_dict(state_dict)
    print('')
    print('len(pretrain_state_dict.keys()) = %d'%len(pretrain_state_dict.keys()))
    print('len(state_dict.keys())          = %d'%len(state_dict.keys()))
    print('loaded    = %d'%i)
    print('')



###############################################################################
class ConvBn2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x

# bottleneck type C
class BasicBlock(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, is_shortcut=False):
        super(BasicBlock, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2d(in_channel,    channel, kernel_size=3, padding=1, stride=stride)
        self.conv_bn2 = ConvBn2d(   channel,out_channel, kernel_size=3, padding=1, stride=1)

        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):
        z = F.relu(self.conv_bn1(x),inplace=True)
        z = self.conv_bn2(z)

        if self.is_shortcut:
            x = self.shortcut(x)

        z += x
        z = F.relu(z,inplace=True)
        return z

#resnet18
class ResNet18(nn.Module):

    def __init__(self, num_class=1000 ):
        super(ResNet18, self).__init__()


        self.block0  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.block1  = nn.Sequential(
             nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
             BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,),
          * [BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,) for i in range(1,2)],
        )
        self.block2  = nn.Sequential(
             BasicBlock( 64,128,128, stride=2, is_shortcut=True, ),
          * [BasicBlock(128,128,128, stride=1, is_shortcut=False,) for i in range(1,2)],
        )
        self.block3  = nn.Sequential(
             BasicBlock(128,256,256, stride=2, is_shortcut=True, ),
          * [BasicBlock(256,256,256, stride=1, is_shortcut=False,) for i in range(1,2)],
        )
        self.block4 = nn.Sequential(
             BasicBlock(256,512,512, stride=2, is_shortcut=True, ),
          * [BasicBlock(512,512,512, stride=1, is_shortcut=False,) for i in range(1,2)],
        )

        self.logit = nn.Linear(512,num_class)

    def forward(self, x):
        batch_size = len(x)

        x = self.block0(x)
        x = F.max_pool2d(x,kernel_size=3, padding=1, stride=2, ceil_mode=False)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)
        return logit


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    net = ResNet18()
    load_pretrain(net, is_print=True)

    #---
    if 0:
        print(net)
        print('')

        print('*** print key *** ')
        state_dict = net.state_dict()
        keys = list(state_dict.keys())
        #keys = sorted(keys)
        for k in keys:
            if any(s in k for s in [
                'num_batches_tracked'
                # '.kernel',
                # '.gamma',
                # '.beta',
                # '.running_mean',
                # '.running_var',
            ]):
                continue

            p = state_dict[k].data.cpu().numpy()
            print(' \'%s\',\t%s,'%(k,tuple(p.shape)))
        print('')
        exit(0)

    print('\nsucess!')