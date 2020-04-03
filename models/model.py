from .resnet import *
import copy

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

PRETRAIN_FILE = '/root/face2face/pretrain_model/resnet18-5c106cde.pth'

class MultiHead(nn.Module):

    def load_pretrain(self, skip, is_print=True):
        conversion=copy.copy(CONVERSION)
        for i in range(0,len(conversion)-8,4):
            conversion[i] = 'block.' + conversion[i][5:]
        
        for network in self.networks:
            load_pretrain(network, PRETRAIN_FILE, skip=['logit'], is_print=False)

    def __init__(self, embedding_size=512, nb_heads=4):

        super(MultiHead, self).__init__()

        self.networks = nn.ModuleList()

        for _ in range(nb_heads):
            network = ResNet18()
            self.networks.append(network)

        self.flatten = Flatten()

        self.fc1 = nn.Linear(8192, embedding_size, bias=False)
        self.fc1_bn = nn.BatchNorm1d(embedding_size)

        self.fc2 = nn.Linear(8192, embedding_size, bias=False)
        self.fc2_bn = nn.BatchNorm1d(embedding_size)

        self.fc3 = nn.Linear(8192, embedding_size, bias=False)
        self.fc3_bn = nn.BatchNorm1d(embedding_size)

        self.fc4 = nn.Linear(8192, embedding_size, bias=False)
        self.fc4_bn = nn.BatchNorm1d(embedding_size)

        self.fc_fusion = nn.Linear(8192 * 4, embedding_size, bias=False)
        self.fc_fusion_bn = nn.BatchNorm1d(embedding_size)

        
    def forward(self, inputs): # inputs contains full face, top face, left face, right face
        f_input, l_input, r_input, t_input = inputs

        #---------------- f_input ------------------
        x1 = self.networks[0].block0(f_input)
        x1 = self.networks[0].block1(x1)
        x1 = self.networks[0].block2(x1)
        x1 = self.networks[0].block3(x1)
        x1 = self.networks[0].block4(x1) # 1, 512, 4, 4
        out1 = self.flatten(x1)
        out1 = self.fc1(out1)
        out1 = self.fc1_bn(out1)

        #---------------- l_input ------------------
        x2 = self.networks[1].block0(l_input)
        x2 = self.networks[1].block1(x2)
        x2 = self.networks[1].block2(x2)
        x2 = self.networks[1].block3(x2)
        x2 = self.networks[1].block4(x2)
        out2 = self.flatten(x2)
        out2 = self.fc2(out2)
        out2 = self.fc2_bn(out2)

        #---------------- r_input ------------------
        x3 = self.networks[2].block0(r_input)
        x3 = self.networks[2].block1(x3)
        x3 = self.networks[2].block2(x3)
        x3 = self.networks[2].block3(x3)
        x3 = self.networks[2].block4(x3)
        out3 = self.flatten(x3)
        out3 = self.fc3(out3)
        out3 = self.fc3_bn(out3)

        #---------------- t_input ------------------
        x4 = self.networks[3].block0(t_input)
        x4 = self.networks[3].block1(x4)
        x4 = self.networks[3].block2(x4)
        x4 = self.networks[3].block3(x4)
        x4 = self.networks[3].block4(x4)
        out4 = self.flatten(x4)
        out4 = self.fc4(out4)
        out4 = self.fc4_bn(out4)


        #---------------- combine all info ------------------
        fusion_out = torch.cat((x1, x2, x3, x4), 1)
        fusion_out = self.flatten(fusion_out)
        fusion_out = self.fc_fusion(fusion_out)
        fusion_out = self.fc_fusion_bn(fusion_out)

        return out1, out2, out3, out4, fusion_out



if __name__=='__main__':
    import numpy as np
    input1 = np.random.randn(2, 3, 128, 128)
    input2 = np.random.randn(2, 3, 128, 128)
    input3 = np.random.randn(2, 3, 128, 128)
    input4 = np.random.randn(2, 3, 128, 128)


    net = MultiHead()
    f1,f2,f3,f4 = net((torch.Tensor(input1), torch.Tensor(input2), torch.Tensor(input3), torch.Tensor(input4)))

    print(f1.shape)