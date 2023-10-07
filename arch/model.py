import os
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class CrossModalAttention(nn.Module):
    """ CMA attention Layer"""

    def __init__(self, in_dim, activation=None, ratio=8, cross_value=True):
        super(CrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.cross_value = cross_value

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
        #self.bn = nn.BatchNorm2d(in_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = x.size()

        proj_query = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1)  # B , HW, C
        proj_key = self.key_conv(y).view(B, -1, H*W)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # B, HW, HW
        attention = self.softmax(energy)  # BX (N) X (N)
        if self.cross_value:
            proj_value = self.value_conv(y).view(B, -1, H*W)  # B , C , HW
        else:
            proj_value = self.value_conv(x).view(B, -1, H*W)  # B , C , HW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma*out + x

        if self.activation is not None:
            out = self.activation(out)
        # out = self.bn(out)
        return out  # , attention



class Module1(nn.Module):
    def __init__(self):
        super(Module1, self).__init__()
        self.conv_v1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (3, 3), stride = 1, padding = 1, bias=True)
        self.conv_v2 = nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=(3, 3), stride=1, padding=0, bias=True)
        self.maxpool_v1 = nn.MaxPool2d(kernel_size=(4, 4), stride=2, padding=0)
        self.bn_v1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv_a1 = nn.Conv2d(in_channels = 2, out_channels = 32, kernel_size = (3, 3), stride = 1, padding = 1, bias=True)
        self.conv_a2 = nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=(3, 3), stride=1, padding=0, bias=True)
        self.maxpool_a1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.bn_a1 = nn.BatchNorm2d(32)

        self.conv_fv1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = 1, padding = 0, bias=True)
        self.conv_fa1 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.maxpool_fa1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1,2), padding=(0,1))
        self.maxpool_fv1 = nn.MaxPool2d(kernel_size=(4, 4), stride=4, padding=0)
        self.cross_att1 = CrossModalAttention(64)
        self.cross_att2 = CrossModalAttention(64)

    def forward(self, face, spect):
        x_v1 = self.conv_v1(face)
        x_v1 = self.relu(x_v1)
        x_v1 = self.conv_v2(x_v1)
        x_v1 = self.bn_v1(x_v1)
        x_v1 = self.relu(x_v1)
        x_v1 = self.maxpool_v1(x_v1)

        x_a1 = self.conv_a1(spect)
        x_a1 = self.relu(x_a1)
        x_a1 = self.conv_a2(x_a1)
        x_a1 = self.relu(x_a1)
        x_a1 = self.maxpool_a1(x_a1)
        x_a1 = self.dropout1(x_a1)
        x_a1 = self.bn_a1(x_a1)

        x_fv1 = self.conv_fv1(x_v1) 
        x_fv1 = self.maxpool_fv1(x_fv1)
        x_fa1 = self.conv_fa1(x_a1)
        x_fa1 = self.maxpool_fa1(x_fa1)
        x_fatt_a1 = self.cross_att1(x_fv1, x_fa1)
        x_fatt_v1 = self.cross_att2(x_fa1, x_fv1)

        x_fm1 = torch.cat([x_fatt_a1, x_fatt_v1], dim=1)
        return x_v1, x_a1, x_fm1


class Module2(nn.Module):
    def __init__(self):
        super(Module2, self).__init__()
        self.conv_v3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.conv_v4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0, bias=True)
        self.maxpool_v2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.bn_v2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv_a3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.conv_a4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0, bias=True)
        self.maxpool_a2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.bn_a2 = nn.BatchNorm2d(64)

        self.conv_fv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (5, 5), stride = 1, padding = 0, bias=True)
        self.conv_fa2 = nn.Conv2d(in_channels = 64, out_channels =128, kernel_size = (3, 3), stride=1, padding=1, bias=True)
        self.maxpool_fa2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1,2), padding=0)
        self.maxpool_fv2 = nn.MaxPool2d(kernel_size=(4, 4), stride=4, padding=0)
        self.conv_fm1 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (5, 5), stride = 1, padding = 1, bias=True)
        self.maxpool_fm1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.cross_att1 = CrossModalAttention(128)
        self.cross_att2 = CrossModalAttention(128)

    def forward(self, x_v1, x_a1, x_fm1):
        x_v2 = self.conv_v3(x_v1)
        x_v2 = self.relu(x_v2)
        x_v2 = self.conv_v4(x_v2)
        x_v2 = self.bn_v2(x_v2)
        x_v2 = self.relu(x_v2)
        x_v2 = self.maxpool_v2(x_v2)

        x_a2 = self.conv_a3(x_a1)
        x_a2 = self.relu(x_a2)
        x_a2 = self.conv_a4(x_a2)
        x_a2 = self.relu(x_a2)
        x_a2 = self.maxpool_a2(x_a2)
        x_a2 = self.dropout2(x_a2)
        x_a2 = self.bn_a2(x_a2)

        x_fv2 = self.conv_fv2(x_v2) 
        x_fv2 = self.maxpool_fv2(x_fv2)
        x_fa2 = self.conv_fa2(x_a2)
        x_fa2 = self.maxpool_fa2(x_fa2)
        x_fm2 = self.conv_fm1(x_fm1) 
        x_fm2 = self.maxpool_fm1(x_fm2)

        x_fatt_a2 = self.cross_att1(x_fv2, x_fa2)
        x_fatt_v2 = self.cross_att2(x_fa2, x_fv2)

        x_fm2 = torch.cat([x_fm2, x_fatt_a2, x_fatt_v2], dim=1)
        return  x_v2, x_a2, x_fm2


class Module3(nn.Module):
    def __init__(self):
        super(Module3, self).__init__()
        self.conv_v5 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.conv_v6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=0, bias=True)
        self.maxpool_v3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.bn_v3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.conv_a5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.conv_a6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=0, bias=True)
        self.maxpool_a3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.bn_a3 = nn.BatchNorm2d(128)

        self.conv_fv3 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = 0, bias=True)
        self.conv_fa3 = nn.Conv2d(in_channels = 128, out_channels =256, kernel_size = (1, 1), stride=1, padding=1, bias=True)
        self.maxpool_fa3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1,2), padding=0)
        self.maxpool_fv3 = nn.MaxPool2d(kernel_size=(4, 4), stride=4, padding=0)
        self.conv_fm2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = 1, bias=True)
        self.maxpool_fm2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.cross_att1 = CrossModalAttention(256)
        self.cross_att2 = CrossModalAttention(256)

    def forward(self, x_v2, x_a2, x_fm2):
        x_v3 = self.conv_v5(x_v2)
        x_v3 = self.relu(x_v3)
        x_v3 = self.conv_v6(x_v3)
        x_v3 = self.bn_v3(x_v3)
        x_v3 = self.relu(x_v3)
        x_v3 = self.maxpool_v3(x_v3)

        x_a3 = self.conv_a5(x_a2)
        x_a3 = self.relu(x_a3)
        x_a3 = self.conv_a6(x_a3)
        x_a3 = self.relu(x_a3)
        x_a3 = self.maxpool_a3(x_a3)
        x_a3 = self.dropout3(x_a3)
        x_a3 = self.bn_a3(x_a3)

        x_fv3 = self.conv_fv3(x_v3) 
        x_fv3 = self.maxpool_fv3(x_fv3)
        x_fa3 = self.conv_fa3(x_a3)
        x_fa3 = self.maxpool_fa3(x_fa3)
        x_fm3 = self.conv_fm2(x_fm2) 
        x_fm3 = self.maxpool_fm2(x_fm3)

        x_fatt_a3 = self.cross_att1(x_fv3, x_fa3)
        x_fatt_v3 = self.cross_att2(x_fa3, x_fv3)

        x_fm3 = torch.cat([x_fm3, x_fatt_a3, x_fatt_v3], dim=1)
        return x_v3, x_a3, x_fm3

class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor, self).__init__()
        self.module1 = Module1()
        self.module2 = Module2()
        self.module3 = Module3()

    def forward(self, faces, spects):
        x_v1, x_a1, x_fm1 = self.module1(faces, spects)
        x_v2, x_a2, x_fm2 = self.module2(x_v1, x_a1, x_fm1)
        x_v3, x_a3,  x_fm3 = self.module3(x_v2, x_a2, x_fm2)
        return x_v3, x_a3,  x_fm3

class Head_fusion(nn.Module):
    def __init__(self):
        super(Head_fusion, self).__init__()
        self.bn_f = nn.BatchNorm2d(1024)
        self.fc_f =  nn.Linear(1024, 2)

    def forward(self, x_fm3):
        x_f = self.bn_f(x_fm3)
        x_f = F.adaptive_avg_pool2d(x_f, (1, 1))
        x_f = x_f.view(x_f.size(0), -1)
        x_f = self.fc_f(x_f)
        return x_f

class Head_vision(nn.Module):
    def __init__(self):
        super(Head_vision, self).__init__()
        self.bn_v = nn.BatchNorm2d(512)
        self.fc_v=  nn.Linear(512, 2)
        self.conv_v1 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = 0, bias=True)
        self.conv_v2 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = 0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_v3):
        x_v = self.conv_v1(x_v3)
        x_v = self.relu(x_v)
        x_v = self.conv_v2(x_v)
        x_v = self.bn_v(x_v)
        x_v = F.adaptive_avg_pool2d(x_v, (1, 1))
        x_v = x_v.view(x_v.size(0), -1)
        x_v = self.fc_v(x_v)
        return x_v

class Head_acoustic(nn.Module):
    def __init__(self):
        super(Head_acoustic, self).__init__()
        self.bn_a = nn.BatchNorm2d(512)
        self.fc_a=  nn.Linear(512, 2)
        self.conv_a1 = nn.Conv2d(in_channels = 128, out_channels =256, kernel_size = (1, 1), stride=1, padding=0, bias=True)
        self.conv_a2 = nn.Conv2d(in_channels = 256, out_channels =512, kernel_size = (1, 1), stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_a3):
        x_a = self.conv_a1(x_a3)
        x_a = self.relu(x_a)
        x_a = self.conv_a2(x_a)
        x_a = self.bn_a(x_a)
        x_a = F.adaptive_avg_pool2d(x_a, (1, 1))
        x_a = x_a.view(x_a.size(0), -1)
        x_a = self.fc_a(x_a)
        return x_a

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.feat_exct = Feature_extractor()
        self. head_fusion = Head_fusion()
        self. head_vision = Head_vision()
        self. head_acoustic = Head_acoustic()
    
    def forward(self, faces, spects):
        x_v, x_a,  x_f = self.feat_exct(faces, spects)
        out_f = self. head_fusion(x_f)
        out_v = self. head_vision(x_v)
        out_a = self. head_acoustic(x_a)
        return out_f, out_v, out_a



if __name__ == "__main__":
    faces = torch.rand(size=(8, 3, 128, 128))
    spects = torch.rand(size=(8, 2, 33, 61)) 
    MODEL = Classifier()
    x_f, x_v, x_a = MODEL(faces, spects)
    print(x_f.size(), x_v.size(), x_a.size())
    # from thop import profile
    # macs, params = profile(MODEL, inputs=(faces,spects))
    # print(macs/(1000**3), params/(1000**2))