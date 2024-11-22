from mmseg_custom.models.backbones.base.vrwkv import VRWKV
import torch

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(1, 3, 512, 512)).cuda()
    vrwkv = VRWKV().cuda()
    total_param_num = sum(p.numel() for p in vrwkv.parameters() if p.requires_grad)  # 模型中的全部参数量
    print("{0} parameters to be trained in total".format(total_param_num))  # 查看可训练的参数量
    print("Input shape:", x.shape)
    y = vrwkv(x)
    print('Output:', y[0].shape)