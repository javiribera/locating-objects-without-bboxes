import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,
                 height, width,
                 known_n_points=None,
                 tensortype=torch.FloatTensor):
        super(UNet, self).__init__()

        # With this network depth, there is a minimum image size
        if height < 256 or width < 256:
            raise ValueError('Minimum input image size is 256x256, got {}x{}'.\
                             format(height, width))

        # Type of tensor the output will be
        self.tensortype = tensortype

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.down5 = down(512, 512)
        self.down6 = down(512, 512)
        self.down7 = down(512, 512)
        self.down8 = down(512, 512, normaliz=False)
        self.up1 = up(1024, 512)
        self.up2 = up(1024, 512)
        self.up3 = up(1024, 512)
        self.up4 = up(1024, 512)
        self.up5 = up(1024, 256)
        self.up6 = up(512, 128)
        self.up7 = up(256, 64)
        self.up8 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.out_nonlin = nn.Sigmoid()

        self.known_n_points = known_n_points
        if known_n_points is None:
            self.regressor = nn.Linear(256*256 + 512, 1)
            self.regressor_nonlin = nn.Softplus()

        # This layer is not connected anywhere
        # It is only here for backward compatibility
        self.lin = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)
        x = self.up1(x9, x8)
        x = self.up2(x, x7)
        x = self.up3(x, x6)
        x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        x = self.up8(x, x1)
        x = self.outc(x)
        x = self.out_nonlin(x)

        # Reshape Bx1xHxW -> BxHxW
        # because probability map is real-valued by definition
        x = x.squeeze(1)

        if self.known_n_points is None:
            x_flat = x.view(x.shape[0], -1)
            x9_flat = x9.view(x9.shape[0], -1)
            regression_features = torch.cat((x_flat, x9_flat), dim=1)
            regression = self.regressor(regression_features)
            regression = self.regressor_nonlin(regression)
            return x, regression
        else:
            n_pts = Variable(self.tensortype([self.known_n_points]))
            return x, n_pts
        # summ = torch.sum(x)
        # count = self.lin(summ)

        # count = torch.abs(count)

        # if self.known_n_points is not None:
            # count = Variable(torch.cuda.FloatTensor([self.known_n_points]))

        # return x, count
