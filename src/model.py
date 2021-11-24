import math
import torch
from torch import nn
from torchsummary import summary

class Pixel_Shuffle(nn.Module):
    def __init__(self, scale_factor=2):
        super(Pixel_Shuffle, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.size()

        out_channels = in_channels // (self.scale_factor)
        out_height = in_height * self.scale_factor
        out_width = in_width

        # View as [B, Cout, r, Hin, Win]
        input_view = x.contiguous().view(batch_size, out_channels, self.scale_factor, in_height, in_width)

        # Permute as [B, Cout, Hin, r, Win]
        shuffle_out = input_view.permute(0, 1, 3, 2, 4).contiguous()

        # View as [B, Cout, Hout, Wout]
        return shuffle_out.view(batch_size, out_channels, out_height, out_width)


class ESPCN(nn.Module):
    def __init__(self, scale_factor=2, num_channels=1):
        super(ESPCN, self).__init__()
        self.part1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, padding=5//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=3//2),
            nn.ReLU()
        )
        self.part2 = nn.Conv2d(32, num_channels*(scale_factor), kernel_size=3, padding=3//2)
        self.part3 = Pixel_Shuffle(scale_factor)

    def _initalize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)

        return x        


class UNet_Upsample(nn.Module):
    def make_up_block(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), output_padding=(1,1)):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def make_conv_block(self, in_channels, out_channels, kernel_size=(3,3), padding=(1,1)):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def __init__(self, scale_factor=2, num_channels=1):
        super(UNet_Upsample, self).__init__()
        self.scale_factor = scale_factor
        if scale_factor == 2:
            self.up_block1 = self.make_up_block(num_channels, 64, stride=(2,1), output_padding=(1,0))
        elif scale_factor == 4:
            self.up_block1_1 = self.make_up_block(num_channels, 64, stride=(2,1), output_padding=(1,0))
            self.up_block1_2 = self.make_up_block(64, 64, stride=(2,1), output_padding=(1,0))

        self.up_block2 = self.make_up_block(1024, 512, stride=(2,2), output_padding=(1,1))
        self.up_block3 = self.make_up_block(512, 256, stride=(2,2), output_padding=(1,1))
        self.up_block4 = self.make_up_block(256, 128, stride=(2,2), output_padding=(1,1))
        self.up_block5 = self.make_up_block(128, 64, stride=(2,2), output_padding=(1,1))

        self.conv_block1 = self.make_conv_block(64, 64)
        self.conv_block2 = self.make_conv_block(64, 128)
        self.conv_block3 = self.make_conv_block(128, 256)
        self.conv_block4 = self.make_conv_block(256, 512)
        self.conv_block5 = self.make_conv_block(512, 1024)
        self.conv_block6 = self.make_conv_block(1024, 512)
        self.conv_block7 = self.make_conv_block(512, 256)
        self.conv_block8 = self.make_conv_block(256, 128)
        self.conv_block9 = self.make_conv_block(128, 64)

        self.avgpool_2d = nn.AvgPool2d(kernel_size=(2,2))
        self.dropout = nn.Dropout(p=0.25)

        self.f_conv = nn.Conv2d(in_channels=64, out_channels=num_channels, kernel_size=(1,1))

    def forward(self, x):
        if self.scale_factor == 2:
            x0 = self.up_block1(x)
        elif self.scale_factor == 4:
            x0 = self.up_block1_1(x)
            x0 = self.up_block1_2(x0)
        x1 = self.conv_block1(x0)
        
        x2 = self.avgpool_2d(x1)
        x2 = self.dropout(x2)
        x2 = self.conv_block2(x2)
        
        x3 = self.avgpool_2d(x2)
        x3 = self.dropout(x3)
        x3 = self.conv_block3(x3)

        x4 = self.avgpool_2d(x3)
        x4 = self.dropout(x4)
        x4 = self.conv_block4(x4)

        y4 = self.avgpool_2d(x4)
        y4 = self.dropout(y4)
        y4 = self.conv_block5(y4)
        y4 = self.dropout(y4)
        y4 = self.up_block2(y4)

        y3 = torch.cat((x4, y4), dim=1)
        y3 = self.conv_block6(y3)
        y3 = self.dropout(y3)
        y3 = self.up_block3(y3)

        y2 = torch.cat((x3, y3), dim=1)
        y2 = self.conv_block7(y2)
        y2 = self.dropout(y2)
        y2 = self.up_block4(y2)
        
        y1 = torch.cat((x2, y2), dim=1)
        y1 = self.conv_block8(y1)
        y1 = self.dropout(y1)
        y1 = self.up_block5(y1)

        y0 = torch.cat((x1, y1), dim=1)
        y0 = self.conv_block9(y0)

        y0 = self.f_conv(y0)
        y0 = nn.ReLU()(y0)

        return y0

if __name__ == '__main__':
    # """ Pixel_Shuffle test """
    # x = torch.rand(8, 2, 16, 2650).to('cuda')
    # model = Pixel_Shuffle(scale_factor=2).cuda()
    # out = model(x)
    # print(out.size())

    # x = torch.rand(8, 2, 16, 2650).to('cuda')
    # model = Pixel_Shuffle(scale_factor=4).cuda()
    # out = model(x)
    # print(out.size())


    # """ ESPCN test """
    # x = torch.rand(8, 1, 16, 2048).to('cuda')
    # model = ESPCN(scale_factor=4, num_channels=1).cuda()
    # out = model(x)
    # print(out.size())

    # x = torch.rand(8, 1, 16, 2650).to('cuda')
    # model = ESPCN(scale_factor=4, num_channels=1).cuda()
    # out = model(x)
    # print(out.size())

    """ UNet_Upsample test """
    # x = torch.rand(8, 2, 16, 2048).to('cuda')
    # model = UNet_Upsample(scale_factor=2, num_channels=2).cuda()
    # out = model(x)
    # print(out.size())

    num_channel = 2

    x = torch.rand(4, num_channel, 16, 2048).to('cuda')
    model = UNet_Upsample(scale_factor=4, num_channels=num_channel).cuda()
    out = model(x)

    summary(model, (num_channel, 16, 2048))
    print(out.size())
    pass        