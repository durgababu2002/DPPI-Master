import torch
import torch.nn as nn

class ConvModule(nn.Module):
    def __init__(self, num_features, crop_length):
        super(ConvModule, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 4)),

            nn.Conv2d(64, 128, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 4)),

            nn.Conv2d(128, 256, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 4)),

            nn.Conv2d(256, 512, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AvgPool2d((1, int(crop_length / 64))),
            nn.Flatten()
        )

        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

class DPPI_Model(nn.Module):
    def __init__(self, num_features=20, crop_length=512):
        super(DPPI_Model, self).__init__()

        self.conv = ConvModule(num_features, crop_length)

        self.rp1_linear = nn.Linear(512, 512, bias=True)
        self.rp1_bn = nn.BatchNorm1d(512, eps=1e-5, momentum=0.1)
        self.rp1_relu = nn.ReLU(inplace=True)

        self.rp2_linear = nn.Linear(512, 512, bias=True)
        self.rp2_bn = nn.BatchNorm1d(512, eps=1e-5, momentum=0.1)
        self.rp2_relu = nn.ReLU(inplace=True)

        nn.init.normal_(self.rp1_linear.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.rp1_linear.bias, 0.0)
        nn.init.normal_(self.rp2_linear.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.rp2_linear.bias, 0.0)

        
        for p in self.rp1_linear.parameters():
            p.requires_grad = False
        for p in self.rp2_linear.parameters():
            p.requires_grad = False

        self.final = nn.Linear(1024, 1)
        nn.init.constant_(self.final.bias, 0.0)

    def forward(self, protein_a, protein_b):
        fa = self.conv(protein_a) 
        fb = self.conv(protein_b)   

        a1 = self.rp1_relu(self.rp1_bn(self.rp1_linear(fa)))
        a2 = self.rp2_relu(self.rp2_bn(self.rp2_linear(fa)))

        b1 = self.rp1_relu(self.rp1_bn(self.rp1_linear(fb)))
        b2 = self.rp2_relu(self.rp2_bn(self.rp2_linear(fb)))

        r1 = torch.cat([a1, a2], dim=1)
        r2 = torch.cat([b2, b1], dim=1)

        interaction = r1 * r2

        logits = self.final(interaction)

        return logits
