from imports import *


class ConvolutionNN(nn.Module):
    """
    NEED TO UPDATE SQ OF LAYERS
    shape: (N, C, H, W)
    """
    def __init__(self, input_shape: tuple[int, int, int] = (1, 1, 300, 300), conv_filter_size: int = 32,
                 kernel_size: tuple[int, int] = (3, 3), pool_size: tuple[int, int] = (2, 2)):
        super(ConvolutionNN, self).__init__()
        # 1 300 300
        self.conv1 = nn.Conv2d(in_channels=input_shape[0],
                               out_channels=conv_filter_size,
                               kernel_size=kernel_size,
                               padding=1,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(conv_filter_size)
        # 32 300 300
        self.conv2 = nn.Conv2d(in_channels=conv_filter_size,
                               out_channels=conv_filter_size * 2,
                               kernel_size=pool_size,
                               stride=2)
        self.bn2 = nn.BatchNorm2d(conv_filter_size * 2)
        # 64 150 150
        self.conv3 = nn.Conv2d(in_channels=conv_filter_size * 2,
                               out_channels=conv_filter_size * 4,
                               kernel_size=kernel_size,
                               padding=2,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(conv_filter_size * 4)
        # 128 152 152
        self.mpl1 = nn.MaxPool2d(pool_size, stride=1, padding=1)
        # 128 153 153
        self.conv4 = nn.Conv2d(in_channels=conv_filter_size * 4,
                               out_channels=conv_filter_size * 8,
                               kernel_size=kernel_size,
                               padding=2,
                               stride=1)
        self.bn4 = nn.BatchNorm2d(conv_filter_size * 8)
        # 256 155 155
        self.conv5 = nn.Conv2d(in_channels=conv_filter_size * 8,
                               out_channels=conv_filter_size * 16,
                               kernel_size=pool_size,
                               stride=1)
        self.bn5 = nn.BatchNorm2d(conv_filter_size * 16)
        # 512 154 154
        self.conv6 = nn.Conv2d(in_channels=conv_filter_size * 16,
                               out_channels=conv_filter_size * 32,
                               kernel_size=kernel_size,
                               stride=1)
        self.bn6 = nn.BatchNorm2d(conv_filter_size * 32)
        # 1024 152 152
        self.mpl2 = nn.MaxPool2d(pool_size, stride=2)
        # 1024 75 75
        self.conv7 = nn.Conv2d(in_channels=conv_filter_size * 32,
                               out_channels=conv_filter_size * 48,
                               kernel_size=pool_size,
                               stride=1)
        self.bn7 = nn.BatchNorm2d(conv_filter_size * 48)
        # 1536 74 74
        self.conv8 = nn.Conv2d(in_channels=conv_filter_size * 48,
                               out_channels=conv_filter_size * 32,
                               kernel_size=kernel_size,
                               stride=2)
        self.bn8 = nn.BatchNorm2d(conv_filter_size * 32)
        # 1024 36 36
        self.mpl3 = nn.MaxPool2d(pool_size, stride=2)
        # 1024 18 18
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.mpl1(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.mpl2(x)
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(self.mpl3(x))
        return x # shape: (N, 1024, 18, 18)
