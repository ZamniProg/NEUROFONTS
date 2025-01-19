from imports import *


class RPN(nn.Module):
    def __init__(self, in_channels: int, anchors_count=9, kernel_size=(3, 3)):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=kernel_size)
        self.relu = nn.ReLU()

        self.out_deltas = nn.Conv2d(in_channels=512, out_channels=4 * anchors_count, kernel_size=(1, 1))
        self.linear = nn.Linear(9216, 256) # NEED TO RECOMPUTE HERE

        self.out_scores = nn.Conv2d(in_channels=512, out_channels=1 * anchors_count, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv(x))

        out_scores = self.sigmoid(self.out_scores(x))
        x = self.out_deltas(x)
        x = torch.flatten(x, start_dim=1)
        out_deltas = self.linear(x)

        return out_deltas, out_scores

    @staticmethod
    def loss():
        pass


class ROIAlignLayer(nn.Module):
    def __init__(self):
        super(ROIAlignLayer, self).__init__()
        pass

    def forward(self):
        pass