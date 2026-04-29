import torch
import torch.nn as nn
import torch.nn.functional as F

class spatialExtractor(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(spatialExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_channels)
        self.conv5 = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNSignEmbedding(nn.Module):
    def __init__(self, embedding_dim=256, num_landmarks=42, input_channels=3, input_frames=30):
        super(TCNSignEmbedding, self).__init__()

        self.spatial_extractor = spatialExtractor(input_channels, 16, 32)

        self.dim_reduction = nn.Conv1d(32 * num_landmarks, embedding_dim, kernel_size=1)

        num_channels = [embedding_dim, embedding_dim, embedding_dim]
        self.tcn = nn.Sequential(
            TemporalBlock(num_channels[0], num_channels[1], kernel_size=3, stride=1, dilation=1, padding=1),
            TemporalBlock(num_channels[0], num_channels[1], kernel_size=3, stride=1, dilation=1, padding=1),
            TemporalBlock(num_channels[0], num_channels[1], kernel_size=3, stride=1, dilation=1, padding=1),
            TemporalBlock(num_channels[1], num_channels[2], kernel_size=3, stride=1, dilation=2, padding=2),
        )

        self.threshold = nn.Parameter(torch.tensor(4.0))


    def forward(self, x):
        batch_size, frames, landmarks, channels = x.size()

        x = x.permute(0, 1, 3, 2).contiguous().view(-1, channels, landmarks, 1)

        x = self.spatial_extractor(x)

        _, features, _, _ = x.size()
        x = x.view(batch_size, frames, features, landmarks).permute(0, 2, 3, 1)

        x = x.contiguous().view(batch_size, -1, frames)

        x = self.dim_reduction(x)

        x = self.tcn(x)

        x = x.permute(0, 2, 1).contiguous()

        return x

def get_model(path = 'server/model/weights.h5', device='cpu'):
    model = TCNSignEmbedding().to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model
