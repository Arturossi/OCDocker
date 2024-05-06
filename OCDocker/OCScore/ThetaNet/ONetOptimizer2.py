import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomUNet(nn.Module):
    def __init__(self, in_channels, out_channels, init_type='he'):
        super(CustomUNet, self).__init__()
        # Define the U-Net structure
        self.encoder1 = self.contracting_block(in_channels, 64)
        self.encoder2 = self.contracting_block(64, 128)
        self.encoder3 = self.contracting_block(128, 256)
        self.encoder4 = self.contracting_block(256, 512)
        self.bottleneck = self.bottleneck_block(512, 1024)
        self.decoder4 = self.expansive_block(1024, 512, 256)
        self.decoder3 = self.expansive_block(512, 256, 128)
        self.decoder2 = self.expansive_block(256, 128, 64)
        self.decoder1 = self.expansive_block(128, 64, out_channels)
        # Initialize weights
        self.apply(self.initialize_weights(init_type))

    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def expansive_block(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels, out_channels, 2, stride=2)
        )

    def bottleneck_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # U-Net Forward pass
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        b = self.bottleneck(e4)
        d4 = self.decoder4(b)
        d3 = self.decoder3(d4 + e3)
        d2 = self.decoder2(d3 + e2)
        d1 = self.decoder1(d2 + e1)
        return d1

    def initialize_weights(self, init_type):
        def init(m):
            if isinstance(m, nn.Conv2d):
                if init_type == 'he':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        return init

class ONet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ONet, self).__init__()
        self.unet1 = CustomUNet(in_channels, out_channels, init_type='he')
        self.unet2 = CustomUNet(in_channels, out_channels, init_type='xavier')

    def forward(self, x):
        output1 = self.unet1(x)
        output2 = self.unet2(x)
        # Combine outputs, could use concatenation, addition, etc.
        combined_output = output1 + output2
        return combined_output


class ONet(nn.Module): # Promising! TODO: Also make another of this, but with a middle simpler network to deal with ouputs from upper and lower bounds and then call if Theta network
    def __init__(self, in_channels, out_channels):
        super(ONet, self).__init__()
        
        ## Upper bound

        # Encoder (contracting path)
        self.encoder1_up = self.contracting_block(in_channels, 64, apply_pooling=True)
        self.encoder2_up = self.contracting_block(64, 128, apply_pooling=True)
        
        # Bottleneck
        self.bottleneck_up = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (expansive path)
        self.decoder2_up = self.expansive_block(256, 128, 64)
        self.decoder1_up = self.expansive_block(128, 64, out_channels)

        ## Lower bound

        # Encoder (contracting path)
        self.encoder1_down = self.contracting_block(in_channels, 512, apply_pooling=True)
        self.encoder2_down = self.contracting_block(512, 1024, apply_pooling=True)
        
        # Bottleneck
        self.bottleneck_down = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (expansive path)
        self.decoder2_down = self.expansive_block(2048, 1024, 512)
        self.decoder1_down = self.expansive_block(1024, 512, out_channels)

        # Final Convolution
        self.final_conv = nn.Conv2d(out_channels * 2, desired_out_channels, kernel_size=1)

        # Initialize weights differently for upper and lower bounds
        self.encoder1_up.apply(self.init_weights_he)
        self.encoder2_up.apply(self.init_weights_he)
        self.bottleneck_up.apply(self.init_weights_he)
        self.decoder2_up.apply(self.init_weights_he)
        self.decoder1_up.apply(self.init_weights_he)

        self.encoder1_down.apply(self.init_weights_xavier)
        self.encoder2_down.apply(self.init_weights_xavier)
        self.bottleneck_down.apply(self.init_weights_xavier)
        self.decoder2_down.apply(self.init_weights_xavier)
        self.decoder1_down.apply(self.init_weights_xavier)

    def contracting_block(self, in_channels, out_channels, apply_pooling=True, use_dropout=False, dropout_prob=0.5):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Batch normalization before activation
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_prob))  # Dropout after activation
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))  # Another batch normalization
        layers.append(nn.ReLU(inplace=True))
        if apply_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def expansive_block(self, in_channels, mid_channels, out_channels, use_dropout=False, dropout_prob=0.5):
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_prob))
        layers.append(nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def init_weights_he(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights_xavier(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        encoder1_up   = self.encoder1_up(x)
        encoder1_down = self.encoder1_down(x)

        encoder2_up = self.encoder2_up(torch.cat([encoder1_up, encoder1_down], dim=1))
        encoder2_down = self.encoder2_down(encoder1_down)
        
        # Bottleneck
        bottleneck_up = self.bottleneck_up(encoder2_up)
        bottleneck_down = self.bottleneck_down(encoder2_down)
        
        # Decoder
        decoder4_up = self.decoder4_up(bottleneck_up)
        decoder3_up = self.decoder3_up(torch.cat([decoder4_up, encoder3_up], dim=1))

        decoder2_down = self.decoder2_down(bottleneck_down)
        decoder2_up = self.decoder2_up(torch.cat([bottleneck_up, bottleneck_down], dim=1))

        decoder1_down = self.decoder1_down(decoder2_down)
        decoder1_up = self.decoder1_up(torch.cat([decoder2_up, encoder1_up, decoder1_down, encoder1_down], dim=1))

        final_output = self.final_conv(torch.cat([decoder1_up, decoder1_down], dim=1))
        
        return final_output
