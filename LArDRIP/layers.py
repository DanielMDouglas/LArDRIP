import torch
import torch.nn as nn
import MinkowskiEngine as ME

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, input):
        return input

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size, name = 'resBlock'):
        super(ResNetBlock, self).__init__()

        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features)
        else:
            self.residual = Identity()
        
        self.norm1 = ME.MinkowskiBatchNorm(in_features)
        self.act1 = ME.MinkowskiReLU()
        self.conv1 = ME.MinkowskiConvolution(in_channels = in_features,
                                             out_channels = out_features,
                                             kernel_size = kernel_size,
                                             stride = 1,
                                             dimension = 3)

        self.norm2 = ME.MinkowskiBatchNorm(out_features)
        self.act2 = ME.MinkowskiReLU()
        self.conv2 = ME.MinkowskiConvolution(in_channels = out_features,
                                             out_channels = out_features,
                                             kernel_size = kernel_size,
                                             stride = 1,
                                             dimension = 3)
        
    def forward(self, x):

        residual = self.residual(x)
        
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))
        out += residual

        return out

class DropoutBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size, name = 'DropoutBlock'):
        super(DropoutBlock, self).__init__()

        self.conv1 = ME.MinkowskiConvolution(in_channels = in_features,
                                             out_channels = out_features,
                                             kernel_size = kernel_size,
                                             stride = 1,
                                             dimension = 3)
        self.dropout1 = ME.MinkowskiDropout()
        self.norm1 = ME.MinkowskiBatchNorm(out_features)
        self.act1 = ME.MinkowskiReLU()

        self.conv2 = ME.MinkowskiConvolution(in_channels = out_features,
                                             out_channels = out_features,
                                             kernel_size = kernel_size,
                                             stride = 1,
                                             dimension = 3)
        self.dropout2 = ME.MinkowskiDropout()
        self.norm2 = ME.MinkowskiBatchNorm(out_features)
        self.act2 = ME.MinkowskiReLU()
        
    def forward(self, x):

        out = self.act1(self.norm1(self.dropout1(self.conv1(x))))
        out = self.act2(self.norm2(self.dropout2(self.conv2(out))))

        return out

class ResNetEncoderBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size, dropout = False, name = 'ResNetEncoderBlock'):
        super(ResNetEncoderBlock, self).__init__()


        if dropout:
            self.convBlock1 = DropoutBlock(in_features, in_features, kernel_size)
            self.convBlock2 = DropoutBlock(in_features, in_features, kernel_size)
        else:
            self.convBlock1 = ResNetBlock(in_features, in_features, kernel_size)
            self.convBlock2 = ResNetBlock(in_features, in_features, kernel_size)
        
        self.downSampleBlock = DownSample(in_features, out_features, 2, dropout = dropout)

    def forward(self, x):
        out = self.convBlock1(x)
        out = self.convBlock2(out)
        out = self.downSampleBlock(out)

        return out

class DownSample(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size, dropout = False, name = 'DownSample'):
        super(DownSample, self).__init__()
        
        self.norm = ME.MinkowskiBatchNorm(in_features)
        self.act = ME.MinkowskiReLU()
        self.conv = ME.MinkowskiConvolution(in_channels = in_features,
                                            out_channels = out_features,
                                            kernel_size = kernel_size,
                                            stride = kernel_size,
                                            dimension = 3)

        self.useDropout = dropout
        self.dropout = ME.MinkowskiDropout()
        
    def forward(self, x):
        
        out = self.conv(self.act(self.norm(x)))
        if self.useDropout:
            out = self.dropout(out)

        return out

class ResNetEncoder(torch.nn.Module):
    def __init__(self, in_features, kernel_size = 3, depth = 2, nFilters = 16, name='uresnet'):
        super(ResNetEncoder, self).__init__()

        self.depth = depth # number of pool/unpool layers, not including input + output
        self.nFilters = nFilters
        self.in_features = in_features
        self.kernel_size = kernel_size
        
        self.input_block = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = self.in_features,
                out_channels = self.nFilters,
                kernel_size = self.kernel_size,
                stride = 1,
                dimension = 3,
            ) 
        )

        self.featureSizesEnc = [(self.nFilters*2**i, self.nFilters*2**(i+1))
                                for i in range(self.depth)]
        
        self.encoding_layers = []

        self.encoding_blocks = []
        
        for i in range(self.depth):
            self.encoding_layers.append(
                ME.MinkowskiConvolution(
                    in_channels = self.featureSizesEnc[i][0],
                    out_channels = self.featureSizesEnc[i][1],
                    kernel_size = 2,
                    stride = 2,
                    dimension = 3)
            )
            self.encoding_blocks.append(
                nn.Sequential(
                    ResNetBlock(self.featureSizesEnc[i][1],
                                self.featureSizesEnc[i][1],
                                self.kernel_size),
                    ResNetBlock(self.featureSizesEnc[i][1],
                                self.featureSizesEnc[i][1],
                                self.kernel_size),
                )
            )
        self.encoding_layers = nn.Sequential(*self.encoding_layers)
        self.encoding_blocks = nn.Sequential(*self.encoding_blocks)

    def forward(self, x):
        encodingFeatures = []
        coordKeys = []

        out = self.input_block(x)
        for i in range(self.depth):
            encodingFeatures.append(Identity()(out))
            coordKeys.append(out.coordinate_map_key)

            out = self.encoding_layers[i](out)
            out = self.encoding_blocks[i](out)

        return out
