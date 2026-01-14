import timm
import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

from garmentds.keypoint_detection.models.backbones.base_backbone import Backbone
from garmentds.keypoint_detection.models.backbones.convnext_unet import UpSamplingBlock


class ConvBlock(nn.Module):
    """
    self-defined convolution block with batchnorm and relu activation
    """

    def __init__(self, n_channels_in, n_channels_out, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=n_channels_in,
            out_channels=n_channels_out,
            kernel_size=kernel_size,
            bias=False,
        )

        self.norm1 = nn.BatchNorm2d(n_channels_out)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        return x

class MaxVitUnet(Backbone):
    """
    Pretrained MaxVit(MBConv (Efficient Net) + Blocked Local Attention + Grid global attention) as Encoder for the U-Net.

    the outputs of the stem and all Multi-Axis (Max) stages are used as feature layers
    note that the paper uses only stage 2-4 for segmentation w/ Mask-RCNN.

    maxvit_nano_rw_256 is a version trained on 256x256 images in timm, that differs slightly from the paper
    but is a much more lightweight model (approx. 15M params)

    It is approx 4 times slower than the ConvNeXt femto backbone (5M params), and still
    about 2 times slower than convnext_nano @ 15M params, yet provided better results
    than both convnext variants in some initial experiments.

    The model can deal with input sizes divisible by 32, but for pretrained weights you are restricted to multiples of the pretrained
    models: 224, 256, 384. From the accompanying notebook, it seems that the model easily handles images that are 3 times as big as the
    training size. (see https://github.com/rwightman/pytorch-image-models/issues/1475 for more details)

    For now only 256 is supported so input sizes are restricted to 256,512,...



    orig                    ---   1/1  -->                       --->       (head)
        stem                ---   1/2  -->             decode4
            stage 1         ---   1/4  -->         decode3
                stage 2     ---   1/8  -->     decode2
                    stage 3 ---   1/16 --> decode1
                        stage 4 ---1/32----|
    """

    # 15M params
    FEATURE_CONFIG = [
        {"down": 2, "channels": 64},
        {"down": 4, "channels": 64},
        {"down": 8, "channels": 128},
        {"down": 16, "channels": 256},
        {"down": 32, "channels": 512},
    ]
    MODEL_NAME = "maxvit_nano_rw_256"
    feature_layers = ["stem", "stages.0", "stages.1", "stages.2", "stages.3"]

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.pass_rgb_to_decoder = kwargs.get("pass_rgb_to_decoder")
        self.use_conv_on_decoder_output = kwargs.get("use_conv_on_decoder_output")
        self.encoder = timm.create_model(self.MODEL_NAME, pretrained=True, num_classes=0)
        self.feature_extractor = create_feature_extractor(self.encoder, self.feature_layers)
        self.decoder_blocks = nn.ModuleList()

        # used on decoder output, try to learn structure information
        self.weight_blocks = nn.ModuleList()

        for config_skip, config_in in zip(self.FEATURE_CONFIG, self.FEATURE_CONFIG[1:]):
            if self.pass_rgb_to_decoder:
                block = UpSamplingBlock(config_in["channels"], config_skip["channels"]+3, config_skip["channels"], 3)
            else:
                block = UpSamplingBlock(config_in["channels"], config_skip["channels"], config_skip["channels"], 3)
            self.decoder_blocks.append(block)
            self.weight_blocks.append(ConvBlock(config_in["channels"], config_in["channels"], 1))

        self.final_conv = nn.Conv2d(
            self.FEATURE_CONFIG[0]["channels"], self.FEATURE_CONFIG[0]["channels"], 3, padding="same"
        )
        self.final_upsampling_block = UpSamplingBlock(
            self.FEATURE_CONFIG[0]["channels"], 3, self.FEATURE_CONFIG[0]["channels"], 3
        )

    def forward(self, x):
        orig_x = torch.clone(x)
        features = list(self.feature_extractor(x).values())

        x = features.pop(-1)
        if self.use_conv_on_decoder_output:
            weight_conv = self.weight_blocks.pop(-1)
            x = weight_conv(x)

        idxs = list(range(len(self.decoder_blocks)))
        for block, idx in zip(self.decoder_blocks[::-1], idxs[::-1]):
            feature = features.pop(-1)
            if self.use_conv_on_decoder_output:
                weight_conv = self.weight_blocks.pop(-1)
                feature = weight_conv(feature)

            if self.pass_rgb_to_decoder:
                down_size = self.FEATURE_CONFIG[idx]["down"]
                #pool = torch.nn.AvgPool2d(down_size, down_size)
                pool = torch.nn.MaxPool2d(down_size, down_size)
                x_skip = torch.cat([feature, pool(orig_x)], dim=1)
            else:
                x_skip = feature
            x = block(x, x_skip)

        # x = nn.functional.interpolate(x, scale_factor=2)
        # x = self.final_conv(x)
        x = self.final_upsampling_block(x, orig_x)
        return x

    def get_n_channels_out(self):
        return self.FEATURE_CONFIG[0]["channels"]


class MaxVitPicoUnet(MaxVitUnet):
    MODEL_NAME = "maxvit_rmlp_pico_rw_256"  # 7.5M params.
    FEATURE_CONFIG = [
        {"down": 2, "channels": 32},
        {"down": 4, "channels": 32},
        {"down": 8, "channels": 64},
        {"down": 16, "channels": 128},
        {"down": 32, "channels": 256},
    ]


if __name__ == "__main__":
    from PIL import Image
    import timm
    import cv2
    import numpy as np
    import random

    """
    random_state, np_random_state, torch_random_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    img = Image.open("~/research/garmentds/outputs/data/test_data/tshirt_sp/0/mesh_rendered.png")
    img_np = np.array(img)
    print("img size:", img.size, "type of img: ", type(img))

    model = timm.create_model(
        'maxvit_nano_rw_256',
        pretrained=True,
        features_only=True,
    )
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    print(data_config)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    print(transforms)

    for i in range(1):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        img_transformed = transforms(img)
        print(f"img_transformed shape: {img_transformed.shape}, type: {type(img_transformed)}")
        print(torch.max(img_transformed), torch.min(img_transformed))

    output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

    for o in output:
        # print shape of each feature map in output
        # e.g.:
        #  torch.Size([1, 64, 128, 128])
        #  torch.Size([1, 64, 64, 64])
        #  torch.Size([1, 128, 32, 32])
        #  torch.Size([1, 256, 16, 16])
        #  torch.Size([1, 512, 8, 8])

        if type(o) == torch.Tensor:
            print(o.shape)
        elif type(o) == list:
            print(len(o))
        else:
            print(type(o))

    random.setstate(random_state)
    np.random.set_state(np_random_state)
    torch.random.set_rng_state(torch_random_state)
    """

    """
    #model = timm.create_model("maxvit_rmlp_pico_rw_256")
    model = timm.create_model("maxvit_nano_rw_256")

    feature_extractor = create_feature_extractor(model, ["stem", "stages.0", "stages.1", "stages.2", "stages.3"])
    x = torch.zeros((1, 3, 256, 256), requires_grad=True)
    y = feature_extractor(x)
    print(y.keys(), model(x).shape)
    features = list(feature_extractor(x).values())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"num params = {n_params/10**6:.2f} M")
    feature_config = []
    for x in features:
        print(f"{x.shape=}")
        config = {"down": 256 // x.shape[2], "channels": x.shape[1]}
        feature_config.append(config)
    print(f"{feature_config=}")

    #model = MaxVitPicoUnet()
    #x = torch.zeros((1, 3, 256, 256))
    #y = model(x)
    #print(f"{y.shape=}")
    """


    # get the grad after one forward pass
    model = MaxVitUnet(pass_rgb_to_decoder=False)
    x = torch.randn((1, 3, 256, 256), requires_grad=True)
    y = model(x)
    print(f"{y.shape}")

    """
    import numpy as np
    loss = y[0, 0, 1, 1]
    loss.backward()
    np.save("grad1.npy", x.grad.detach().numpy())
    """