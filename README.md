# animefaceMAE
## Requirements
    python3
    torch
    torchvision
    PIL
    matplotlib
    numpy
    seaborn

## Files
    -animefaceMAE
        README.md
        -data
            -face2animetest
                -testB
                    xxx.jpg    #validation data
            -face2animetrain
                -trainB
                    xxx.jpg    #training data
        -logs
            xxx.txt            #training/validation loss info
            xxx.png            #training/validation loss fig
        -model
            xxx.pkl            #saved models
        train.py               #for training
        visualize.py           #for visualization
        models.py              #MAE models
        utils.py               #position embedding

## Training
    python3 train.py
Before this command, you can change some parameters in the file train.py.

TRAIN_DATA_DIR: training data folder

VALID_DATA_DIR: validation data folder

BATCH_SIZE: batch size

device: device

PATCH_SIZE: patch size

EMBED_DIM: embedding dimension, should be changed with patch size. eg: patchsize=16,embeddim=16\*16\*3

EPOCHS: training epochs

SAVING: saving internals

MASKRATE: masking rate \* 100 when training. eg: 75, 87.5

modelname: saving model name

logfile: saving loss file name. logfile.txt and logfile.png

## Visualization
    python3 visualize.py
Before this command, you can change some parameters in the file visualize.py.

device: device

MAE: model path

IMAGEFILE: image path

MASKRATE1: the masking rate of original model

MASKRATE2: the masking rate of validation

The visualization result would be saved as vis_MASKRATE1_MASKRATE2.png.

## Report

For more information, please refer to google docs. https://docs.google.com/document/d/1TIlKBpxeiT4c0bT-ES2kIWLHVIIWN2V-vk4tsOw6c5Q/edit?usp=sharing