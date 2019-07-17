# ESPNet: Towards Fast and Efficient Semantic Segmentation on the Embedded Devices

This folder contains the python scripts for training models on the Cityscape dataset.


## Getting Started

### Training ESPNet-C

You can start training the model using below command:

```
python main.py 
```

By default, **ESPNet-C** will be trained with p=2 and q=8. Since the spatial dimensions of the output of ESPNet-C are 1/8th of original image size, please set scaleIn parameter to 8. If you want to change the parameters, you can do so by using the below command:

```
python main.py --scaleIn 8 --p <value of p> --q <value of q>

Example:

python main.py --scaleIn 8 --p 2 --q 8
```

### Training ESPNet
Once you are done training the ESPNet-C, you can attach the light-weight decoder and train the ESPNet model

```
python main.py --scaleIn 1 --p <value of p> --q <value of q> --decoder True --pretrained <path of the pretrained ESPNet-C file>

Example: 

python main.py --scaleIn 1 --p 2 --q 8 --decoder True --pretrained ../pretrained/encoder/espnet_p_2_q_8.pth
```

**Note 1:** Currently, we support only single GPU training. If you want to train the model on multiple-GPUs, you can use **nn.DataParallel** api provided by PyTorch.

**Note 2:** To train on a specific GPU (single), you can specify the GPU_ID using the CUDA_VISIBLE_DEVICES as:

```
CUDA_VISIBLE_DEVICES=2 python main.py
```

This will run the training program on GPU with ID 2.
