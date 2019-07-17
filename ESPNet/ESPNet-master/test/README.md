# ESPNet: Towards Fast and Efficient Semantic Segmentation on the Embedded Devices

This folder contains the python scripts for running our pretrained models on the Cityscape dataset.

## Getting Started
We provide the pretrained weights for ESPNet and ESPNet-C. Recall that ESPNet is the same as ESPNet-C, but with light weight decoder.

Pre-requisites: 
* By default, we expect all images inside the ./data directory. If they are in different directory, please change the  **data_dir** argument in the VisualizeResults.py file.

* Also, if the image format is different (e.g. jpg), please change in the VisualizeResults.py file.

This can be done using the below command:

```
python VisualizeResults.py --data_dir <data_dir> --img_extn <image extension>
```


### Running ESPNet-C models
To run the ESPNet-C models, execute the following commands

```
python VisualizeResults.py --modelType 2 --p 2 --q 3
```

Here, p and q are the depth multipliers. Our models only support p=2 and q=3,5,8


### Running ESPNet models
To run the ESPNet models, execute the following commands

```
python VisualizeResults.py --modelType 1 --p 2 --q 3
```

Here, p and q are the depth multipliers. Our models only support p=2 and q=3,5,8
