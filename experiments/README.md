# RGB weed segmentation experiment

Compare ImageNet-initialized DeepLabV3 with a MobileNetV3-Large backbone against the measured U-Net32 candidate. Both use the official spatial WeedsGalore split, identical RGB inputs and three output classes at 224 pixels. Keep paired image and mask rotations and cosine learning-rate decay.

```sh
python -m ml.experiments --require-gpu --output data/experiment-weeds-deeplab
python kaggle/prepare_kernel.py weeds --experiment --username YOUR_KAGGLE_USERNAME --output data/kaggle-experiment
```

The runner logs validation mean IoU, weed IoU, held-out metrics and CPU latency in MLflow. It exports logits in the current PlantGuard ONNX format. The U-Net32 candidate scored 74.95% test mean IoU and 64.99% test weed IoU.
