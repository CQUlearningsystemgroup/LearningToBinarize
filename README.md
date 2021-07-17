## Learning to Binarized

This is the source code for the paper Learning to Binarize Convolutional Neural Networks with Adaptive Neural Encoder.

### Dependencies

- Python 3.5
- Pytorch 1.3
- Cuda

### Train

- Download the pretrained teacher model [resnet18.pth](https://drive.google.com/file/d/1qjMwbDy5dEdytfdqsjbksRqdeQ4HFe1G/view?usp=sharing) .
- Modify the run.sh file with your ImageNet dataset directory

```
clear
python3 trainimagenet.py --data /IMAGENET-DATA | tee -a log/log.txt
```

- Run the run.sh file in terminal

```
./run.sh
```

### Reference

- [Bi-Real-net](https://github.com/liuzechun/Bi-Real-net)

- [ReActNet](https://github.com/liuzechun/ReActNet)

- [MetaQuant](https://github.com/csyhhu/MetaQuant)