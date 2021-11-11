# tf.keras-gradcamplusplus

![example](https://img.shields.io/badge/Python-3.x-blue.svg) ![example](https://img.shields.io/badge/Tensorflow-2.x-yellow.svg) ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

Grad-Cam and Grad-Cam++ implemented in tf.keras 2.X (tensorflow 2.X).

Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization by Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra (https://arxiv.org/abs/1610.02391).

Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks by Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, Vineeth N Balasubramanian (https://arxiv.org/abs/1710.11063).

Adapted and optimized code from https://github.com/totti0223/gradcamplusplus.

# Description
Resolve the problem of using eager mode in tf.keras, and almost follow the formula in grad-cam++ paper.

# Results
![result](https://i.imgur.com/FjmSw3g.jpg)

For more results, check the images in [results](https://github.com/samson6460/tf.keras-gradcamplusplus/tree/master/results) folder.

# Usage
1. Execute following command in terminal:
```
git clone https://github.com/samson6460/tf.keras-gradcamplusplus.git
cd tf.keras-gradcamplusplus
```
2. Create a new python file and import `utils` and `gradcam`.
3. Pass your model and image array to `grad_cam()` or `grad_cam_plus()` func, and it will return a heatmap.
4. Pass image path and heatmap to `show_imgwithheat()` func, and it will show a superimposed image.

# Example
Here's an example model that can classify bone X-rays into three categories: wrist, shoulder and elbow based on VGG16.

The model was pretrained on ImageNet and finetuned on **MURA** dataset.

Get the model by calling `vgg16_mura_model(destination_path)`.If it's the first time it will download the weights automatically.

Get the MURA(musculoskeletal radiographs) dataset from https://stanfordmlgroup.github.io/competitions/mura/.

Or test the model with no copyright images in [images](https://github.com/samson6460/tf.keras-gradcamplusplus/tree/master/images) folder.

Images source:
- http://pixabay.com/
- https://visualhunt.com/

Run *example.py*, you will understand more.[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pRlq73Wkd5np3mV-clZOhxmkdnGgpoec?usp=sharing)