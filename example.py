# Copyright 2020 Samson Woof

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# %%
from utils import vgg16_mura_model, preprocess_image, show_imgwithheat
from gradcam import grad_cam,grad_cam_plus

# %% load the model
model = vgg16_mura_model('model_weights/VGG16_MURA.h5')
model.summary()

# %%
img_path = 'images/4320878114_30a836d428_z.jpg'
img = preprocess_image(img_path)

# %% result of grad cam
heatmap = grad_cam(model, img,
                   label_name = ['WRIST', 'ELBOW', 'SHOULDER'],
                   #category_id = 0,
                   )
show_imgwithheat(img_path, heatmap)

# %% result of grad cam++
heatmap_plus = grad_cam_plus(model, img)
show_imgwithheat(img_path, heatmap_plus)