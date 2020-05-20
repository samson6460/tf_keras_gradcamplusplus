# %%
from utils import vgg16_mura_model,preprocess_image,show_imgwithheat
from gradcam import grad_cam,grad_cam_plus

# %% load the model
model = vgg16_mura_model('model_weights/VGG16_MURA.h5')
model.summary()

# %%
img_path = 'images/patient01347_study1_negative_image2.png'
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
# %%
