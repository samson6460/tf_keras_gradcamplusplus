# %%
import requests
import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# %%
def download_file_from_google_drive(id, destination):
    """
    source: https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive).
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination) 

def vgg16_mura_model(path):
    """
    get a vgg16 model can classify bone X-rays into three categories: wrist, shoulder and elbow.

    Parameters:
        path:  string, if there's no model in the path, it will download the weights automatically.

    Return:
        model object.
    """
    model_path = path
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print("downloading the weights of model to",path,"...")
        download_file_from_google_drive('175QH-aIvlLvxrUGyCEpfQAQ5qiVfE_s5', model_path)
        print("done.")
        model = load_model(model_path)

    return model

def preprocess_image(img_path, target_size=(224, 224)):
    """
    preprocess the image by reshape and normalization.

    Parameters:
        img_path:  string.
        target_size: tuple, reshape to this size.
    Return:
        image array.
    """
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img /= 255

    return img

# %%
def show_imgwithheat(img_path, heatmap, alpha=0.4, return_array=False):
    """
    show the image with heatmap.

    Parameters:
        img_path: string.
        heatmap:  image array, get it by calling grad_cam().
        alpha:    float, transparency of heatmap.
        return_array: bool, return a superimposed image array or not.
    Return:
        None or image array.
    """
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),-1)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype('uint8')
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img,0,255).astype('uint8')
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    imgwithheat = Image.fromarray(superimposed_img)  
    display(imgwithheat)

    if return_array:
        return superimposed_img

# %%
