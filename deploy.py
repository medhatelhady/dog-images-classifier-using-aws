import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
feature_extract = True
num_classes = 133
input_size = 224
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def Net():


    model_ft = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    return model_ft

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # calling gpu



def model_fn(model_dir):

    model = Net().to(device)
    
    with open(os.path.join(model_dir, "model.pt"), "rb") as f:
        print("Loading the classifier")
        checkpoint = torch.load(f , map_location =device)
        print("loading checkpoint")
        model.load_state_dict(checkpoint)
        logger.info('model loaded successfully')
    model.eval()
    return model




def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.debug(f'Request body CONTENT-TYPE: {content_type}')
    logger.debug(f'Request body TYPE: {type(request_body)}')
    if content_type == JPEG_CONTENT_TYPE: return Image.open(io.BytesIO(request_body))
    logger.debug('SO loded JPEG content')    
    if content_type == JSON_CONTENT_TYPE:
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))
def predict_fn(input_object, model):
 
    logger.info('prediction process')
    test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    input_object=test_transform(input_object)
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction