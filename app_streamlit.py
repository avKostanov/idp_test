import albumentations as A
import numpy as np
import pandas as pd
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from net import getModel

np.set_printoptions(suppress=True)

mapper = {
    '0': 'Shih-Tzu',
    '1': 'Rhodesian ridgeback',
    '2': 'Beagle',
    '3': 'English foxhound',
    '4': 'Australian terrier',
    '5': 'Border terrier',
    '6': 'Golden retriever',
    '7': 'Old English sheepdog',
    '8': 'Samoyed',
    '9': 'Dingo'
}

transform = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def inference(image, model):
    model.eval()
    x = np.array(image.convert('RGB'))
    x_processed = transform(image=x)['image']
    preds = torch.softmax(model(x_processed.unsqueeze(0)), dim=1)
    return preds.detach().numpy()[0]


def top_k_confidence(preds, k=5):
    result = pd.DataFrame(
        data={'confidence': preds.tolist(), 'breed': list(mapper.values())}
    ).nlargest(k, 'confidence').reset_index(drop=True)
    result.index += 1
    return result


model = getModel()

st.title('Dog Breed Classifier')
st.markdown(
    """
        This model can classify several dog breeds:
        * Shih-Tzu
        * Rhodesian ridgeback
        * Beagle
        * English foxhound
        * Australian terrier
        * Border terrier
        * Golden retriever
        * Old English sheepdog
        * Samoyed
        * Dingo

        Model based on finetuned Efficient-Net-B3
    """
)
uploaded_file = st.file_uploader("Choose a file:")

if uploaded_file is not None:
    st.image(uploaded_file, width=500)
    image = Image.open(uploaded_file)
    label = inference(image, model)
    st.table(top_k_confidence(label))
