import torch 
import gradio as gr
from torch import nn
from pathlib import Path

LABELS = Path('class_names.txt').read_text().splitlines()



model = nn.Sequential(
    nn.Conv2d(1, 64, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(128, 256, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(2304, 512),
    nn.ReLU(),
    nn.Linear(512, len(LABELS)),
)
state_dict = torch.load('pytorch_model.bin',    map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

def predict(img):
    x = torch.tensor(img).unsqueeze(0).unsqueeze(0) / 255.

    with torch.no_grad():
        out = model(x)
    
    probabilities = torch.nn.functional.softmax(out[0], dim = 0)
    values, indices = torch.topk(probabilities, k = 5)
   
    return {LABELS[i]: v.item() for i, v in zip(indices, values)}


interface  = gr.Interface(fn=predict,
             inputs="sketchpad",
             outputs="label",
             live=True)

interface.launch()