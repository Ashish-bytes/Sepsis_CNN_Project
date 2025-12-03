import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

# Import your model class
from model_class import CNN   # ❗ change CNN to your actual class name if different

# Load model weights (model file will be uploaded to Hugging Face)
MODEL_PATH = "best_sepsis_cnn.pth"

def load_model():
    model = CNN()    # ❗ change CNN if needed
    state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

    # Handle both normal and checkpoint formats
    try:
        model.load_state_dict(state)
    except:
        model.load_state_dict(state["state_dict"])

    model.eval()
    return model

model = load_model()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

labels = ["No Sepsis", "Sepsis"]  # edit if needed

# Prediction function
def predict(image):
    image = image.convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]

    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload medical image"),
    outputs=gr.Label(num_top_classes=2),
    title="Sepsis Detection using CNN",
    description="Upload a medical image to detect Sepsis."
)

demo.launch()

