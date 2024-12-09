import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# Load the label encoder
label_encoder = joblib.load('./fine_tuned_model/label_encoder.joblib')

# Function for making predictions
def predict(text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    predicted_class = probs.argmax().item()
    return label_encoder.inverse_transform([predicted_class])[0]

# if __name__ == "__main__":
#     new_text = "build(deps): bump github.com/urfave/cli/v2 from 2.11.0 to 2.11.1"
#     predicted_label = predict(new_text)
#     print(f"Predicted label: {predicted_label}")

