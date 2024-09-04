import streamlit as st
import torch
from model import MyGPTModel
from config import GPT_774_CONFIG
from tiktoken import get_encoding
from utils import format_input, generate_text_v2
from fine_tune_classifier import classify_review

@st.cache_resource
def load_model(app_mode, device):
    model = MyGPTModel(GPT_774_CONFIG)
    model.to(device)
    if app_mode == "Instruction-Tuned Model":
        checkpoint = torch.load("model_and_optimizer_3.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
    elif app_mode == "Spam Detector":
        num_classes = 2                                                                                             
        model.output_head = torch.nn.Linear(in_features=GPT_774_CONFIG["embedding_dim"], out_features=num_classes)
        checkpoint = torch.load("classifier.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def generate_response(input_text, model, temperature, top_k, device):
    input_text = format_input({
        "instruction": input_text,
        "input": "",
    })
    input_ids = tokenizer.encode(input_text, allowed_special={"<|endoftext|>"})
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    output = generate_text_v2(model, input_ids, max_generated_tokens=256, context_size=GPT_774_CONFIG["context_size"], eos_token_id=50256, temperature=temperature, top_k=top_k)
    decoded_output = tokenizer.decode(output.squeeze(0).tolist())
    response_start = len(input_text)
    response = decoded_output[response_start:].replace("### Response:", "").strip()
    return response

tokenizer = get_encoding("gpt2")
device = ("cuda" if torch.cuda.is_available() else "cpu")

app_mode = st.sidebar.radio("Choose the application", ["Instruction-Tuned Model", "Spam Detector"])

if app_mode == "Instruction-Tuned Model":
    model = load_model(app_mode, device)


    st.title("MyGPT Interactive Interface")
    st.write("Enter text and generate a response using the model.")

    user_input = st.text_area("Input text:", "")

    temperature = st.slider("Temperature", 0.1, 2.0, 1.0)
    top_k = st.slider("Top-k Sampling", 1, 100, 50)

    if st.button("Generate Response"):
        response = generate_response(user_input, model, temperature, top_k, device)
        st.write("### Response")
        st.write(response)
elif app_mode == "Spam Detector":
    model = load_model(app_mode, device)
    st.title("Spam Detector")
    st.write("Enter text to check if it is spam.")

    user_input = st.text_area("Input text:", "")

    if st.button("Classify"):
        classification = classify_review(user_input, model, tokenizer, device, max_length=1024)
        st.write("### Classification")
        st.write(f"The input text is probably **{classification}**")
