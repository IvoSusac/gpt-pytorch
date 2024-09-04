# Generative Pre-trained Transformer in Pytorch
This project implements a Generative Pre-trained Transformer (GPT) model (equivalent to OpenAI's GPT-2) from scratch using PyTorch. The goal is to create a foundational understanding of the GPT architecture by building and training a model that can generate text based on input prompts. This project is intended for educational purposes and to serve as a starting point as I continue my research in the field of NLP and GenAI.

## The GPT Architecture
The architecture of the model is based on the [original GPT paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf).
It mostly consists of stacked Transformer decoder layers, each composed of:

- Multi-Head Self-Attention Mechanism: This allows the model to focus on different parts of the input sequence when making predictions, enabling it to capture contextual dependencies.
- Layer Normalization: Applied to stabilize and accelerate training by normalizing inputs across the batch dimension.
- Feedforward (Linear) Layer: A fully connected network that processes the output of the self-attention layer and introduces non-linearity.
- Residual Connections: These connections help prevent the vanishing gradient problem by allowing gradients to bypass model layers.

The following diagram provides a conceptual overview of the mentioned architecture:

<img src="https://github.com/user-attachments/assets/cd45144b-9a1b-4a33-ad0a-7a3f76dc6b03" alt="Conceptual architecture of a GPT model"/> <br>
*Figure 1: Conceptual architecture of a GPT model*  
*Cited from*: A Mathematical Investigation of Hallucination and Creativity in GPT Models - Scientific Figure on ResearchGate.  
*Available from*: [ResearchGate](https://www.researchgate.net/figure/Conceptual-architecture-of-a-GPT-model_fig1_370853178)

The implementation of each of these layers, along with the entire GPT model can be found in the **`modules`** folder.

## Pre-Training
By pre-training the model on large corpora, the model learns general language representations by predicting the next word in a sentence, given all previous words. The model is trained on raw text, in an unsupervised manner. Even though pre-training is implemented in the `train.py` script, it requires **a lot** of computational resources. Luckily, pre-trained model weights for all of the GPT-2 models are open-sourced, and can be downloaded and loaded into the implemented model by leveraging the functions in the `get_gpt_weights.py` script.

## Fine-Tuning
After pretraining, the model is fine-tuned on a smaller, task-specific dataset. Fine-tuning allows the model to adapt its general language understanding to specific tasks, such as text classification, summarization, or question answering. 

In this project, I fine-tuned the model on two datasets:

- [Alpaca-Stanford Dataset](https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-release): Used for instructional fine-tuning, where the model learns to generate responses to instructional prompts. I used 10,000 of the ~50,000 entries for computational reasons.
- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset): Used for spam classification, where the model is fine-tuned to distinguish between spam and legitimate text messages.

# Fine-Tuning Results (5 epochs on 2 NVIDIA GeForce RTX 3090 GPUs)
Instructional Fine-Tuning (Alpaca-Stanford):
- Training Loss: 0.271
- Validation Loss: 0.267

# Spam Classification (SMSSpam):
- Training Loss: 0.084
- Validation Loss: 0.072
- Test Set Accuracy: 96.5%

## Dependencies
The dependencies for this project can be installed with `pip`, by running:
```
pip install -r requirements.txt
```

## Usage
**Note: A more user-friendly framework for this project is in development.**

Right now, to fine-tune the model on your specific dataset, you must modify the `fine_tune_instructional.py` or `fine_tune_classifier.py` script to load your own dataset and model and run:
```
python fine_tune_instructional.py
```
or
```
python fine_tune_classifier.py
```
depending on the task you want to fine-tune for.

Evaluation of the instruction-finetuned model can be done using `GPT-4o-mini`. 
The `evaluate_responses.py` script generates model responses for 200 (for computational reasons, can be increased depending on your setup) test dataset questions.
The `gpt_eval.py` script reads the responses and computes the average scores (which range from 0 to 100).

By running
```
streamlit run app.py
```
you can play around with the models using Streamlit UI.

## Acknowledgments
Special thanks to my mentor, Dr. Domagoj Matijević, for his guidance and support throughout the development of this project. <br>
This code is influenced by [S. Raschka, Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch), a book that greatly helped me in building this project.

## Contributing
Contributions to this project are welcome! Please feel free to submit a pull request or open an issue to discuss potential improvements or new features.



