from transformers import AutoModelWithLMHead, AutoTokenizer
import gradio as gr
import torch

# Load your model from hub
username = "kpavan2004"      # change it to your HuggingFace username
my_repo = "medicalQnA-gpt2"

my_checkpoint = username + '/' + my_repo       # eg. "yrajm1997/gita-text-generation-gpt2"
my_checkpoint

loaded_model = AutoModelWithLMHead.from_pretrained(my_checkpoint)
# Load your tokenizer from hub
loaded_tokenizer = AutoTokenizer.from_pretrained(my_checkpoint)

# Function for response generation

def generate_query_response(prompt, max_length=200):

    model = loaded_model
    tokenizer = loaded_tokenizer

    # YOUR CODE HERE ...
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if max_length is None:
        max_length = len(input_ids[0]) + 1

    # Check the device of the model
    device = next(model.parameters()).device

    # Move input_ids to the same device as the model
    input_ids = input_ids.to(device)

    # Create the attention mask and pad token id
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.eos_token_id

    output = model.generate(
        input_ids,
        max_length=int(max_length),
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Gradio elements

iface = gr.Interface(fn=generate_query_response,
                    inputs = [gr.Textbox(label="Enter your your prompt"),
                               gr.Textbox(label="Enter max output length")],
                    outputs="textbox",
                    title = "Medical QnA Bot using GPT2",
                    description = "Medical QnA Bot using GPT2, trained using MedQuAD dataset",
                    allow_flagging = 'never')

iface.launch(share = True,debug=True)