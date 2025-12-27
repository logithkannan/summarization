import torch
import gradio as gr
from transformers import pipeline

# Create summarization pipeline (GPU + FP16 safe)
text_summary = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16 if torch.cuda.is_available() else None
)

def summary(input_text):
    if not input_text.strip():
        return "Please enter some text to summarize."

    output = text_summary(
        input_text,
        max_length=60,
        min_length=20,
        do_sample=False
    )
    return output[0]["summary_text"]

gr.close_all()

demo = gr.Interface(
    fn=summary,
    inputs=gr.Textbox(label="Input text to summarize", lines=6, placeholder="Paste your text here..."),
    outputs=gr.Textbox(label="Summarized text", lines=4),
    title="@LOGITHKANNAN Project 1: Text Summarizer",
    description="This application summarizes input text using a Transformer-based AI model."
)

demo.launch()
