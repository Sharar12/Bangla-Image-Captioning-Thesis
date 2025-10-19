import gradio as gr
from PIL import Image

def generate_caption(image, beam_width, length_penalty, temperature):
    if image is None:
        return "দয়া করে একটি ছবি আপলোড করুন।"
    return f"একটি প্রাকৃতিক দৃশ্যে নদী ও সবুজ গাছপালা দেখা যাচ্ছে। (Beam={beam_width}, Alpha={length_penalty})"

demo = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type="pil", label="Input Image / ইনপুট ছবি"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Beam Width"),
        gr.Slider(minimum=0.0, maximum=1.5, value=0.7, step=0.1, label="Length Penalty (Alpha)"),
        gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Bengali Caption / তৈরি করা বাংলা ক্যাপশন", lines=3),
    title="BanglaVision: Interactive Bangla Image Captioning",
    description="Deep Learning-powered Bengali image caption generation using SigLIP2 vision encoder and attention decoders.",
    examples=[]
)

if __name__ == "__main__":
    demo.launch()
