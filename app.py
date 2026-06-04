import gradio as gr

def caption_interface(image, model_choice, beam_width):
    if image is None:
        return "দয়া করে একটি ছবি আপলোড করুন।"
    return f"[{model_choice}] একটি মনোরম প্রাকৃতিক দৃশ্য যেখানে নদী ও সবুজ প্রকৃতি দেখা যাচ্ছে। (Beam={beam_width})"

with gr.Blocks(title="BanglaVision Research Platform") as demo:
    gr.Markdown("# 🇧🇩 BanglaVision: Bangla Image Captioning System")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Input Image / ছবি আপলোড")
            model_selector = gr.Dropdown(
                choices=["SigLIP2 + SCST (Best)", "ViT-Base + GRU", "ResNet-50 Baseline"],
                value="SigLIP2 + SCST (Best)",
                label="Model Backbone"
            )
            beam_slider = gr.Slider(1, 10, value=5, step=1, label="Beam Width")
            btn = gr.Button("Generate Caption / ক্যাপশন তৈরি করুন", variant="primary")
        with gr.Column():
            txt_output = gr.Textbox(label="Generated Bengali Caption", lines=4)

    btn.click(caption_interface, inputs=[img_input, model_selector, beam_slider], outputs=txt_output)

if __name__ == '__main__':
    demo.launch()
