from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

model = LlamaForCausalLM.from_pretrained("./BanglaTDM")
tokenizer = PreTrainedTokenizerFast.from_pretrained("./BanglaTDM")

prompt = "কুকুর"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
