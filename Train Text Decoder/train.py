import os
import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils._pytree")
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
)
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    GenerationConfig,
)
from transformers.trainer_utils import get_last_checkpoint


class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": [1] * len(self.input_ids[idx]),
        }


if __name__ == "__main__":
    print("Reading captions.txt...")
    captions = []
    with open("captions.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                captions.append(line.split(maxsplit=1)[-1])
    print(f"Loaded {len(captions)} captions")

    # If tokenizer already exists in output_dir, we could load it,
    # but rebuilding BPE on captions is very fast so we keep it inline.
    print("Training tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    bpe_trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        show_progress=True,
        min_frequency=2,
    )
    tokenizer.train_from_iterator(captions, bpe_trainer)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )

    print("Tokenizing dataset...")
    encodings = hf_tokenizer(
        captions,
        truncation=True,
        max_length=64,
        padding=False,
    )
    tokenized_dataset = CaptionDataset(encodings["input_ids"])

    config = LlamaConfig(
        vocab_size=hf_tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        max_position_embeddings=128,
        pad_token_id=hf_tokenizer.pad_token_id,
        bos_token_id=hf_tokenizer.bos_token_id,
        eos_token_id=hf_tokenizer.eos_token_id,
    )

    output_dir = "./BanglaTDM"

    # ---------------------------------------------------------
    # RESUME CHECKPOINT DETECTION
    # ---------------------------------------------------------
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            print(f"Found existing checkpoint at {last_checkpoint}. Will resume training.")
        else:
            print("No valid checkpoint found. Starting fresh.")

    # Initialize model (if resuming, Trainer will automatically load weights later)
    model = LlamaForCausalLM(config)
    print(f"Model parameters: {model.num_parameters():,}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=hf_tokenizer,
        mlm=False,
    )

    batch_size = 384

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,

        # SAVING STRATEGY
        save_strategy="epoch",
        save_total_limit=2,  # Keep 2 to prevent corruption during an unexpected crash

        logging_steps=50,
        logging_first_step=True,
        prediction_loss_only=True,

        # Hardware Setup
        dataloader_num_workers=1,
        bf16=True,
        torch_compile=True,
        report_to="none",
    )

    # ---------------------------------------------------------
    # CYCLIC LR SCHEDULER & OPTIMIZER SETUP
    # ---------------------------------------------------------
    steps_per_epoch = len(tokenized_dataset) // batch_size
    if len(tokenized_dataset) % batch_size != 0:
        steps_per_epoch += 1

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0, fused=True)

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=5e-4,
        step_size_up=steps_per_epoch // 2,
        mode="triangular",
        cycle_momentum=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler)
    )

    print("Starting training...")
    # Pass the detected checkpoint here. If it's None, it starts from scratch.
    trainer.train(resume_from_checkpoint=last_checkpoint)

    model.save_pretrained(output_dir, safe_serialization=True)
    hf_tokenizer.save_pretrained(output_dir)

    gen_config = GenerationConfig(
        bos_token_id=hf_tokenizer.bos_token_id,
        eos_token_id=hf_tokenizer.eos_token_id,
        pad_token_id=hf_tokenizer.pad_token_id,
        max_length=64,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    gen_config.save_pretrained(output_dir)

    print(f"\nModel saved to {output_dir}/")