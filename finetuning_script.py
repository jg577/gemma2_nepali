import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, disable_progress_bar
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader, IterableDataset
from collections import Counter
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import re
import os
import sys
import unicodedata
import argparse
sys.path.append(("fast-vocabulary-transfer"))
from fvt.fvt import FastVocabularyTransfer

def setup_environment():
    """Setup environment variables and paths"""

    os.environ.update({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "USE_TORCH": "1",
        "HF_HUB_DOWNLOAD_TIMEOUT": "500",
        "HF_HUB_DOWNLOAD_CHUNK_SIZE": "10485760",
        "CC": "gcc"
    })
    disable_progress_bar()

def load_base_model(model_name, max_seq_length=512, dtype=None, load_in_4bit=True):
    """Load the base model and tokenizer"""
    return FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

def preprocess_nepali(values_dict):
    """Preprocess Nepali text with specific rules"""
    text = values_dict['Article']
    if not text or not isinstance(text, str):
        return ""
        
    # Add spaces before common Nepali suffixes
    text = re.sub(r'(ले|को|मा|बाट|देखि|सम्म)$', r' \1', text)
    
    # Remove non-Devanagari characters
    text = re.sub(r'[^\u0900-\u097F\s।0-9]', '', text)
    
    # Add spaces after word separators
    text = re.sub(r'([।])(\S)', r'\1 \2', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class NepaliIterableDataset(IterableDataset):
    """Custom IterableDataset for Nepali text data"""
    def __init__(self, dataset, tokenizer, max_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_examples = 1000000

    def __iter__(self):
        for item in self.dataset:
            text = preprocess_nepali(item)
            if text:
                encodings = self.tokenizer(
                    text, 
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                yield {
                    "input_ids": encodings["input_ids"][0],
                    "attention_mask": encodings["attention_mask"][0],
                    "labels": encodings["input_ids"][0].clone()
                }

    def __len__(self):
        return self.num_examples

def custom_prepare_model_for_kbit_training(model):
    """Prepare model for k-bit training"""
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_disable()
    
    model.config.use_cache = False
    
    for name, param in model.named_parameters():
        if param.ndim == 1:
            param.data = param.data.to(torch.float16)
    
    return model

def setup_training(model, new_tokenizer, train_dataset):
    """Setup training configuration and trainer"""
    training_args = TrainingArguments(
        output_dir='./nepali_lora_weights_ver2',
        num_train_epochs=3,
        warmup_steps=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        save_steps=500,
        logging_steps=5,
        learning_rate=2e-4,
        fp16=True,
        optim="paged_adamw_32bit",
        save_total_limit=2,
        no_cuda=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=new_tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

def main():
    # Setup
    setup_environment()
    
    # Load base model
    model, tokenizer = load_base_model('unsloth/gemma-2-9b-bnb-4bit')
    
    # Load dataset
    iriis_train = load_dataset(
        'IRIISNEPAL/Nepali-Text-Corpus',
        split='train',
        revision='main',
        streaming=True,
        download_mode='force_redownload',
    )

    # Prepare tokenizer
    iriis_tokenizer_dataset = iriis_train.take(100000)
    mapped_tokenizer_iterator = map(preprocess_nepali, iriis_tokenizer_dataset)
    new_tokenizer = tokenizer.train_new_from_iterator(
        mapped_tokenizer_iterator,
        vocab_size=6400,
        show_progress=True,
    )

    # Transfer vocabulary
    fvt = FastVocabularyTransfer()
    new_model = fvt.transfer(
        in_tokenizer=new_tokenizer,
        gen_tokenizer=tokenizer,
        gen_model=model
    )

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Using {n_gpus} GPUs")

    # Prepare model
    model = new_model.to(device)
    model = custom_prepare_model_for_kbit_training(model)

    # Setup LoRA
    lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=['q_proj','v_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create dataset and trainer
    train_dataset = NepaliIterableDataset(iriis_train, new_tokenizer)
    trainer = setup_training(model, new_tokenizer, train_dataset)

    # Train and save
    trainer.train()
    model.save_pretrained("./nepali_lora_weights")

if __name__ == "__main__":
    main()