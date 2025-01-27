import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import DatasetDict
from transformers import BitsAndBytesConfig


class QLoRAFineTuner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", self.device)
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self.dataset_name = "euclaise/logician"
        self.output_dir = "./model_outputs/qlora_mistral_finetuned"

    def load_tokenizer_and_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS token as padding token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False
            ),
            device_map="cuda:0"
        )
        self.model.gradient_checkpointing_enable()

    def load_dataset(self):
        self.dataset = load_dataset(self.dataset_name)

    def tokenize_dataset(self):
        # Retrieve the tokenizer's maximum model length with a fallback
        max_length = min(self.tokenizer.model_max_length, 1024)  # Use 1024 as an upper bound for safety
        print(f"Tokenizer max length constrained to: {max_length}")

        # Filter the dataset to exclude entries that are too long
        def filter_function(example):
            input_text = f"Instruction: {example['instruction']} [SEP] Response: {example['response']}"
            tokenized_length = len(
                self.tokenizer(
                    input_text,
                    truncation=False,
                    add_special_tokens=True,
                )["input_ids"]
            )
            return tokenized_length <= max_length

        # Log dataset size before and after filtering
        print(f"Original dataset size: {len(self.dataset['train'])}")
        self.dataset = self.dataset.filter(filter_function)
        print(f"Filtered dataset size: {len(self.dataset['train'])}")

        # Tokenize the dataset
        def tokenize_function(batch):
            # Extract lists of instructions and responses from the batch
            input_texts = [
                f"Instruction: {instruction} [SEP] Response: {response}"
                for instruction, response in zip(batch["instruction"], batch["response"])
            ]

            # Tokenize the input texts with padding and truncation
            tokenized = self.tokenizer(
                input_texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",  # Ensure tensors are returned directly
            )

            # Create labels by copying input_ids
            labels = tokenized["input_ids"].clone()
            for i, (instruction, response) in enumerate(zip(batch["instruction"], batch["response"])):
                instruction_length = len(
                    self.tokenizer(
                        f"Instruction: {instruction} [SEP]",
                        truncation=True,
                        max_length=max_length,
                    )["input_ids"]
                )
                # Mask the instruction portion of the input
                labels[i][:instruction_length] = -100
            tokenized["labels"] = labels
            return tokenized

        # Apply tokenization sequentially for simplicity
        tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,  # Process batches
            batch_size=16,  # Ensure batch size matches processing capacity
            remove_columns=self.dataset["train"].column_names,  # Remove original columns
        )

        # Split the tokenized dataset into train, validation, and test sets
        train_test = tokenized_dataset["train"].train_test_split(test_size=0.2, seed=42)
        self.processed_dataset = DatasetDict({
            "train": train_test["train"],
            "validation": train_test["test"].train_test_split(test_size=0.5, seed=42)["test"],
            "test": train_test["test"].train_test_split(test_size=0.5, seed=42)["train"],
        })

    def configure_lora(self):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05
        )
        self.model = get_peft_model(self.model, lora_config)

        # Ensure trainable parameters have requires_grad=True
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"{name} requires gradients.")
            else:
                print(f"WARNING: {name} does not require gradients!")
        self.model.print_trainable_parameters()

    def prepare_training_args(self):
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=500,
            eval_steps=500,
            logging_steps=100,
            learning_rate=2e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            warmup_steps=500,
            weight_decay=0.01,
            fp16=True,
            push_to_hub=False,
            report_to="none"
        )

    def train_model(self):
        # Use DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Disable MLM as this is causal LM training
        )

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.processed_dataset["train"],
            eval_dataset=self.processed_dataset["validation"],
            data_collator=data_collator,  # Correctly specify the data collator
        )
        trainer.train()
        trainer.save_model(self.output_dir)
        print(f"Model saved to {self.output_dir}")


if __name__ == "__main__":
    finetuner = QLoRAFineTuner()
    finetuner.load_tokenizer_and_model()
    finetuner.load_dataset()
    finetuner.tokenize_dataset()
    finetuner.configure_lora()
    finetuner.prepare_training_args()
    print(finetuner.device)
    finetuner.train_model()
