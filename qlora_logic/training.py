import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import DatasetDict
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training


class QLoRAFineTuner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self.dataset_name = "euclaise/logician"
        self.output_dir = "./model_outputs/qlora_mistral_finetuned"

    def load_tokenizer_and_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS token as padding token

        # Apply quantization first
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="cuda:0",
                                                          quantization_config=quantization_config)
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=16,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(model, config)

    def load_dataset(self):
        self.dataset = load_dataset(self.dataset_name)

    def tokenize_dataset(self):
        max_length = min(self.tokenizer.model_max_length, 1024)
        print(f"Tokenizer max length constrained to: {max_length}")

        def filter_function(example):
            input_text = f"Instruction: {example['instruction']} [SEP] Response: {example['response']}"
            tokenized_length = len(
                self.tokenizer(input_text, truncation=False, add_special_tokens=True)["input_ids"]
            )
            return tokenized_length <= max_length

        print(f"Original dataset size: {len(self.dataset['train'])}")
        self.dataset = self.dataset.filter(filter_function)
        print(f"Filtered dataset size: {len(self.dataset['train'])}.")

        def tokenize_function(batch):
            input_texts = [
                f"Instruction: {instruction} [SEP] Response: {response}"
                for instruction, response in zip(batch["instruction"], batch["response"])
            ]
            tokenized = self.tokenizer(
                input_texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            labels = tokenized["input_ids"].clone()
            for i, (instruction, response) in enumerate(zip(batch["instruction"], batch["response"])):
                instruction_length = len(
                    self.tokenizer(f"Instruction: {instruction} [SEP]", truncation=True, max_length=max_length)[
                        "input_ids"]
                )
                labels[i][:instruction_length] = -100
            tokenized["labels"] = labels
            return tokenized

        tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            batch_size=16,
            remove_columns=self.dataset["train"].column_names,
        )

        train_test = tokenized_dataset["train"].train_test_split(test_size=0.2, seed=42)
        self.processed_dataset = DatasetDict({
            "train": train_test["train"],
            "validation": train_test["test"].train_test_split(test_size=0.5, seed=42)["test"],
            "test": train_test["test"].train_test_split(test_size=0.5, seed=42)["train"],
        })


    def prepare_training_args(self):
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="steps",
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
            fp16=False,
            push_to_hub=False,
            report_to="none"
        )

    def train_model(self):
        torch.set_grad_enabled(True)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.processed_dataset["train"],
            eval_dataset=self.processed_dataset["validation"],
            data_collator=data_collator,
        )

        trainer.create_optimizer()

        # Print optimizer groups to debug
        optimizer_grouped_params = trainer.optimizer.param_groups
        for i, group in enumerate(optimizer_grouped_params):
            print(f"Optimizer group {i}: {len(group['params'])} parameters")

        # Ensure at least one parameter is trainable
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_trainable}")

        if num_trainable == 0:
            raise RuntimeError("No parameters require gradients! Check LoRA and quantization settings.")

        trainer.train()
        trainer.save_model(self.output_dir)
        print(f"Model saved to {self.output_dir}")


if __name__ == "__main__":
    finetuner = QLoRAFineTuner()
    finetuner.load_tokenizer_and_model()
    finetuner.load_dataset()
    finetuner.tokenize_dataset()
    finetuner.prepare_training_args()
    finetuner.train_model()

