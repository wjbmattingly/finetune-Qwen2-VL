import torch
import os
import base64
from io import BytesIO

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from PIL import Image
from functools import partial
from tqdm import tqdm
import datetime

from accelerate import Accelerator

def find_assistant_content_sublist_indexes(l):
    start_indexes = []
    end_indexes = []

    for i in range(len(l) - 1):
        if l[i] == 151644 and l[i + 1] == 77091:
            start_indexes.append(i)
            for j in range(i + 2, len(l)):
                if l[j] == 151645:
                    end_indexes.append(j)
                    break

    return list(zip(start_indexes, end_indexes))

class HuggingFaceDataset(Dataset):
    def __init__(self, dataset, image_column, text_column, user_text="Convert this image to text"):
        self.dataset = dataset
        self.image_column = image_column
        self.text_column = text_column
        self.user_text = user_text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column]
        assistant_text = item[self.text_column]

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.user_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_text}
                    ]
                }
            ]
        }

def ensure_pil_image(image, min_size=256):
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, str):
        if image.startswith('data:image'):
            image = image.split(',')[1]
        image_data = base64.b64decode(image)
        pil_image = Image.open(BytesIO(image_data))
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    if pil_image.width < min_size or pil_image.height < min_size:
        scale = max(min_size / pil_image.width, min_size / pil_image.height)
        new_width = int(pil_image.width * scale)
        new_height = int(pil_image.height * scale)
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    return pil_image

def collate_fn(batch, processor, device):
    messages = [item['messages'] for item in batch]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    
    images = [ensure_pil_image(msg[0]['content'][0]['image']) for msg in messages]
    
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)

    return inputs, labels_ids

def validate(model, val_loader):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            inputs, labels = batch
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    model.train()
    return avg_val_loss

def train_and_validate(model_name, output_dir, dataset_name, image_column, text_column, device="cuda", user_text="Convert this image to text", num_accumulation_steps=2, eval_steps=10000, max_steps=100000, train_select_start=0, train_select_end=1000, val_select_start=0, val_select_end=1000, train_batch_size=1, val_batch_size=1, train_field="train", val_field="validation"):
    accelerator = Accelerator(gradient_accumulation_steps=num_accumulation_steps)
    device = accelerator.device
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map=device
    )

    processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=512*28*28, padding_side="right")

    dataset = load_dataset(dataset_name)
    train_dataset = dataset[train_field].shuffle(seed=42).select(range(train_select_start, train_select_end))
    val_dataset = dataset[val_field].shuffle(seed=42).select(range(val_select_start, val_select_end))

    train_dataset = HuggingFaceDataset(train_dataset, image_column, text_column, user_text)
    val_dataset = HuggingFaceDataset(val_dataset, image_column, text_column, user_text)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device)
    )

    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    global_step = 0
    progress_bar = tqdm(total=max_steps, desc="Training")

    while global_step < max_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                global_step += 1
                inputs, labels = batch
                outputs = model(**inputs, labels=labels)
                
                loss = outputs.loss
                accelerator.backward(loss)
                
                if global_step % num_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

            if global_step % eval_steps == 0 or global_step == max_steps:
                avg_val_loss = validate(model, val_loader)
                accelerator.print(f"Step {global_step}, Validation Loss: {avg_val_loss}")

                if accelerator.is_local_main_process:
                    save_dir = os.path.join(output_dir, f"model_step_{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(save_dir)
                    processor.save_pretrained(save_dir)
                    accelerator.print(f"Model and processor saved at step {global_step}")

                model.train()

            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

    progress_bar.close()

    if accelerator.is_local_main_process:
        save_dir = os.path.join(output_dir, "final")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)


        accelerator.print("Training completed.")

        # Add command-line argument parsing
        import argparse

        def main():
            parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL model")
            parser.add_argument("--model_name", type=str, required=True, help="Name of the model to fine-tune")
            parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
            parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
            parser.add_argument("--image_column", type=str, required=True, help="Name of the image column in the dataset")
            parser.add_argument("--text_column", type=str, required=True, help="Name of the text column in the dataset")
            parser.add_argument("--user_text", type=str, default="Convert this image to text", help="User text prompt")
            parser.add_argument("--num_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps")
            parser.add_argument("--eval_steps", type=int, default=100, help="Number of steps between evaluations")
            parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of training steps")
            parser.add_argument("--train_select_start", type=int, default=0, help="Start index for training data selection")
            parser.add_argument("--train_select_end", type=int, default=1000, help="End index for training data selection")
            parser.add_argument("--val_select_start", type=int, default=0, help="Start index for validation data selection")
            parser.add_argument("--val_select_end", type=int, default=1000, help="End index for validation data selection")
            parser.add_argument("--train_batch_size", type=int, default=3, help="Training batch size")
            parser.add_argument("--val_batch_size", type=int, default=3, help="Validation batch size")
            parser.add_argument("--train_field", type=str, default="train", help="Field name for training data")
            parser.add_argument("--val_field", type=str, default="validation", help="Field name for validation data")

            args = parser.parse_args()

            # Call the training function with parsed arguments
            train_and_validate(
                model_name=args.model_name,
                output_dir=args.output_dir,
                dataset_name=args.dataset_name,
                image_column=args.image_column,
                text_column=args.text_column,
                user_text=args.user_text,
                num_accumulation_steps=args.num_accumulation_steps,
                eval_steps=args.eval_steps,
                max_steps=args.max_steps,
                train_select_start=args.train_select_start,
                train_select_end=args.train_select_end,
                val_select_start=args.val_select_start,
                val_select_end=args.val_select_end,
                train_batch_size=args.train_batch_size,
                val_batch_size=args.val_batch_size,
                train_field=args.train_field,
                val_field=args.val_field
            )

        if __name__ == "__main__":
            main()
