import argparse
import json
import os
from pathlib import Path

import torch
import jiwer
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
    StoppingCriteria,
    StoppingCriteriaList,
)
from huggingface_hub import whoami

ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100


class FinetuneDataset(Dataset):
    def __init__(
        self,
        processor,
        dataset_name,
        split,
        training,
        text_column="text",
        audio_column="audio",
        max_samples=None,
        rank=0,
        world_size=1,
    ):
        self.data = load_dataset(dataset_name, split=split)
        if max_samples is not None:
            self.data = self.data.select(range(max_samples))
        if world_size > 1:
            self.data = self.data.shard(num_shards=world_size, index=rank)
        self.processor = processor
        self.instruction = "Transcribe the audio clip into text."
        self.training = training
        self.text_column = text_column
        self.audio_column = audio_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Each example in the dataset is expected to have:
          - '{audio_column}': a dict with keys "array" and "sampling_rate"
          - '{text_column}': the transcription string.
        """
        data = self.data[idx]
        user_message = {
            "role": "user",
            "content": "<|audio_1|> " + self.instruction,
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            audios=[
                (
                    data[self.audio_column]["array"],
                    data[self.audio_column]["sampling_rate"],
                )
            ],
            return_tensors="pt",
        )
        answer = f"{data[self.text_column]}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors="pt").input_ids
        if self.training:
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1] :] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
        }


# Utility functions for batching
def pad_sequence(sequences, padding_side="right", padding_value=0):
    """
    Pad a list of tensors to the same length.
    sequences: list of tensors with shape [seq_len, ...]
    """
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(seq.size(0) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output[i, :length] = seq
        else:
            output[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """
    Concatenate tensors along a specified dimension, padding to match dimensions.
    """
    ndim = tensors[0].dim()
    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)
    index = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(index, index + t.shape[dim])
        output[tuple(slices)] = t
        index += t.shape[dim]
    return output


def collate_fn(batch):
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    for sample in batch:
        input_ids_list.append(sample["input_ids"][0])
        labels_list.append(sample["labels"][0])
        input_audio_embeds_list.append(sample["input_audio_embeds"])
        audio_embed_sizes_list.append(sample["audio_embed_sizes"])
        audio_attention_mask_list.append(
            sample["input_audio_embeds"].new_full(
                (sample["input_audio_embeds"].size(1),), True, dtype=torch.bool
            )
        )

    input_ids = pad_sequence(input_ids_list, padding_side="left", padding_value=0)
    labels = pad_sequence(labels_list, padding_side="left", padding_value=0)
    audio_attention_mask = (
        pad_sequence(
            audio_attention_mask_list, padding_side="right", padding_value=False
        )
        if len(audio_attention_mask_list) > 1
        else None
    )
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    return BatchFeature(
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_audio_embeds": input_audio_embeds,
            "audio_embed_sizes": audio_embed_sizes,
            "audio_attention_mask": audio_attention_mask,
            "input_mode": 2,  # speech mode
        }
    )


# Stopping criteria for generation that handles multiple stop tokens
class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(
            batch_size, dtype=torch.long, device=stop_tokens.device
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        generated_inputs = torch.eq(
            input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens
        )
        equal_generated_inputs = torch.all(generated_inputs, dim=2)
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]
        return torch.all(self.stop_tokens_idx)


def create_model(model_name_or_path, use_flash_attention=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        trust_remote_code=True,
    ).to("cuda")
    return model


@torch.no_grad()
def evaluate(
    model,
    processor,
    eval_dataset,
    save_path=None,
    disable_tqdm=False,
    eval_batch_size=1,
    metric="wer",
):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model.eval()
    all_generated_texts = []
    all_labels = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )
    stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(
        stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt"
    )["input_ids"]
    stop_tokens_ids = stop_tokens_ids.to(f"cuda:{local_rank}")

    for inputs in tqdm(
        eval_dataloader, disable=(rank != 0) or disable_tqdm, desc="running eval"
    ):
        stopping_criteria = StoppingCriteriaList(
            [
                MultipleTokenBatchStoppingCriteria(
                    stop_tokens_ids, batch_size=inputs.input_ids.size(0)
                )
            ]
        )
        inputs = inputs.to(f"cuda:{local_rank}")
        generated_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=64,
            stopping_criteria=stopping_criteria,
        )
        stop_tokens_idx = stopping_criteria[0].stop_tokens_idx.reshape(
            inputs.input_ids.size(0), -1
        )[:, 0]
        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - stop_tokens_ids.shape[-1],
            generated_ids.shape[-1],
        )
        generated_text = [
            processor.decode(
                _pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx)
        ]
        all_generated_texts.extend(generated_text)
        labels = [
            processor.decode(_label_ids[_label_ids != 0]).removesuffix(ANSWER_SUFFIX)
            for _label_ids in inputs["labels"]
        ]
        all_labels.extend(labels)

    all_generated_texts = gather_object(all_generated_texts)
    all_labels = gather_object(all_labels)

    if rank == 0:
        assert len(all_generated_texts) == len(all_labels)
        # Use either Word Error Rate (WER) or Character Error Rate (CER) based on parameter
        if metric.lower() == "wer":
            error_score = jiwer.wer(" ".join(all_labels), " ".join(all_generated_texts))
            metric_name = "WER"
        else:  # Use CER
            error_score = jiwer.cer(" ".join(all_labels), " ".join(all_generated_texts))
            metric_name = "CER"

        print(f"{metric_name} Score: {error_score:.4f}")
        if save_path:
            with open(save_path, "w") as f:
                save_dict = {
                    "all_generated_texts": all_generated_texts,
                    "all_labels": all_labels,
                    metric_name.lower(): error_score,
                }
                json.dump(save_dict, f, indent=4, ensure_ascii=False)
        return error_score
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-4-multimodal-instruct",
        help="Model name or path to load from",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="JacobLinCool/common_voice_19_0_zh-TW",
        help="Dataset name to use for training and evaluation",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Dataset split to use for training",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        help="Dataset split to use for evaluation",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        help="Maximum number of training samples",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        help="Maximum number of evaluation samples",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention for more efficient training on compatible hardware",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/",
        help="Output directory for saving model checkpoints and logs",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=128,
        help="Total batch size across all GPUs (global_batch_size = batch_size_per_gpu * num_gpus * gradient_accumulation_steps)",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=4,
        help="Training batch size per GPU (decrease this value if you encounter OOM errors)",
    )
    parser.add_argument(
        "--eval_batch_size_per_gpu",
        type=int,
        default=1,
        help="Evaluation batch size per GPU (can typically be larger than training batch size since no gradients are stored)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of complete passes through the training dataset",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4.0e-5,
        help="Peak learning rate for optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient for regularization",
    )
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="wer",
        choices=["wer", "cer"],
        help="Evaluation metric: 'wer' for Word Error Rate (word-level) or 'cer' for Character Error Rate (character-level)",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the column containing the transcription text",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="audio",
        help="Name of the column containing the audio data",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the Hugging Face Hub after training",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="Repository name to push to on the Hugging Face Hub (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--no-tqdm", dest="tqdm", action="store_false", help="Disable tqdm"
    )
    parser.add_argument(
        "--skip_initial_eval",
        action="store_true",
        help="Skip evaluation before training",
    )
    args = parser.parse_args()

    if args.push_to_hub:
        try:
            whoami()
        except Exception as e:
            print(
                "You need to be logged in to the Hugging Face Hub to push the model. "
                "Please run `huggingface-cli login`."
            )
            raise e

    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )

    model.set_lora_adapter("speech")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    eval_dataset = FinetuneDataset(
        processor,
        dataset_name=args.dataset_name,
        split=args.eval_split,
        training=False,
        text_column=args.text_column,
        audio_column=args.audio_column,
        max_samples=args.max_eval_samples,
        rank=rank,
        world_size=world_size,
    )
    train_dataset = FinetuneDataset(
        processor,
        dataset_name=args.dataset_name,
        split=args.train_split,
        training=True,
        text_column=args.text_column,
        audio_column=args.audio_column,
        max_samples=args.max_train_samples,
        rank=rank,
        world_size=world_size,
    )

    num_gpus = accelerator.num_processes
    print(f"training on {num_gpus} GPUs")
    assert (
        args.batch_size % (num_gpus * args.batch_size_per_gpu) == 0
    ), "Batch size must be divisible by the number of GPUs"
    gradient_accumulation_steps = args.batch_size // (
        num_gpus * args.batch_size_per_gpu
    )

    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        warmup_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy="epoch",
        save_total_limit=3,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        deepspeed=None,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.hub_model_id else None,
    )

    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Make initial evaluation optional based on the new flag
    if not args.skip_initial_eval:
        score = evaluate(
            model,
            processor,
            eval_dataset,
            save_path=out_path / "eval_before.json",
            disable_tqdm=not args.tqdm,
            eval_batch_size=args.eval_batch_size_per_gpu,
            metric=args.eval_metric,
        )
        if accelerator.is_main_process:
            metric_name = args.eval_metric.upper()
            print(f"{metric_name} Score before finetuning: {score:.4f}")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
    )

    trainer.train()

    trainer.save_model()
    if accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)

    accelerator.wait_for_everyone()

    # Evaluate after fine-tuning
    del model
    del trainer
    __import__("gc").collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation=(
            "flash_attention_2" if args.use_flash_attention else "sdpa"
        ),
    ).to("cuda")

    score = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / "eval_after.json",
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.eval_batch_size_per_gpu,
        metric=args.eval_metric,
    )
    if accelerator.is_main_process:
        metric_name = args.eval_metric.upper()
        print(f"{metric_name} Score after finetuning: {score:.4f}")


if __name__ == "__main__":
    main()
