import os
import json

import torch
import jiwer
from accelerate.utils import gather_object
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    StoppingCriteria,
    StoppingCriteriaList,
)
import librosa

ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100


class MultipleTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: torch.LongTensor) -> None:
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_token_list in self.stop_tokens:
            token_len = stop_token_list.shape[0]
            if token_len <= input_ids.shape[1] and torch.all(
                input_ids[0, -token_len:] == stop_token_list
            ):
                return True
        return False


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


class BaseDataset(Dataset):
    def __init__(
        self,
        processor,
        dataset_name,
        split,
        text_column="text",
        audio_column="audio",
        max_samples=None,
        rank=0,
        world_size=1,
        dataset_subset=None,
    ):
        self.data = (
            load_dataset(dataset_name, dataset_subset, split=split)
            if dataset_subset
            else load_dataset(dataset_name, split=split)
        )
        if max_samples is not None:
            self.data = self.data.select(range(max_samples))
        if world_size > 1:
            self.data = self.data.shard(num_shards=world_size, index=rank)
        self.processor = processor
        self.instruction = "Transcribe the audio clip into text."
        self.text_column = text_column
        self.audio_column = audio_column

    def __len__(self):
        return len(self.data)


class EvalDataset(BaseDataset):
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
        input_ids = inputs.input_ids
        labels = answer_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
        }


class FinetuneDataset(BaseDataset):
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
        dataset_subset=None,
    ):
        super().__init__(
            processor,
            dataset_name,
            split,
            text_column,
            audio_column,
            max_samples,
            rank,
            world_size,
            dataset_subset,
        )
        self.training = training

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


def load_model_and_processor(model_name_or_path, use_flash_attention=False):
    """Load the model and processor from the specified path or model name."""
    processor = AutoProcessor.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        trust_remote_code=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    return model, processor


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
    """
    Evaluate the model on the dataset and calculate WER and/or CER
    """
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
            max_new_tokens=320,
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

    results = {}
    if rank == 0:
        assert len(all_generated_texts) == len(all_labels)

        # Calculate either WER or CER or both based on parameter
        if metric.lower() == "wer" or metric.lower() == "both":
            wer_score = jiwer.wer(" ".join(all_labels), " ".join(all_generated_texts))
            results["wer"] = wer_score
            print(f"WER Score: {wer_score:.4f}")

        if metric.lower() == "cer" or metric.lower() == "both":
            cer_score = jiwer.cer(" ".join(all_labels), " ".join(all_generated_texts))
            results["cer"] = cer_score
            print(f"CER Score: {cer_score:.4f}")

        results["num_samples"] = len(all_labels)
        print(f"Number of samples: {len(all_labels)}")

        if save_path:
            with open(save_path, "w") as f:
                save_dict = {
                    "all_generated_texts": all_generated_texts,
                    "all_labels": all_labels,
                    **results,
                }
                json.dump(save_dict, f, indent=4, ensure_ascii=False)

    return results


def transcribe_audio(model, processor, audio_path):
    """Transcribe audio from the given file path."""

    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=16000)

    # Prepare input for the model
    user_message = {
        "role": "user",
        "content": "<|audio_1|> Transcribe the audio clip into text.",
    }
    prompt = processor.tokenizer.apply_chat_template(
        [user_message], tokenize=False, add_generation_prompt=True
    )

    # Process input
    inputs = processor(
        text=prompt,
        audios=[(audio, sr)],
        return_tensors="pt",
    )

    # Move inputs to the same device as the model
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
    }

    # Set up stopping criteria
    stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(
        stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt"
    )["input_ids"].to(model.device)
    stopping_criteria = StoppingCriteriaList(
        [MultipleTokenStoppingCriteria(stop_tokens_ids)]
    )

    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=320,
            stopping_criteria=stopping_criteria,
            do_sample=False,  # Deterministic generation
        )

    # Decode the generated text
    transcription = processor.decode(
        generated_ids[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Clean up the transcription (remove any potential end tokens that weren't caught by the stopping criteria)
    for token in stop_tokens:
        transcription = transcription.replace(token, "")

    return transcription.strip()
