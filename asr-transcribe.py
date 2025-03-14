import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)
import librosa


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
            max_new_tokens=64,
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


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Phi-4-multimodal model"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-4-multimodal-instruct",
        help="Model name or path to load from",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the audio file to transcribe",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention for more efficient inference on compatible hardware",
    )
    args = parser.parse_args()

    # Load model and processor
    print(f"Loading model from {args.model_name_or_path}...")
    model, processor = load_model_and_processor(
        args.model_name_or_path, use_flash_attention=args.use_flash_attention
    )

    # Transcribe audio
    print(f"Transcribing audio from {args.audio_path}...")
    transcription = transcribe_audio(model, processor, args.audio_path)

    # Print the transcription
    print("\nTranscription:")
    print("-" * 40)
    print(transcription)
    print("-" * 40)


if __name__ == "__main__":
    main()
