import argparse

from .common import load_model_and_processor, transcribe_audio


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
