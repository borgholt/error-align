import regex as re

from error_align.utils import basic_normalizer, basic_tokenizer

def clean_tedlium_ref(example: dict) -> dict:
    """
    Cleans the TED-LIUM dataset by removing examples with empty transcriptions.
    
    Args:
        text: The text content of the TED-LIUM dataset.
        
    Returns:
        A cleaned version of the text with empty transcriptions removed.
    """
    
    # Get example.
    text = example["ref"]

    # Remove all tags, e.g., <unk>
    text = re.sub(r'<[^>]+>', '', text)
    
    # Re-contract apostrophes
    text = re.sub(r"(\w) '(\w)", r"\1'\2", text)
    
    # Get normalized tokens.
    normalized_tokens = [basic_normalizer(token.group()) for token in basic_tokenizer(text)]

    # Remove leading/trailing whitespace and update example.
    example["ref"] = " ".join(normalized_tokens)

    return example

def clean_whisper_hyp(example: dict) -> dict:
    """
    Cleans the Whisper hypothesis by removing examples with empty transcriptions.

    Args:
        example: The example to clean.

    Returns:
        A cleaned version of the example with empty transcriptions removed.
    """
    # Get example and lower case text.
    text = example["hyp"]
    
    # Get normalized tokens.
    normalized_tokens = [basic_normalizer(token.group()) for token in basic_tokenizer(text)]

    # Remove leading/trailing whitespace and update example.
    example["hyp"] = " ".join(normalized_tokens)

    return example