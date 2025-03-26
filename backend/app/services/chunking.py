# agentic_chunking, verify_chunks
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Tuple
# Download the 'punkt_tab' data package
# nltk.download('punkt_tab')
def agentic_chunking(text: str, max_tokens: int = 300, overlap: int = 30) -> List[str]:
    """
    Splits text into chunks with absolutely strict token limit enforcement.

    Args:
        text (str): Input text
        max_tokens (int): Absolute maximum tokens per chunk
        overlap (int): Number of tokens to overlap between chunks

    Returns:
        List[str]: List of chunks, each guaranteed to be <= max_tokens
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    def tokenize_text(text: str) -> List[str]:
        return word_tokenize(text)

    def reconstruct_text(tokens: List[str]) -> str:
        """Carefully reconstruct text from tokens."""
        text = ' '.join(tokens)
        # Basic cleanup of space before punctuation
        text = text.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?')
        return text

    chunks = []
    sentences = sent_tokenize(text)
    current_tokens = []

    for sentence in sentences:
        sentence_tokens = tokenize_text(sentence)

        # If single sentence is longer than max_tokens, split it
        if len(sentence_tokens) > max_tokens:
            # First add any existing chunk
            if current_tokens:
                chunks.append(reconstruct_text(current_tokens))
                current_tokens = []

            # Split long sentence into fixed-size chunks
            for i in range(0, len(sentence_tokens), max_tokens - overlap):
                chunk_tokens = sentence_tokens[i:i + max_tokens]
                if len(chunk_tokens) > overlap:  # Only add if chunk is substantial
                    chunks.append(reconstruct_text(chunk_tokens))
            continue

        # Check if adding new sentence exceeds limit
        if len(current_tokens) + len(sentence_tokens) > max_tokens:
            # If current chunk plus overlap would exceed limit
            if len(current_tokens) > max_tokens - overlap:
                # Save current chunk
                chunks.append(reconstruct_text(current_tokens))
                # Start new chunk with overlap
                overlap_size = min(overlap, len(current_tokens))
                current_tokens = current_tokens[-overlap_size:] if overlap_size > 0 else []

            # If remaining tokens plus new sentence still exceed limit
            if len(current_tokens) + len(sentence_tokens) > max_tokens:
                if current_tokens:  # Save any remaining tokens
                    chunks.append(reconstruct_text(current_tokens))
                current_tokens = []

                # Handle sentence that's too long for remaining space
                if len(sentence_tokens) > max_tokens - overlap:
                    chunks.append(reconstruct_text(sentence_tokens[:max_tokens]))
                    current_tokens = sentence_tokens[-overlap:] if overlap > 0 else []
                else:
                    current_tokens = sentence_tokens
            else:
                current_tokens.extend(sentence_tokens)
        else:
            current_tokens.extend(sentence_tokens)

    # Add final chunk if any tokens remain
    if current_tokens:
        if len(current_tokens) > max_tokens:
            # Split final chunk if it's too long
            chunks.append(reconstruct_text(current_tokens[:max_tokens]))
            if len(current_tokens) > max_tokens + overlap:
                chunks.append(reconstruct_text(current_tokens[max_tokens:]))
        else:
            chunks.append(reconstruct_text(current_tokens))

    return chunks

def verify_chunks(chunks: List[str], max_tokens: int) -> Tuple[bool, List[int]]:
    """
    Verify that no chunk exceeds the token limit.

    Returns:
        Tuple[bool, List[int]]: (success, list of chunk lengths)
    """
    chunk_lengths = [len(word_tokenize(chunk)) for chunk in chunks]
    success = all(length <= max_tokens for length in chunk_lengths)
    return success, chunk_lengths