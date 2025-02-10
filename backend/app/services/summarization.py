import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Optional
from app.services.keyword_extraction import extract_keywords_keybert

def generate_with_flant5_individual(
    chunks: List[str],
    query: Optional[str] = None,
    include_reasoning: bool = False,
    model_name: str = "google/flan-t5-small",
    max_prompt_tokens: int = 512,
    max_length: int = 100,
    min_length: int = 40,
    num_beams: int = 5,
    length_penalty: float = 0.7,
    no_repeat_ngram_size: int = 3,
    temperature: float = 0.65,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2
) -> str:
    """
    Summarize each chunk independently using FLAN-T5 with GPU acceleration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.half()
    torch.cuda.empty_cache()

    instruction_prefix = """Provide a detailed response with the following structure:
1. Initial thoughts about the query
2. Analysis of relevant information
3. Reasoning process
4. Final answer

For the following text:
"""
    aggregated_responses = []
    
    # Handle both single chunks and lists of chunks
    if isinstance(chunks, str):
        chunks_to_process = [chunks]
    elif isinstance(chunks, list) and len(chunks) > 0:
        if isinstance(chunks[0], dict) and 'content' in chunks[0]:
            # Handle chunks from chat interface
            chunks_to_process = [chunk['content'] for chunk in chunks]
        else:
            # Handle regular chunks
            chunks_to_process = chunks
    else:
        return "No valid chunks to process"

    for i, chunk in enumerate(chunks_to_process):
        # Ensure chunk is a string
        if isinstance(chunk, list):
            chunk = chunk[0] if chunk else ""
        
        print(f"Processing chunk {i + 1}/{len(chunks_to_process)}...")

        # Prepare prompt
        if include_reasoning:
            prompt = f"Based on this query: {query}\n{instruction_prefix}{chunk}"
        else:
            prompt = f"{instruction_prefix}{chunk}"

        # Tokenize and generate summary
        tokenized_prompt = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens
        ).to(device)

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=tokenized_prompt["input_ids"],
                attention_mask=tokenized_prompt["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                early_stopping=True
            )

        summary = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        try:
            # Extract keywords from the original chunk text
            keywords = extract_keywords_keybert(chunk if isinstance(chunk, str) else str(chunk))
            aggregated_responses.append(
                f"Chunk {i + 1} Summary:\n{summary}\nExtracted Keywords:\n{keywords}"
            )
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            aggregated_responses.append(f"Chunk {i + 1} Summary:\n{summary}")

        torch.cuda.empty_cache()

    return "\n\n".join(aggregated_responses)

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from typing import List, Optional
# from app.services.keyword_extraction import extract_keywords_keybert

# def generate_with_flant5_individual(
#     chunks: List[str],
#     query: Optional[str] = None,
#     model_name: str = "google/flan-t5-large",
#     max_prompt_tokens: int = 512,
#     max_length: int = 100,
#     min_length: int = 40,
#     num_beams: int = 7,
#     length_penalty: float = 0.7,
#     no_repeat_ngram_size: int = 3,
#     temperature: float = 0.5,
#     top_p: float = 0.9,
#     repetition_penalty: float = 1.2
# ) -> str:
#     """
#     Summarize each chunk independently using FLAN-T5 with GPU acceleration.
#     """

#     # Check if CUDA (GPU) is available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")  # Should print 'cuda' if GPU is used

#     # Initialize model and tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  # Move model to GPU
    
#     instruction_prefix = "Write a concise summary of the following text:\n\n"
#     aggregated_responses = []

#     chunk_ids, chunks1 = [chunk[0] for chunk in chunks], [[chunk[1]] for chunk in chunks]
#     chunks = chunks1  # Restructure chunks
    
#     for i, chunk in enumerate(chunks):
#         print(f"Processing chunk {chunk_ids[i]}/{len(chunks)}...")

#         # Prepare prompt with instruction
#         prompt = f"Based on this query: {query}\n{instruction_prefix}{chunk}" if query else f"{instruction_prefix}{chunk}"

#         # Tokenize with truncation
#         tokenized_prompt = tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=max_prompt_tokens
#         ).to(device)  # Move input tensors to GPU

#         # Generate summary on GPU
#         with torch.no_grad():  # Disable gradient calculation for faster inference
#             output_sequences = model.generate(
#                 input_ids=tokenized_prompt.input_ids,
#                 attention_mask=tokenized_prompt.attention_mask,
#                 max_length=max_length,
#                 min_length=min_length,
#                 num_beams=num_beams,
#                 length_penalty=length_penalty,
#                 no_repeat_ngram_size=no_repeat_ngram_size,
#                 temperature=temperature,
#                 top_p=top_p,
#                 do_sample=True,
#                 repetition_penalty=repetition_penalty,
#                 early_stopping=True
#             )

#         # Decode the generated summary
#         summary = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

#         # Extract keywords (if function is defined)
#         try:
#             keywords = extract_keywords_keybert1(chunk)
#             aggregated_responses.append(
#                 f"Chunk {i + 1} Summary:\n{summary}\nExtracted Keywords:\n{keywords}"
#             )
#         except NameError:
#             aggregated_responses.append(f"Chunk {i + 1} Summary:\n{summary}")

#     # Combine all summaries
#     final_output = "\n\n".join(aggregated_responses)
#     return final_output
