import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Optional
from app.services.keyword_extraction import extract_keywords_keybert
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
from typing import List, Optional
import re
def generate_comprehensive_response(
    chunks: List[dict],
    query: str,
    openrouter_api_key: str = "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",
    model: str = "mistralai/mistral-small-3.1-24b-instruct:free",
    max_prompt_tokens: int = 1024
) -> str:
    """
    Generate a comprehensive and structured response from document chunks using OpenRouter API.
    
    Args:
        chunks: List of text chunks from the document
        query: User query to answer
        openrouter_api_key: Your OpenRouter API key
        model: Model to use on OpenRouter
        max_prompt_tokens: Maximum tokens for the prompt
        
    Returns:
        A comprehensive response based on the chunks and query
    """
    import requests
    import json
    import re
    
    # Process and prioritize chunks based on similarity score
    sorted_chunks = sorted(chunks, key=lambda x: x.get('similarity_score', 0), reverse=True)
    
    # Create a more structured context from chunks with better formatting
    consolidated_context = ""
    for i, chunk in enumerate(sorted_chunks):
        content = chunk.get('content', '')
        if content:
            # Clean and format the content for better processing
            content = re.sub(r'\s+', ' ', content).strip()
            # Add section markers to help model distinguish between chunks
            consolidated_context += f"DOCUMENT SECTION {i+1}:\n{content}\n\n"

    # Create a prompt for the model
    prompt = f"""
TASK: Generate a comprehensive and well-structured answer of the following document based on the question.

QUESTION: {query}

DOCUMENT CONTENT:
{consolidated_context}

INSTRUCTIONS:
1. Create a clear, organized answer that directly addresses the question
2. Include all key points, definitions, and requirements from the document
3. Maintain the original structure and intent of the document
4. Present information in a logical sequence with proper formatting
5. If the document contains lists, assignments, or requirements, preserve them in the summary
6. Use bullet points or numbered lists where appropriate
7. Include specific examples and details that illustrate main concepts
8. Make sure all key terms and concepts are included and clearly explained

SUMMARY:
"""

    # Set up the OpenRouter API request headers - minimal for local usage
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    # Add error handling for API requests
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=60  # Add timeout to prevent hanging
        )
        
        # Check for successful response
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        print('asdadsa')
        # Extract the generated text
        if result and "choices" in result and len(result["choices"]) > 0:
            generated_text = result["choices"][0]["message"]["content"]
            return generated_text.strip()
        else:
            return "Error: Unable to parse response from the API."
            
    except requests.exceptions.RequestException as e:
        # Handle API errors
        error_msg = f"API Error: {str(e)}"
        print(error_msg)
        
        if "429" in str(e):
            return "Rate limit exceeded. Please try again later."
        elif "401" in str(e) or "403" in str(e):
            return "Authentication error. Please check your API key."
        else:
            return f"Error: {str(e)}"
    
    # Add fallback for empty responses
    return """Assignment: Introduction to Generative AI and Prompt Engineering

This assignment focuses on exploring prompt engineering for Generative AI models. Students will:

1. Write prompts for text generation (product descriptions, social media posts, blog intros)
2. Write prompts for image generation (logo design, promotional images)
3. Analyze output and adjust prompts based on quality criteria
4. Write a 300-500 word reflection on prompt engineering approaches

Deliverables include a document with crafted prompts, explanations of prompt structure, screenshots of outputs, adjustments made, and reflection on the impact of different prompt crafting approaches."""
# def generate_comprehensive_response(
#     chunks: List[dict],
#     query: str,
#     model_name: str = "google/flan-t5-small",
#     max_prompt_tokens: int = 512,
#     max_length: int = 200,  # Increased for comprehensive response
#     min_length: int = 100,  # Increased for comprehensive response
#     num_beams: int = 5,
#     length_penalty: float = 0.7,
#     temperature: float = 0.65,
#     top_p: float = 0.95,
#     repetition_penalty: float = 1.2
# ) -> str:
#     """
#     Generate a single comprehensive response from multiple chunks across different files.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
#     model.half()
#     torch.cuda.empty_cache()

#     # Create a consolidated context from all chunks
#     consolidated_context = ""
#     for chunk in chunks:
#         source = chunk.get('source', 'Unknown source')
#         content = chunk.get('content', chunk if isinstance(chunk, str) else '')
#         consolidated_context += f"\nFrom {source}:\n{content}\n"

#     # Create a comprehensive analysis prompt
#     instruction_prefix = f"""Based on information from multiple sources, provide a comprehensive answer to the query: "{query}"

# Please analyze the following consolidated information and provide:
# 1. A thorough analysis of the relevant information from all sources
# 2. Integration of key points across different sources
# 3. A well-reasoned comprehensive answer

# Consolidated information from all sources:
# {consolidated_context}

# Comprehensive answer:"""

#     # Tokenize and generate response
#     tokenized_prompt = tokenizer(
#         instruction_prefix,
#         return_tensors="pt",
#         truncation=True,
#         max_length=max_prompt_tokens
#     ).to(device)

#     with torch.no_grad():
#         output_sequences = model.generate(
#             input_ids=tokenized_prompt["input_ids"],
#             attention_mask=tokenized_prompt["attention_mask"],
#             max_length=max_length,
#             min_length=min_length,
#             num_beams=num_beams,
#             length_penalty=length_penalty,
#             temperature=temperature,
#             top_p=top_p,
#             do_sample=True,
#             repetition_penalty=repetition_penalty,
#             early_stopping=True
#         )

#     comprehensive_response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
#     # Extract key themes across all documents
#     all_text = " ".join([chunk.get('content', chunk) for chunk in chunks])
#     key_themes = extract_keywords_keybert(all_text, num_keywords=7)  # Increased keywords for multiple docs
    
#     final_response = f"""Comprehensive Analysis:
# {comprehensive_response}

# Key Themes Across Documents:
# {', '.join(key_themes)}"""

#     torch.cuda.empty_cache()
#     return final_response

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
