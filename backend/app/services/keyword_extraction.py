# extract_keywords_keybert
from keybert import KeyBERT

# Initialize KeyBERT model
#i donnooo
# kw_model = KeyBERT('paraphrase-multilingual-mpnet-base-v2')
# Extracted Keywords: ['documents sample questions', 'utilize local', 'llm runs', 'initialization provided pdf', 'insights ensure llm']

# Extracted Keywords: ['documents extract text', 'query engine', 'document embeddings integrate', 'content engine', 'comparison analysis documents']
# kw_model = KeyBERT('multi-qa-mpnet-base-dot-v1')
# Extracted Keywords: ['google tesla', 'uber development', 'compare numbers documents', 'insights ensure llm', 'content pdfs compare']
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# # Define base and custom stop words
# base_stop_words = set(ENGLISH_STOP_WORDS)
# additional_stop_words = {
#     'data', 'pdf', 'file', 'document', 'information',
#     'result', 'using', 'system', 'content', 'process',
#     'output', 'input', 'extract', 'based', 'develop', 'analysis'
# }
# Combine stop words
# final_stop_words = base_stop_words.union(additional_stop_words)


def extract_keywords_keybert(chunk,kw_model = KeyBERT('sentence-transformers/all-mpnet-base-v2'), num_keywords=5):
    """
    Extracts keywords from a text chunk using KeyBERT.

    Parameters:
        chunk (str): The input text chunk.
        num_keywords (int): Number of keywords to extract.

    Returns:
        list: List of extracted keywords.
    """
    # Generate keywords
    #keywords = kw_model.extract_keywords(chunk, keyphrase_ngram_range=(1, 2), , top_n=num_keywords)
    keywords=kw_model.extract_keywords(chunk,keyphrase_ngram_range=(1, 3), use_maxsum=True,stop_words='english', nr_candidates=20, top_n=5)

    return [keyword[0] for keyword in keywords]

# # Example chunk
# chunk = """
# #  for processing and generating insights. Ensure the LLM runs locally and is not exposed to any external APIs. 2. Initialization You are provided three PDF documents containing the Form 10-K filings of multinational companies. These documents will serve as the basis for your comparison analysis. The documents are as follows : 1. Alphabet Inc. Form 10-K 2. Tesla, Inc. Form 10-K 3. Uber Technologies, Inc. Form 10-K You will use these documents to implement and test your Content Engine. Your task is to retrieve the content from these PDFs, compare them, and answer queries highlighting the information across all documents. Additionally, the end system should feature a chatbot interface where users can interact and obtain insights about information from the documents, compare numbers within these three documents, and more. Sample Questions - 1 ) What are the risk factors associated with Google and Tesla? 2 ) What is the total revenue for Google Search? 3 ) What are the differences in the business of Tesla and Uber? 3. Development ● Parse Documents : Extract text and structure from PDFs. ● Generate Vectors : Use a local embedding model to create embeddings for document content. ● Store in Vector Store : Utilize local persisting methods in the chosen vector store. ● Configure Query Engine : Set up retrieval tasks based on document embeddings. ● Integrate LLM : Run a local instance of a Large Language Model for contextual insights. ● Develop Chatbot Interface : Use Streamlit to facilitate user interaction and display comparative insights. 4.
# #  """

# #  # Extract keywords using KeyBERT
# keywords = extract_keywords_keybert(chunk)
# print("Extracted Keywords:", keywords)


# from keybert import KeyBERT

# # Initialize KeyBERT model once
# kw_model = KeyBERT('sentence-transformers/all-mpnet-base-v2')

# def extract_keywords_keybert(chunk, num_keywords=5):
#     """
#     Extracts keywords from a text chunk using KeyBERT.

#     Parameters:
#         chunk (str or list): The input text chunk. Can be a string or list of sentences.
#         num_keywords (int): Number of keywords to extract.

#     Returns:
#         list: List of extracted keywords.
#     """
#     print(chunk)
#     # If chunk is a list, join it into a single string
#     if isinstance(chunk, list):
#         chunk = chunk[0][1]

#     # Ensure chunk is not empty
#     if not chunk.strip():
#         return []

#     # Generate keywords
#     keywords = kw_model.extract_keywords(
#         chunk,
#         keyphrase_ngram_range=(1, 3),
#         use_maxsum=True,
#         stop_words='english',
#         nr_candidates=20,
#         top_n=num_keywords
#     )

#     return [keyword[0] for keyword in keywords]
