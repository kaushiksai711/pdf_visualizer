# extract_text_from_pdf
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    
    Args:
        pdf_path (str): The path to the PDF file.
    
    Returns:
        str: The extracted text from the PDF.
    """
    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)  # Create a PDF reader object
            text = ""
            # Loop through all pages and extract text
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return ""