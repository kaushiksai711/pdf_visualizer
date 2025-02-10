import { PDFDocument } from 'pdf-lib';

export const extractTextFromPDF = async (fileName) => {
  try {
    const response = await fetch(`http://localhost:5000/uploads/${fileName}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch PDF: ${response.statusText}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const pdfDoc = await PDFDocument.load(arrayBuffer);
    
    let text = '';
    for (let i = 0; i < pdfDoc.getPageCount(); i++) {
      const page = pdfDoc.getPage(i);
      const content = await page.getTextContent();
      text += content.items.map(item => item.str).join(' ') + ' ';
    }
    
    return text;
  } catch (error) {
    console.error('Error extracting text from PDF:', error);
    throw error; // Rethrow the error for further handling
  }
}; 