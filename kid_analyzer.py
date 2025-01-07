"""
KID Analyzer - Analyse des documents d'informations clés (KID)
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

class KIDAnalyzer:
    def __init__(self, hf_token=None):
        """Initialise l'analyseur KID."""
        self.model_name = "Qwen/Qwen-7B-Chat"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=hf_token,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            use_auth_token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            max_memory={0: "10GB"}
        ).eval()

    def convert_pdf_to_images(self, pdf_path):
        """Convertit les pages PDF en images."""
        return convert_from_path(pdf_path)

    def extract_text_from_image(self, image):
        """Extrait le texte d'une image avec OCR."""
        return pytesseract.image_to_string(image)

    def analyze_kid(self, pdf_path):
        """Analyse un document KID et extrait les informations structurées."""
        print(f"\nAnalyzing document: {pdf_path}")
        
        # Convert PDF to images
        print("\nConverting PDF to images...")
        images = self.convert_pdf_to_images(pdf_path)
        
        # System prompt for the model
        system_prompt = """Analyze this KID document and extract key information in this format:

1. PRODUCT IDENTIFICATION
Name:
ISIN:
Manufacturer:
Contact Details:

2. RISK ASSESSMENT
Risk Level:
Risk Description:

3. PERFORMANCE SCENARIOS
Investment Amount:
Scenarios:
- Stress:
- Unfavorable:
- Moderate:
- Favorable:

4. COSTS
Entry Costs:
Exit Costs:
Annual Costs:

5. RECOMMENDED HOLDING PERIOD
Period:
Early Exit Conditions:

Extract ONLY the information that is present in the text. Use 'Not specified' for missing information."""

        # Process each page
        full_analysis = ""
        
        for i, image in enumerate(images):
            print(f"\nAnalyzing page {i+1}...")
            
            # Extract text from image
            extracted_text = self.extract_text_from_image(image)
            print(f"\nExtracted text from page {i+1}:")
            print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
            
            # Analyze the extracted text
            query = f"Please extract the relevant information from this page. Here is the text:\n\n{extracted_text}"
            
            response = self.model.chat(
                self.tokenizer,
                query,
                history=[{"role": "system", "content": system_prompt}]
            )
            
            # Handle tuple response from model
            if isinstance(response, tuple):
                response = response[0]
            
            print(f"\nModel response for page {i+1}:")
            print(response[:200] + "..." if len(response) > 200 else response)
            
            full_analysis += response + "\n\n"
        
        # Save raw output
        with open("kid_analysis_raw.txt", "w") as f:
            f.write(full_analysis)
            
        print("\nAnalysis complete. Results saved to kid_analysis_raw.txt")
        return full_analysis

def main():
    """Point d'entrée principal."""
    # Initialize analyzer
    analyzer = KIDAnalyzer()
    
    # Example PDF path
    pdf_path = "XS1914695009-EN.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    # Analyze document
    analyzer.analyze_kid(pdf_path)

if __name__ == "__main__":
    main()
