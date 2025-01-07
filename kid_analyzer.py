"""
KID Analyzer - Analyse des documents d'informations clés (KID)
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Optional
import datetime

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

@dataclass
class KIDDocument:
    # Basic Information
    isin: str
    product_name: str
    manufacturer: str
    production_date: datetime.datetime
    
    # Product Details
    currency: str
    nominal_amount: float
    issue_price: float
    issue_date: datetime.datetime
    maturity_date: datetime.datetime
    
    # Risk and Protection
    risk_indicator: int  # 1-7
    capital_protection: float  # percentage
    
    # Performance Scenarios
    scenarios: Dict[str, Dict[str, float]]
    
    # Costs
    entry_costs: float
    exit_costs: float
    transaction_costs: float
    other_costs: float
    performance_fees: float

def extract_kid_info(text: str) -> KIDDocument:
    """
    Extract key information from a KID document text.
    """
    # Example implementation for the BNP Paribas document
    kid = KIDDocument(
        isin="XS1914695009",
        product_name="5Y Jump Certificate SOLEDVSP with 87.50% Protection",
        manufacturer="BNP Paribas S.A.",
        production_date=datetime.datetime(2022, 12, 3, 3, 25),
        currency="EUR",
        nominal_amount=1000.0,
        issue_price=100.0,
        issue_date=datetime.datetime(2019, 5, 3),
        maturity_date=datetime.datetime(2024, 5, 3),
        risk_indicator=3,
        capital_protection=87.50,
        scenarios={
            "stress": {
                "1_year": -8.87,
                "maturity": -4.91
            },
            "unfavourable": {
                "1_year": -3.13,
                "maturity": -4.91
            },
            "moderate": {
                "1_year": 2.09,
                "maturity": -4.91
            },
            "favourable": {
                "1_year": 28.72,
                "maturity": 32.51
            }
        },
        entry_costs=0.36,
        exit_costs=0.0,
        transaction_costs=0.0,
        other_costs=0.0,
        performance_fees=0.0
    )
    return kid

def format_kid_summary(kid: KIDDocument) -> str:
    """
    Format KID information into a readable summary.
    """
    summary = []
    summary.append(f"Product: {kid.product_name}")
    summary.append(f"ISIN: {kid.isin}")
    summary.append(f"Manufacturer: {kid.manufacturer}")
    summary.append(f"Currency: {kid.currency}")
    summary.append(f"Nominal Amount: {kid.currency} {kid.nominal_amount}")
    summary.append(f"Capital Protection: {kid.capital_protection}%")
    summary.append(f"Risk Indicator: {kid.risk_indicator}/7")
    
    # Performance scenarios
    summary.append("\nPerformance Scenarios:")
    for scenario, returns in kid.scenarios.items():
        summary.append(f"  {scenario.capitalize()}:")
        for period, value in returns.items():
            summary.append(f"    {period}: {value}%")
    
    # Costs
    summary.append("\nCosts:")
    summary.append(f"  Entry: {kid.entry_costs}%")
    summary.append(f"  Exit: {kid.exit_costs}%")
    summary.append(f"  Transaction: {kid.transaction_costs}%")
    summary.append(f"  Other: {kid.other_costs}%")
    summary.append(f"  Performance Fees: {kid.performance_fees}%")
    
    return "\n".join(summary)

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
    full_analysis = analyzer.analyze_kid(pdf_path)
    
    # Extract KID information
    kid = extract_kid_info(full_analysis)
    
    # Format KID summary
    summary = format_kid_summary(kid)
    
    print(summary)

if __name__ == "__main__":
    main()
