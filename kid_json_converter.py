from typing import Dict, Any
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class KIDJsonConverter:
    def __init__(self, model_path="mistralai/Mistral-7B-Instruct-v0.1"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def _create_prompt(self, text: str) -> str:
        example_json = '''
{
    "product_info": {
        "name": "5Y Jump Certificate SOLEDVSP with 87.50% Protection",
        "isin": "XS1914695009",
        "manufacturer": {
            "name": "BNP Paribas S.A.",
            "contact": "+33 (0)1 57 08 22 00"
        },
        "issuer": "BNP Paribas Issuance B.V.",
        "guarantor": "BNP Paribas S.A."
    },
    "dates": {
        "production_date": "2022-12-03",
        "issue_date": "2019-05-03",
        "maturity_date": "2024-05-03",
        "strike_date": "2019-04-26",
        "redemption_valuation_date": "2024-04-26"
    },
    "financial_details": {
        "currency": "EUR",
        "nominal_amount": 1000.0,
        "issue_price": 100.0,
        "underlying": {
            "name": "Solactive European Deep Value Select 50 Index",
            "bloomberg_code": "SOLEDVSP"
        },
        "capital_protection": {
            "percentage": 87.50,
            "conditions": "Protection applies only at maturity"
        }
    },
    "risk_assessment": {
        "sri_rating": 3,
        "risk_description": "Medium-low risk product",
        "capital_risk": "87.50% of capital protected at maturity",
        "market_risk": "Performance linked to underlying index",
        "currency_risk": "Product in EUR"
    },
    "performance_scenarios": {
        "stress": {
            "one_year": {
                "return": -8.87,
                "value": 9113.11
            },
            "maturity": {
                "return": -4.91,
                "value": 9311.93
            }
        },
        "unfavourable": {
            "one_year": {
                "return": -3.13,
                "value": 9686.59
            },
            "maturity": {
                "return": -4.91,
                "value": 9311.93
            }
        },
        "moderate": {
            "one_year": {
                "return": 2.09,
                "value": 10209.38
            },
            "maturity": {
                "return": -4.91,
                "value": 9311.93
            }
        },
        "favourable": {
            "one_year": {
                "return": 28.72,
                "value": 12872.49
            },
            "maturity": {
                "return": 32.51,
                "value": 14899.09
            }
        }
    },
    "costs": {
        "entry": 0.36,
        "exit": 0.0,
        "transaction": 0.0,
        "other_ongoing": 0.0,
        "performance_fees": 0.0,
        "total_costs": {
            "after_one_year": 1.08,
            "at_maturity": 0.36
        }
    },
    "investor_profile": {
        "recommended_holding_period": "Until 03 May 2024 (maturity)",
        "intended_retail_investor": {
            "knowledge_level": "Informed or experienced",
            "financial_situation": "Able to bear limited losses",
            "investment_horizon": "Short term (less than 3 years)",
            "risk_tolerance": "Willing to accept medium-low risk"
        }
    },
    "regulatory_info": {
        "competent_authority": "Autorité des marchés financiers (AMF)",
        "jurisdiction": "Ireland",
        "selling_restrictions": ["Not for US persons"],
        "complaint_procedure": {
            "contact": "BNP Paribas CLM Regulations - Complaints Management",
            "website": "https://kid.bnpparibas.com/cib",
            "email": "cib.priips.complaints@bnpparibas.com"
        }
    }
}'''

        prompt = f"""<s>[INST] Tu es un expert en analyse de documents financiers KID (Key Information Document).
Ta tâche est d'extraire les informations du document et de les structurer en JSON.

Voici un exemple complet de la structure attendue (avec des valeurs d'exemple) :
{example_json}

Règles importantes :
1. Utilise EXACTEMENT la même structure que l'exemple
2. Remplace les valeurs de l'exemple par les vraies données trouvées dans le document
3. Conserve tous les champs, même ceux qui ne sont pas dans l'exemple
4. Utilise les valeurs exactes (ne pas arrondir les nombres)
5. Pour les dates, utilise le format ISO (YYYY-MM-DD)
6. Si une information n'est pas trouvée, mets null
7. Respecte les types (number pour les nombres, string pour le texte)

Voici le document à analyser :
{text}

Réponds UNIQUEMENT avec le JSON complet contenant les vraies données, sans autre texte.[/INST]</s>"""

        return prompt

    def convert_to_json(self, text: str) -> Dict[str, Any]:
        prompt = self._create_prompt(text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=4096,
            temperature=0.1,
            top_p=0.95,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response (assuming the model might add some text before/after)
        try:
            # Find the first { and last } to extract the JSON part
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing JSON response: {e}")
            print("Raw response:", response)
            return {}

def main():
    # Initialize converter
    converter = KIDJsonConverter()
    
    # Read the raw text file
    with open("kid_analysis_raw.txt", "r") as f:
        raw_text = f.read()
    
    # Convert to JSON
    json_data = converter.convert_to_json(raw_text)
    
    # Save the JSON output
    with open("kid_analysis.json", "w") as f:
        json.dump(json_data, f, indent=2)
    
    print("\nAnalysis complete. Results saved to kid_analysis.json")

if __name__ == "__main__":
    main()
