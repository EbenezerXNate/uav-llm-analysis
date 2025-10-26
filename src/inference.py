import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class UAVReportAnalyzer:
    """
    Fine-tuned LLM for extracting structured information from UAV sensor reports.
    Converts unstructured military UAV reports to consistent JSON format.
    """
    
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"✅ UAV Report Analyzer loaded from {model_path}")
    
    def analyze(self, report_text: str) -> dict:
        """
        Analyze UAV report and extract structured information.
        
        Args:
            report_text: Unstructured UAV report text
            
        Returns:
            Dictionary with success status and extracted data
        """
        prompt = f"### Instruction:\nUAV Report: {report_text}\n\n### Response:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=400,
                num_beams=3,
                early_stopping=True,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response
        if "### Response:" in result_text:
            response = result_text.split("### Response:")[1].strip()
        else:
            response = result_text
        
        # Parse JSON
        try:
            json_start = response.find('{')
            json_end = response.find('}', json_start) + 1
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                structured_data = json.loads(json_str)
                return {
                    "success": True,
                    "data": structured_data,
                    "raw_output": response
                }
        except json.JSONDecodeError:
            pass
        
        return {
            "success": False,
            "error": "Could not parse JSON",
            "raw_output": response
        }

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = UAVReportAnalyzer("./models/uav-llm-finetuned")
    
    # Test examples
    test_reports = [
        "UAV Patrol Alpha detected three thermal signatures at grid 38FRP123456. No civilian activity.",
        "UAV Recon Bravo spotted two enemy patrols at grid 42GQM789123. Civilian vehicles nearby."
    ]
    
    for i, report in enumerate(test_reports, 1):
        print(f"\nReport {i}: {report}")
        result = analyzer.analyze(report)
        if result["success"]:
            print("✅ Extracted Data:")
            print(json.dumps(result["data"], indent=2))
        else:
            print(f"❌ Error: {result['error']}")
