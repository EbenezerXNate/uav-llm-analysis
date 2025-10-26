"""
Example usage of the UAV Report Analyzer
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference import UAVReportAnalyzer

def main():
    # Initialize the analyzer
    print("🚀 Initializing UAV Report Analyzer...")
    analyzer = UAVReportAnalyzer("./models/uav-llm-finetuned")
    
    # Test cases
    test_reports = [
        "UAV Patrol Alpha detected three thermal signatures at grid 38FRP123456. Possible enemy patrol. No civilian activity seen.",
        "UAV Recon Bravo spotted two vehicle movements and one supply depot at grid 42GQM789123. Civilian activity detected nearby.",
        "UAV Surveillance Charlie identified four weapon systems at grid 51HRN456789. No civilian presence observed.",
        "UAV Strike Delta observed single enemy patrol at grid 63JSP111222. High confidence identification."
    ]
    
    print("\n🧪 Running Test Cases...")
    print("=" * 60)
    
    for i, report in enumerate(test_reports, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {report}")
        
        result = analyzer.analyze(report)
        
        if result["success"]:
            print("✅ SUCCESS - Extracted Structured Data:")
            import json
            print(json.dumps(result["data"], indent=2))
        else:
            print(f"❌ FAILED: {result['error']}")
            print(f"Raw output: {result['raw_output']}")
        
        print("-" * 40)

if __name__ == "__main__":
    main()