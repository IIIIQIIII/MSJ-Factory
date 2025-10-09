# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ChnSentiCorp Sentiment Analysis Comparison Evaluation Script

Features:
- Compare model performance before and after fine-tuning
- Integrated inference and evaluation pipeline
- Compliant with LlamaFactory development standards
"""
import json
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any
import gc

import fire
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class SentimentEvaluator:
    """Sentiment Analysis Evaluator"""
    
    def __init__(
        self,
        csv_path: str,
        base_model_path: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        finetuned_model_path: str = "saves/qwen2_5-coder-1.5b/freeze/sft",
        template: str = "qwen",
        max_samples: Optional[int] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 256,
        device: str = "auto"
    ):
        self.csv_path = csv_path
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.template = template
        self.max_samples = max_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device
        
        # Load dataset
        print(f"📁 Loading dataset: {csv_path}")
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=42)
        print(f"Dataset size: {len(self.df)} samples")
        print(f"Label distribution:\n{self.df['label'].value_counts()}\n")
    
    def create_prompt(self, text: str) -> str:
        """Create sentiment analysis prompt"""
        return f"""Please perform sentiment analysis on the following Chinese text and determine its sentiment orientation.

Task Description:
- Analyze the overall sentiment attitude expressed in the text
- Determine whether it is positive (1) or negative (0)

Text Content:
```sentence
{text}
```

Output Format:
```json
{{
  "sentiment": 0 or 1
}}
```

Please output the JSON result only, without any other irrelevant text"""
    
    def parse_sentiment(self, response: str) -> Optional[int]:
        """Parse sentiment label from model output"""
        # Method 1: Find content wrapped in ```json ... ```
        json_match = re.search(r'```json\s*\{[^}]*"sentiment"\s*:\s*([01])[^}]*\}\s*```', response, re.DOTALL)
        if json_match:
            return int(json_match.group(1))
        
        # Method 2: Find content wrapped in ```
        json_match = re.search(r'```\s*\{[^}]*"sentiment"\s*:\s*([01])[^}]*\}\s*```', response, re.DOTALL)
        if json_match:
            return int(json_match.group(1))
        
        # Method 3: Directly find "sentiment": 0 or 1
        match = re.search(r'"sentiment"\s*:\s*([01])', response)
        if match:
            return int(match.group(1))
        
        # Method 4: Try JSON parsing
        try:
            cleaned = re.sub(r'```json|```', '', response).strip()
            data = json.loads(cleaned)
            if "sentiment" in data and data["sentiment"] in [0, 1]:
                return int(data["sentiment"])
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
    
    def evaluate_model(self, model_path: str, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model"""
        print(f"\n{'='*70}")
        print(f"🔍 Evaluating Model: {model_name}")
        print(f"📂 Model Path: {model_path}")
        print(f"{'='*70}\n")
        
        # Load model
        print("Loading model...")
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=self.device,
            trust_remote_code=True
        )
        model.eval()
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully (Time: {load_time:.2f}s)\n")
        
        # Initialize statistics
        results = {
            "model_name": model_name,
            "model_path": model_path,
            "total": 0,
            "correct": 0,
            "parse_failed": 0,
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "predictions": [],
            "inference_time": 0
        }
        
        # Inference and evaluation for each sample
        inference_start = time.time()
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"Inference Progress ({model_name})"):
            text = row['text_a']
            true_label = int(row['label'])
            
            # Create messages
            messages = [
                {"role": "user", "content": self.create_prompt(text)}
            ]
            
            # Apply chat template
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Inference
            model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    top_p=0.9
                )
            
            # Decode response
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Parse prediction result
            predicted_label = self.parse_sentiment(response)
            
            if predicted_label is None:
                results["parse_failed"] += 1
                predicted_label = 0  # Default value
            
            # Check if prediction is correct
            is_correct = (predicted_label == true_label)
            if is_correct:
                results["correct"] += 1
            
            # Update confusion matrix
            if true_label == 1 and predicted_label == 1:
                results["true_positives"] += 1
            elif true_label == 0 and predicted_label == 0:
                results["true_negatives"] += 1
            elif true_label == 0 and predicted_label == 1:
                results["false_positives"] += 1
            elif true_label == 1 and predicted_label == 0:
                results["false_negatives"] += 1
            
            results["total"] += 1
            
            # Save detailed results (only first 100)
            if len(results["predictions"]) < 100:
                results["predictions"].append({
                    "qid": int(row['qid']),
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "is_correct": is_correct,
                    "response": response[:200] + "..." if len(response) > 200 else response
                })
        
        results["inference_time"] = time.time() - inference_start
        
        # Calculate metrics
        total = results["total"]
        accuracy = results["correct"] / total if total > 0 else 0
        
        tp = results["true_positives"]
        tn = results["true_negatives"]
        fp = results["false_positives"]
        fn = results["false_negatives"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results["metrics"] = {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1_score * 100, 2)
        }
        
        # Print results
        self._print_model_results(results)
        
        # Clean up model
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
        return results
    
    def _print_model_results(self, results: Dict[str, Any]):
        """Print evaluation results for a single model"""
        metrics = results["metrics"]
        print(f"\n{'='*70}")
        print(f"📊 {results['model_name']} Evaluation Results")
        print(f"{'='*70}")
        print(f"\nTotal Samples: {results['total']}")
        print(f"Inference Time: {results['inference_time']:.2f}s")
        print(f"Average Speed: {results['total']/results['inference_time']:.2f} samples/s")
        print(f"\nEvaluation Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:6.2f}%")
        print(f"  Precision: {metrics['precision']:6.2f}%")
        print(f"  Recall:    {metrics['recall']:6.2f}%")
        print(f"  F1-Score:  {metrics['f1_score']:6.2f}%")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives  (TP): {results['true_positives']:4d}  |  False Positives (FP): {results['false_positives']:4d}")
        print(f"  False Negatives (FN): {results['false_negatives']:4d}  |  True Negatives  (TN): {results['true_negatives']:4d}")
        print(f"\nParsing Statistics:")
        print(f"  Success: {results['total'] - results['parse_failed']}/{results['total']} ({(results['total'] - results['parse_failed'])/results['total']*100:.1f}%)")
        print(f"  Failed:  {results['parse_failed']}/{results['total']} ({results['parse_failed']/results['total']*100:.1f}%)")
        print(f"{'='*70}\n")
    
    def run_comparison(self, output_file: str = "sentiment_comparison_results.json"):
        """Run complete comparison evaluation"""
        print("\n" + "🚀 " + "="*66 + " 🚀")
        print("  ChnSentiCorp Sentiment Analysis - Pre/Post Fine-tuning Comparison")
        print("🚀 " + "="*66 + " 🚀\n")
        
        all_results = {
            "dataset": "ChnSentiCorp",
            "dataset_path": self.csv_path,
            "total_samples": len(self.df),
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": {}
        }
        
        # Evaluate base model (before fine-tuning)
        base_results = self.evaluate_model(
            self.base_model_path,
            "Base Model (Pre-finetuning)"
        )
        all_results["models"]["base"] = base_results
        
        # Evaluate fine-tuned model
        finetuned_results = self.evaluate_model(
            self.finetuned_model_path,
            "Fine-tuned Model"
        )
        all_results["models"]["finetuned"] = finetuned_results
        
        # Calculate improvement
        comparison = self._compute_comparison(base_results, finetuned_results)
        all_results["comparison"] = comparison
        
        # Print comparison results
        self._print_comparison(comparison)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Complete evaluation results saved to: {output_file}\n")
        
        return all_results
    
    def _compute_comparison(self, base_results: Dict, finetuned_results: Dict) -> Dict:
        """Compute comparison metrics"""
        base_metrics = base_results["metrics"]
        finetuned_metrics = finetuned_results["metrics"]
        
        comparison = {}
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            base_val = base_metrics[metric]
            finetuned_val = finetuned_metrics[metric]
            improvement = finetuned_val - base_val
            improvement_pct = (improvement / base_val * 100) if base_val > 0 else 0
            
            comparison[metric] = {
                "base": base_val,
                "finetuned": finetuned_val,
                "improvement": round(improvement, 2),
                "improvement_percentage": round(improvement_pct, 2)
            }
        
        return comparison
    
    def _print_comparison(self, comparison: Dict):
        """Print comparison results"""
        print("\n" + "🎯 " + "="*66 + " 🎯")
        print("  Performance Comparison: Pre vs Post Fine-tuning")
        print("🎯 " + "="*66 + " 🎯\n")
        
        print(f"{'Metric':<15} {'Pre-FT':>10} {'Post-FT':>10} {'Improve':>10} {'Improve %':>12}")
        print("-" * 70)
        
        metric_names = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-Score"
        }
        
        for metric, data in comparison.items():
            name = metric_names.get(metric, metric)
            base = data["base"]
            finetuned = data["finetuned"]
            improvement = data["improvement"]
            improvement_pct = data["improvement_percentage"]
            
            # Add symbol to indicate direction
            symbol = "↑" if improvement > 0 else ("↓" if improvement < 0 else "→")
            
            print(f"{name:<15} {base:>9.2f}% {finetuned:>9.2f}% {symbol}{abs(improvement):>8.2f}% {improvement_pct:>10.2f}%")
        
        print("-" * 70)
        print()


def main(
    csv_path: str = "data/ChnSentiCorp_test.csv",
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    finetuned_model: str = "saves/qwen2_5-coder-1.5b/freeze/sft",
    max_samples: Optional[int] = None,
    temperature: float = 0.1,
    max_new_tokens: int = 256,
    output_file: str = "sentiment_comparison_results.json",
    device: str = "auto"
):
    """
    ChnSentiCorp Sentiment Analysis Comparison Evaluation Main Function
    
    Args:
        csv_path: Path to test data CSV file
        base_model: Path to base model (before fine-tuning)
        finetuned_model: Path to fine-tuned model
        max_samples: Maximum number of samples to evaluate (None for all)
        temperature: Generation temperature
        max_new_tokens: Maximum number of tokens to generate
        output_file: Path to output results file
        device: Device selection (auto/cuda/cpu)
    
    Examples:
        # Evaluate all data
        python scripts/eval_sentiment_compare.py
        
        # Evaluate 100 samples (quick test)
        python scripts/eval_sentiment_compare.py --max_samples 100
        
        # Specify output file
        python scripts/eval_sentiment_compare.py --output_file my_results.json
        
        # Use specific GPU
        CUDA_VISIBLE_DEVICES=0 python scripts/eval_sentiment_compare.py
        
        # Use multiple GPUs
        CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_sentiment_compare.py
    """
    evaluator = SentimentEvaluator(
        csv_path=csv_path,
        base_model_path=base_model,
        finetuned_model_path=finetuned_model,
        max_samples=max_samples,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        device=device
    )
    
    results = evaluator.run_comparison(output_file=output_file)
    
    return results


if __name__ == "__main__":
    fire.Fire(main)

