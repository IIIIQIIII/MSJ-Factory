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
Convert ChnSentiCorp CSV format to LlamaFactory standard JSONL format

This script converts the ChnSentiCorp dataset from CSV format to the
JSONL format used by LlamaFactory for training and evaluation.
"""
import json
from pathlib import Path

import fire
import pandas as pd


def create_sentiment_prompt(text: str) -> str:
    """
    Create sentiment analysis prompt matching the training format
    
    Args:
        text: Input text for sentiment analysis
        
    Returns:
        Formatted prompt string
    """
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


def convert_csv_to_jsonl(
    csv_path: str = "data/ChnSentiCorp_test.csv",
    output_path: str = "data/chnsenticorp_test.jsonl",
    include_label: bool = True
):
    """
    Convert CSV to JSONL format
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output JSONL file
        include_label: Whether to include labels (for evaluation)
    """
    print(f"📁 Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset size: {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Convert to JSONL format
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Create user message
            user_content = create_sentiment_prompt(row['text_a'])
            messages = [{"role": "user", "content": user_content}]
            
            # Add assistant response (contains correct answer for evaluation)
            if include_label:
                assistant_response = f"""```json
{{
  "sentiment": {int(row['label'])}
}}
```"""
                messages.append({"role": "assistant", "content": assistant_response})
            
            # Create entry
            entry = {"messages": messages}
            
            # Add metadata
            if 'qid' in row:
                entry["qid"] = int(row['qid'])
            if include_label and 'label' in row:
                entry["label"] = int(row['label'])
            
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Conversion completed: {output_path}")
    print(f"Total: {len(df)} samples")
    
    return output_path


def main(
    csv_path: str = "data/ChnSentiCorp_test.csv",
    output_path: str = "data/chnsenticorp_test.jsonl",
    include_label: bool = True
):
    """
    Main function to convert ChnSentiCorp dataset
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output JSONL file
        include_label: Whether to include labels in output
    
    Examples:
        # Convert test set with labels
        python scripts/convert_chnsenticorp.py
        
        # Convert with custom paths
        python scripts/convert_chnsenticorp.py \
            --csv_path data/ChnSentiCorp_test.csv \
            --output_path data/chnsenticorp_test.jsonl
        
        # Convert without labels (for inference only)
        python scripts/convert_chnsenticorp.py --include_label False
    """
    convert_csv_to_jsonl(csv_path, output_path, include_label)


if __name__ == "__main__":
    fire.Fire(main)

