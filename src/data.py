from pathlib import Path
import json
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent

class Task2Dataset:
    def __init__(self, data_dir: str, label_path: str = None, ):
        self.data_dir = BASE_DIR / data_dir
        self.label_path = BASE_DIR / label_path if label_path else None
        self.labels = self.load_labels()
        self.samples = self.load_samples()

    def load_labels(self) -> Dict[str, List[str]]:
        with open(self.label_path, "r", encoding = "utf-8") as f:
            return json.load(f)

    def load_samples(self) -> List[Dict]:
        samples = []
        for case_dir in sorted(self.data_dir.iterdir()):
            if case_dir.is_dir():
                case_id = case_dir.name
                base_case = (case_dir / "base_case.txt").read_text(encoding = "utf-8")
                entailed_fragment = (case_dir / "entailed_fragment.txt").read_text(encoding = "utf-8")
                para_dir = case_dir / "paragraphs"
                paragraphs = []
                for para_file in para_dir.iterdir():
                    para_id = para_file.name
                    paragraph = para_file.read_text(encoding = "utf-8")
                    if case_id in self.labels:
                        label = 1 if para_id in self.labels[case_id] else 0
                    paragraphs.append({
                        "para_id": para_id,
                        "paragraph": paragraph,
                        "label": label
                    })      
                samples.append({
                    "case_id": case_id,
                    "entailed_fragment": entailed_fragment,
                    "base_case": base_case[:100],
                    "paragraphs": paragraphs
                })
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# def main():
#     dataset = Task2Dataset(
#         data_dir = "Data/task2_train_files_2025",
#         label_path = "Data/task2_train_labels_2025.json"
#     )
#     for p in dataset[0]['paragraphs']:
#         print(p['para_id'], p['label'])
# if __name__ == "__main__":
#     main()
    

                
        