

import pandas as pd
import spacy
import json
import random
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

class FinancialNERTrainer:
    def __init__(self):
        self.nlp = None
        self.ner = None
        self.entity_labels = set()
    
    def load_data_from_csv(self, file_path):
        """Load training data from CSV with JSON-formatted entities"""
        df = pd.read_csv(file_path)
        train_data = []
        
        for _, row in df.iterrows():
            text = row['text']
            if pd.isna(text):
                continue
                
            # Parse JSON entities
            entities = []
            if not pd.isna(row['entities']):
                try:
                    # Ensure the string is valid JSON
                    entities_str = row['entities'].replace("'", '"')
                    entities_list = json.loads(entities_str)
                    for entity in entities_list:
                        # Validate entity format
                        if isinstance(entity, dict) and 'start' in entity and 'end' in entity and 'label' in entity:
                            entities.append((entity['start'], entity['end'], entity['label']))
                            self.entity_labels.add(entity['label'])
                        else:
                             print(f"Skipping invalid entity format: {entity} for text: {text}")
                except json.JSONDecodeError as e:
                    print(f"Error parsing entities for text: {text} - {e}")
                    continue
                except Exception as e:
                    print(f"Unexpected error processing entities for text: {text} - {e}")
                    continue

            train_data.append((text, {'entities': entities}))
        
        print(f"Loaded {len(train_data)} training samples")
        print(f"Detected entity labels: {self.entity_labels}")
        return train_data

    def initialize_model(self):
        """Create a blank model with enhanced financial patterns"""
        self.nlp = spacy.blank('en')
        
        # Add entity ruler for financial patterns
        # Ensure 'entity_ruler' is added before 'ner' if 'ner' exists
        if 'ner' in self.nlp.pipe_names:
             ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        else:
             ruler = self.nlp.add_pipe("entity_ruler")


        patterns = [
            # Money patterns
            {"label": "MONEY", "pattern": [{"TEXT": {"REGEX": "^\$[\d,]+(\.\d+)?[MBT]?$"}}]},
            {"label": "MONEY", "pattern": [{"TEXT": {"REGEX": "^[\d,]+(\.\d+)?[MBT]? dollars$"}}]},
            
            # Percent patterns
            {"label": "PERCENT", "pattern": [{"TEXT": {"REGEX": "^[\d\.]+%?$"}}]}, # Allow % optionally
            
            # Date/Quarter patterns
            {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "^Q[1-4] 20\d{2}$"}}]},
             {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "^(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}$"}}]}, # Full date
              {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "^\d{1,2}/\d{1,2}/\d{2,4}$"}}]}, # mm/dd/yy(yy) date
            
            # Common financial terms
            {"label": "EVENT", "pattern": [{"LOWER": {"IN": ["earnings", "dividend", "report", "call"]}}]},
            {"label": "ORG", "pattern": [{"TEXT": {"REGEX": "^[A-Z]{2,5}$"}}]}, # Potential stock tickers
        ]
        ruler.add_patterns(patterns)
        
        # Add NER pipeline if it doesn't exist
        if 'ner' not in self.nlp.pipe_names:
             self.ner = self.nlp.add_pipe('ner')
        else:
             self.ner = self.nlp.get_pipe('ner')

        
        # Add all detected labels
        for label in self.entity_labels:
            if label not in self.ner.labels:
                 self.ner.add_label(label)

    def train_model(self, train_data, n_iter=50, dropout=0.4):
        """Enhanced training process with progress tracking"""
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner' and pipe != 'entity_ruler']
        with self.nlp.disable_pipes(*other_pipes):
            if self.nlp.has_pipe("ner"):
                optimizer = self.nlp.begin_training()
            else:
                print("NER pipe not found. Cannot begin training.")
                return

            print("Training the model...")
            for itn in range(n_iter):
                random.shuffle(train_data)
                losses = {}
                
                # Adjust batching for potentially small datasets
                batch_size = compounding(4.0, 32.0, 1.001) if len(train_data) > 32 else len(train_data)
                
                batches = minibatch(train_data, size=batch_size)
                
                for batch in tqdm(batches, desc=f"Iteration {itn + 1}/{n_iter}"):
                    examples = []
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        try:
                            examples.append(Example.from_dict(doc, annotations))
                        except ValueError as e:
                            print(f"Skipping invalid example: {text} - {e}")
                            continue

                    if examples: # Only update if there are valid examples
                        self.nlp.update(
                            examples,
                            drop=dropout,
                            losses=losses,
                            sgd=optimizer,
                            annotates=["entities"]
                        )
                
                # Print loss only if NER loss is available
                if 'ner' in losses:
                    print(f"Loss: {losses['ner']:.4f}")
                else:
                     print("No NER loss reported (possibly due to no valid training examples in batch).")


    def save_model(self, output_dir):
        """Save model with versioning"""
        version = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = Path(f"{output_dir}_{version}")
        try:
            self.nlp.to_disk(output_path)
            print(f"Model saved to {output_path}")
        except Exception as e:
            print(f"Error saving model to {output_path}: {e}")

    def evaluate_sample(self, text):
        """Test the model on sample text"""
        if self.nlp is None:
            print("Model not initialized or trained.")
            return []

        doc = self.nlp(text)
        results = []
        for ent in doc.ents:
            results.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return results

# Usage Example
if __name__ == "__main__":
    # Initialize trainer
    trainer = FinancialNERTrainer()
    
    # Load your dataset
    # IMPORTANT: Replace with the actual path to your CSV file
    # You might need to upload the file to Colab's temporary storage or Google Drive
    try:
        train_data = trainer.load_data_from_csv("/content/financial_ner_dataset.csv")
        
        if train_data:
            # Initialize and train model
            trainer.initialize_model()
            trainer.train_model(train_data, n_iter=50)
            
            # Save the trained model
            trainer.save_model("financial_ner_model")
            
            # Test with sample text
            test_text = "Apple announced $150 billion revenue for Q3 2023. Stock price was $170.50. The next earnings call is in January 2024."
            print("\nTest Prediction:")
            prediction = trainer.evaluate_sample(test_text)
            print(prediction)

            # Example of loading the saved model (optional)
            # from spacy import load
            # latest_model_path = sorted(Path(".").glob("financial_ner_model_*"))[-1] # Get the latest saved model
            # loaded_nlp = load(latest_model_path)
            # loaded_doc = loaded_nlp(test_text)
            # print("\nPrediction with loaded model:")
            # for ent in loaded_doc.ents:
            #      print({"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char})

        else:
            print("No valid training data loaded. Cannot train the model.")

    except FileNotFoundError:
        print("Error: /content/financial_ner_dataset.csv not found.")
        print("Please upload your CSV file to the Colab environment or update the path.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")

