"""
BERT-based Email Classifier for PO Detection
Uses transformers library for state-of-the-art text classification
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
import inspect
import importlib

# Check if transformers is available
try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        pipeline
    )
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Install transformers: pip install transformers torch")


class EmailDataset(Dataset):
    """Dataset class for email classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTEmailClassifier:
    """BERT-based classifier for PO email detection."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to saved model, or None to use pre-trained base
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch required. Install with: pip install transformers torch")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "distilbert-base-uncased"  # Smaller, faster than BERT
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.labels = ['NOT_PO', 'PO']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_base_model()
    
    def _initialize_base_model(self):
        """Initialize with pre-trained model."""
        print(f"Loading base model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label={0: 'NOT_PO', 1: 'PO'},
            label2id={'NOT_PO': 0, 'PO': 1}
        )
        self.model.to(self.device)
        
    def load_model(self, model_path: str):
        """Load a trained model from disk."""
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def save_model(self, output_path: str):
        """Save the trained model."""
        os.makedirs(output_path, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"Model saved to: {output_path}")
    
    def prepare_email_text(self, subject: str, body: str, max_body_chars: int = 500) -> str:
        """
        Prepare email text for classification.
        Combines subject and truncated body.
        """
        # Clean body - remove excessive whitespace and HTML remnants
        body_clean = ' '.join(body.split())[:max_body_chars]
        
        # Format: [SUBJECT] subject text [BODY] body text
        text = f"[SUBJECT] {subject} [BODY] {body_clean}"
        return text

    @staticmethod
    def _parse_version(version_str: str) -> Tuple[int, int, int]:
        """Parse version strings like '1.1.0', '1.1.0.dev0' into a comparable tuple."""
        parts = []
        for chunk in version_str.split('.'):
            digits = ""
            for ch in chunk:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits == "":
                parts.append(0)
            else:
                parts.append(int(digits))

        while len(parts) < 3:
            parts.append(0)

        return tuple(parts[:3])

    def _ensure_accelerate_compatible(self):
        """Trainer with PyTorch requires accelerate>=1.1.0."""
        try:
            module_name = "accele" + "rate"
            accelerate = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                "Training requires accelerate>=1.1.0. "
                "Install with: pip install \"accelerate>=1.1.0\" "
                "or pip install -r streamlit_app/requirements.txt"
            ) from exc

        acc_version = getattr(accelerate, '__version__', '0.0.0')
        if self._parse_version(acc_version) < (1, 1, 0):
            raise ImportError(
                f"Found accelerate=={acc_version}, but training requires accelerate>=1.1.0. "
                "Upgrade with: pip install --upgrade \"accelerate>=1.1.0\""
            )
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str] = None,
        val_labels: List[int] = None,
        output_path: str = "models/po_classifier",
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ):
        """
        Train the classifier on labeled email data.
        
        Args:
            train_texts: List of email texts (use prepare_email_text)
            train_labels: List of labels (0=NOT_PO, 1=PO)
            val_texts: Optional validation texts
            val_labels: Optional validation labels
            output_path: Where to save the trained model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        self._ensure_accelerate_compatible()
        print(f"Training on {len(train_texts)} samples...")
        
        # Create datasets
        train_dataset = EmailDataset(train_texts, train_labels, self.tokenizer)
        
        eval_dataset = None
        if val_texts and val_labels:
            eval_dataset = EmailDataset(val_texts, val_labels, self.tokenizer)
        
        # Build kwargs first so we can support both old and new Transformers APIs.
        training_kwargs = {
            "output_dir": output_path,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "logging_dir": f"{output_path}/logs",
            "logging_steps": 10,
            "save_strategy": "epoch",
            "load_best_model_at_end": True if eval_dataset else False,
            "learning_rate": learning_rate,
        }

        eval_value = "epoch" if eval_dataset else "no"
        training_args_params = inspect.signature(TrainingArguments.__init__).parameters
        if "evaluation_strategy" in training_args_params:
            training_kwargs["evaluation_strategy"] = eval_value
        elif "eval_strategy" in training_args_params:
            training_kwargs["eval_strategy"] = eval_value

        # Training arguments
        training_args = TrainingArguments(**training_kwargs)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train
        trainer.train()
        
        # Save model
        self.save_model(output_path)
        
        # Initialize classifier pipeline for inference
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        print("Training complete!")
        return trainer.state.log_history
    
    def predict(self, text: str) -> Dict:
        """
        Predict if an email is PO-related.
        
        Args:
            text: Email text (use prepare_email_text first)
        
        Returns:
            Dict with 'label', 'score', and 'is_po'
        """
        # Use logits directly so score always means PO probability (class 1).
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

        not_po_prob = float(probs[0].item())
        po_prob = float(probs[1].item())
        is_po = po_prob >= 0.5
        label = 'PO' if is_po else 'NOT_PO'

        return {
            'label': label,
            'score': po_prob,
            'is_po': is_po,
            'confidence': 'HIGH' if po_prob > 0.8 else 'MEDIUM' if po_prob > 0.6 else 'LOW',
            'po_score': po_prob,
            'not_po_score': not_po_prob,
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict on multiple emails."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def classify_email(self, subject: str, body: str, attachments: List[str] = None) -> Dict:
        """
        Classify a single email.
        
        Args:
            subject: Email subject
            body: Email body text
            attachments: Optional list of attachment names
        
        Returns:
            Classification result
        """
        # Prepare text
        text = self.prepare_email_text(subject, body)
        
        # Get prediction
        result = self.predict(text)
        
        # Boost confidence if attachments contain PO indicators
        if attachments:
            for att in attachments:
                att_lower = att.lower()
                if 'po' in att_lower or 'purchase' in att_lower or 'order' in att_lower:
                    if result['score'] < 0.9:
                        result['score'] = min(result['score'] + 0.1, 0.99)
                        result['is_po'] = True
                        result['attachment_boost'] = True
        
        return result


class HybridClassifier:
    """
    Hybrid classifier combining BERT with rule-based patterns.
    Falls back to rules if BERT model is not available.
    """
    
    def __init__(self, bert_model_path: str = None):
        self.bert_classifier = None
        self.use_bert = False
        
        # Try to load BERT
        if TRANSFORMERS_AVAILABLE:
            try:
                self.bert_classifier = BERTEmailClassifier(bert_model_path)
                self.use_bert = True
                print("BERT classifier loaded successfully")
            except Exception as e:
                print(f"BERT not available, using rules only: {e}")
        
        # Rule-based patterns (fallback)
        self.po_keywords = [
            'purchase order', 'po#', 'po number', 'p.o.', 'p.o', 
            'order confirmation', 'order acknowledgment',
            'procurement', 'requisition',
        ]
        
        self.po_patterns = [
            r'\bP[O0]\s*#?\s*[:\-]?\s*[A-Z0-9-]*\d[A-Z0-9-]*\b',
            r'MEL\d{4}PO\d+',
            r'[A-Z]{2,4}\d{4}PO\d+',
        ]
        
        self.negative_keywords = [
            'newsletter', 'unsubscribe', 'meeting invite', 'calendar',
            'out of office', 'automatic reply', 'linkedin', 'facebook',
            'promotional', 'advertisement', 'sale', 'discount offer'
        ]
    
    def _rule_based_score(self, subject: str, body: str) -> Tuple[float, str]:
        """Calculate rule-based score."""
        import re
        
        text = f"{subject} {body}".lower()
        score = 0.0
        
        # Check negative keywords first
        for keyword in self.negative_keywords:
            if keyword in text:
                score -= 0.3
        
        # Check positive keywords
        for keyword in self.po_keywords:
            if keyword in text:
                if keyword in subject.lower():
                    score += 0.2
                else:
                    score += 0.1
        
        # Check patterns
        for pattern in self.po_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.3
        
        # Normalize to 0-1 range
        score = max(0, min(1, (score + 1) / 2))
        
        if score > 0.7:
            confidence = 'HIGH'
        elif score > 0.5:
            confidence = 'MEDIUM'
        elif score > 0.3:
            confidence = 'LOW'
        else:
            confidence = 'NOT_PO'
        
        return score, confidence
    
    def classify(self, subject: str, body: str, attachments: List[str] = None) -> Dict:
        """
        Classify email using BERT if available, otherwise rules.
        """
        if self.use_bert and self.bert_classifier:
            result = self.bert_classifier.classify_email(subject, body, attachments)
            result['method'] = 'BERT'
        else:
            score, confidence = self._rule_based_score(subject, body)
            result = {
                'label': 'PO' if score > 0.5 else 'NOT_PO',
                'score': score,
                'is_po': score > 0.5,
                'confidence': confidence,
                'method': 'RULES'
            }
            
            # Check attachments
            if attachments:
                for att in attachments:
                    att_lower = att.lower()
                    if 'po' in att_lower or 'purchase' in att_lower:
                        result['score'] = min(result['score'] + 0.2, 1.0)
                        result['is_po'] = True
        
        return result
    
    def train_bert(self, train_data: List[Dict], output_path: str = "models/po_classifier"):
        """
        Train BERT classifier on labeled data.
        
        Args:
            train_data: List of dicts with 'subject', 'body', 'is_po' keys
            output_path: Where to save model
        """
        if not self.use_bert:
            raise ValueError("BERT not available. Install transformers and torch.")
        
        # Prepare training data
        texts = []
        labels = []
        
        for item in train_data:
            text = self.bert_classifier.prepare_email_text(
                item['subject'], 
                item['body']
            )
            texts.append(text)
            labels.append(1 if item['is_po'] else 0)
        
        # Split into train/val (90/10)
        split_idx = int(len(texts) * 0.9)
        
        train_texts = texts[:split_idx]
        train_labels = labels[:split_idx]
        val_texts = texts[split_idx:]
        val_labels = labels[split_idx:]
        
        # Train
        self.bert_classifier.train(
            train_texts, train_labels,
            val_texts, val_labels,
            output_path=output_path
        )


def create_training_data_template(output_path: str = "training_data.json"):
    """
    Create a template file for training data.
    User can fill in their own examples.
    """
    template = {
        "instructions": "Fill in examples of PO and non-PO emails. Set is_po to true for PO emails.",
        "examples": [
            {
                "subject": "PO#12345 - Order Confirmation",
                "body": "Please find attached purchase order for the following items...",
                "is_po": True
            },
            {
                "subject": "MEL2025PO12345 - Price Sticker Request",
                "body": "Hi, Please refer attached PO and proceed. We need this by...",
                "is_po": True
            },
            {
                "subject": "Weekly Newsletter - February 2026",
                "body": "Check out our latest news and updates...",
                "is_po": False
            },
            {
                "subject": "Meeting Invitation: Project Review",
                "body": "You are invited to attend a meeting on...",
                "is_po": False
            },
            {
                "subject": "RE: Quote Request",
                "body": "Thank you for your inquiry. Please find our quotation...",
                "is_po": False  # Quotation, not PO
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Training data template created: {output_path}")
    print("Add more examples (aim for 50+ each of PO and non-PO)")
    return output_path


if __name__ == "__main__":
    # Test the classifier
    print("Testing BERT Email Classifier...")
    
    if TRANSFORMERS_AVAILABLE:
        # Create hybrid classifier
        classifier = HybridClassifier()
        
        # Test classifications
        test_emails = [
            {
                "subject": "PO#MEL2025PO12345 - Urgent Order",
                "body": "Please process the attached purchase order immediately."
            },
            {
                "subject": "Weekly Team Newsletter",
                "body": "Here are this week's updates from the team..."
            },
            {
                "subject": "KFD1125013-G86108 - MEL2025PO12232 price sticker",
                "body": "Hi, Please refer attached PO and proceed."
            }
        ]
        
        for email in test_emails:
            result = classifier.classify(email["subject"], email["body"])
            print(f"\nSubject: {email['subject'][:50]}...")
            print(f"  Result: {result['label']} ({result['confidence']}) - Score: {result['score']:.2f}")
            print(f"  Method: {result['method']}")
    else:
        print("Install: pip install transformers torch")
