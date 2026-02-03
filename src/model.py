import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import os

def load_transformer_model(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    print(f"Loading {model_name} from ModelScope...")
    from modelscope import snapshot_download
    model_dir = snapshot_download(model_name, cache_dir='./model_cache')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    return tokenizer, model

class DualHeadEmotionClassifier(nn.Module):
    def __init__(self, num_primary_classes, num_secondary_classes, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        super(DualHeadEmotionClassifier, self).__init__()
        
        self.tokenizer, self.encoder = load_transformer_model(model_name)
        self.embedding_dim = self.encoder.config.hidden_size
        
        self.primary_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_primary_classes)
        )
        
        self.secondary_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_secondary_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for module in [self.primary_head, self.secondary_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs, attention_mask)
        
        primary_logits = self.primary_head(embeddings)
        secondary_logits = self.secondary_head(embeddings)
        return primary_logits, secondary_logits
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen")
            
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen")
    
    def get_optimizer_params(self, encoder_lr=2e-5, head_lr=1e-3, weight_decay=0.01):
        encoder_params = list(self.encoder.named_parameters())
        head_params = list(self.primary_head.named_parameters()) + list(self.secondary_head.named_parameters())
        
        optimizer_params = [
            {'params': [p for n, p in encoder_params if p.requires_grad], 'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in head_params], 'lr': head_lr, 'weight_decay': weight_decay}
        ]
        return optimizer_params
    
    def predict(self, texts, device='cpu'):
        self.eval()
        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            primary_logits, secondary_logits = self.forward(input_ids, attention_mask)
            
            primary_probs = torch.softmax(primary_logits, dim=-1)
            secondary_probs = torch.softmax(secondary_logits, dim=-1)
            
            primary_preds = torch.argmax(primary_probs, dim=-1)
            secondary_preds = torch.argmax(secondary_probs, dim=-1)
            
            primary_confs = primary_probs.max(dim=-1).values
            secondary_confs = secondary_probs.max(dim=-1).values
            
        return {
            'primary_preds': primary_preds,
            'secondary_preds': secondary_preds,
            'primary_confs': primary_confs,
            'secondary_confs': secondary_confs,
            'primary_probs': primary_probs,
            'secondary_probs': secondary_probs
        }


class MultiTaskLoss(nn.Module):
    def __init__(self, primary_weight=1.0, secondary_weight=1.0, primary_class_weights=None, secondary_class_weights=None):
        super(MultiTaskLoss, self).__init__()
        self.primary_weight = primary_weight
        self.secondary_weight = secondary_weight
        
        if primary_class_weights is not None:
            self.primary_criterion = nn.CrossEntropyLoss(weight=torch.tensor(primary_class_weights, dtype=torch.float32), label_smoothing=0.1)
        else:
            self.primary_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            
        if secondary_class_weights is not None:
            self.secondary_criterion = nn.CrossEntropyLoss(weight=torch.tensor(secondary_class_weights, dtype=torch.float32), label_smoothing=0.1)
        else:
            self.secondary_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def to(self, device):
        self.primary_criterion = self.primary_criterion.to(device)
        self.secondary_criterion = self.secondary_criterion.to(device)
        return self
            
    def forward(self, primary_logits, secondary_logits, primary_labels, secondary_labels):
        primary_loss = self.primary_criterion(primary_logits, primary_labels)
        secondary_loss = self.secondary_criterion(secondary_logits, secondary_labels)
        total_loss = self.primary_weight * primary_loss + self.secondary_weight * secondary_loss
        return total_loss, primary_loss, secondary_loss
