# pinhas_model/joint_model.py
import torch.nn as nn
from transformers import AutoModel


class JointTransformer(nn.Module):
    def __init__(self, model_name, num_intents, num_entities):
        super(JointTransformer, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size

        # Cabeças (Heads)
        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        self.entity_classifier = nn.Linear(hidden_size, num_entities)

    def forward(self, input_ids, attention_mask, intent_labels=None, entity_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # Token [CLS]

        # Logits
        intent_logits = self.intent_classifier(self.dropout(pooled_output))
        entity_logits = self.entity_classifier(self.dropout(sequence_output))

        loss = None
        if intent_labels is not None:
            import torch
            loss_fct = nn.CrossEntropyLoss()
            loss_intent = loss_fct(intent_logits, intent_labels)

            loss_fct_ner = nn.CrossEntropyLoss(ignore_index=-100)
            loss_ner = loss_fct_ner(
                entity_logits.view(-1, entity_logits.shape[-1]),
                entity_labels.view(-1),
            )

            # Loss Combinada
            loss = loss_intent + loss_ner

        return {"loss": loss, "intent_logits": intent_logits, "entity_logits": entity_logits}
