# ==============================================================================
# 1. INSTALAÇÃO DE DEPENDÊNCIAS
# ==============================================================================
import json
import time
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from seqeval.metrics import f1_score as ner_f1_score
from seqeval.metrics import classification_report as ner_classification_report
from seqeval.scheme import IOB2
from tqdm.auto import tqdm

# Configuração de Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ==============================================================================
# 2. CONFIGURAÇÃO E CARREGAMENTO
# ==============================================================================
TRAIN_FILE = 'dags/pinhas_model/data/dataset_v3_train.json'
VAL_FILE = 'dags/pinhas_model/data/dataset_v3_val.json'
OUTPUT_DIR = "./dags/pinhas_model/models/modelo_treinado_v3"

# Para comparar com BERTimbau, mude para: "neuralmind/bert-base-portuguese-cased"
MODEL_NAME = "microsoft/mdeberta-v3-base" 

print(f"Carregando Tokenizer: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    # Fallback para sentencepiece se necessário
    print("Instalando sentencepiece se necessário...")
    os.system("pip install sentencepiece")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo '{filepath}' não encontrado. Rode o script de geração (generate_v4.py) primeiro.")

print("Carregando datasets...")
raw_train_data = load_file(TRAIN_FILE)
raw_val_data = load_file(VAL_FILE)

print(f"Exemplos de Treino: {len(raw_train_data)}")
print(f"Exemplos de Validação: {len(raw_val_data)}")

# --- CRIAÇÃO DOS MAPAS DE LABELS ---
# Combinamos os dados apenas para extrair a lista completa de classes possíveis
all_data = raw_train_data + raw_val_data

unique_intents = sorted(list(set(d['intent'] for d in all_data)))
unique_tags = set(['O'])
for item in all_data:
    for ent in item.get('entities', []):
        unique_tags.add(f"B-{ent['entity']}")
        unique_tags.add(f"I-{ent['entity']}")
unique_tags = sorted(list(unique_tags))

# Mapas ID <-> Texto
intent2id = {k: v for v, k in enumerate(unique_intents)}
id2intent = {v: k for k, v in intent2id.items()}
tag2id = {k: v for v, k in enumerate(unique_tags)}
id2tag = {v: k for k, v in tag2id.items()}

print(f"\nResumo das Classes:")
print(f"- Intenções ({len(intent2id)}): {intent2id}")
print(f"- Tags NER ({len(tag2id)}): {tag2id}")

# ==============================================================================
# 3. PROCESSAMENTO E DATASET
# ==============================================================================

def process_data(raw_list, tokenizer, intent2id, tag2id):
    processed = []
    for item in raw_list:
        text = item['text']
        intent = item['intent']
        entities = item.get('entities', [])
        
        # Tokenização com offsets para alinhar labels
        tokenized = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)
        offset_mapping = tokenized['offset_mapping']
        
        labels = ['O'] * len(tokenized['input_ids'])
        
        for ent in entities:
            try:
                start, end = ent['start'], ent['end']
                label_name = ent['entity']
                found = False
                for idx, (os, oe) in enumerate(offset_mapping):
                    if os == 0 and oe == 0: continue # Skip special tokens
                    # Se o token está dentro da entidade
                    if oe > start and os < end:
                        labels[idx] = f"B-{label_name}" if not found else f"I-{label_name}"
                        found = True
            except: continue
            
        processed.append({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'intent_label': intent2id[intent],
            'labels': [tag2id[l] for l in labels]
        })
    return processed

# Processamento
print("Processando dados...")
train_processed = process_data(raw_train_data, tokenizer, intent2id, tag2id)
val_processed = process_data(raw_val_data, tokenizer, intent2id, tag2id)

class JointDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'intent_label': torch.tensor(item['intent_label'], dtype=torch.long),
            'entity_labels': torch.tensor(item['labels'], dtype=torch.long)
        }

def collate_fn(batch):
    # Padding Dinâmico
    input_ids = [x['input_ids'] for x in batch]
    attention_mask = [x['attention_mask'] for x in batch]
    intent_labels = torch.stack([x['intent_label'] for x in batch])
    entity_labels = [x['entity_labels'] for x in batch]
    
    max_len = max(len(x) for x in input_ids)
    
    input_ids_pad = torch.zeros(len(input_ids), max_len, dtype=torch.long)
    mask_pad = torch.zeros(len(attention_mask), max_len, dtype=torch.long)
    entity_pad = torch.ones(len(entity_labels), max_len, dtype=torch.long) * -100 # -100 ignora loss
    
    for i in range(len(input_ids)):
        l = len(input_ids[i])
        input_ids_pad[i, :l] = input_ids[i]
        mask_pad[i, :l] = attention_mask[i]
        entity_pad[i, :l] = entity_labels[i]
        
    return {'input_ids': input_ids_pad, 'attention_mask': mask_pad, 
            'intent_labels': intent_labels, 'entity_labels': entity_pad}

# Loaders
train_loader = DataLoader(JointDataset(train_processed), batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(JointDataset(val_processed), batch_size=32, collate_fn=collate_fn)

# ==============================================================================
# 4. ARQUITETURA DO MODELO (JOINT LEARNING)
# ==============================================================================
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
        pooled_output = sequence_output[:, 0, :] # Token [CLS]
        
        # Logits
        intent_logits = self.intent_classifier(self.dropout(pooled_output))
        entity_logits = self.entity_classifier(self.dropout(sequence_output))
        
        loss = None
        if intent_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_intent = loss_fct(intent_logits, intent_labels)
            
            loss_fct_ner = nn.CrossEntropyLoss(ignore_index=-100)
            loss_ner = loss_fct_ner(entity_logits.view(-1, entity_logits.shape[-1]), entity_labels.view(-1))
            
            # Loss Combinada
            loss = loss_intent + loss_ner
            
        return {'loss': loss, 'intent_logits': intent_logits, 'entity_logits': entity_logits}

model = JointTransformer(MODEL_NAME, len(intent2id), len(tag2id)).to(device)

# ==============================================================================
# 5. TREINAMENTO
# ==============================================================================
def train_and_evaluate(model, train_loader, val_loader, epochs=5):
    # Learning Rate ajustado
    optimizer = AdamW(model.parameters(), lr=2e-5) 
    
    print(f"\n--- 🚀 INICIANDO TREINAMENTO ({epochs} épocas) ---")
    start_train = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
    
    print(f"✅ Treino Finalizado em {time.time() - start_train:.2f}s")
    
    # --- AVALIAÇÃO ---
    print("\n--- 📊 CALCULANDO MÉTRICAS (Dados de Validação Limpos) ---")
    model.eval()
    
    intent_preds, intent_true = [], []
    ner_preds, ner_true = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            # Intenção
            batch_int_preds = torch.argmax(outputs['intent_logits'], dim=1).cpu().numpy()
            intent_preds.extend(batch_int_preds)
            intent_true.extend(batch['intent_labels'].cpu().numpy())
            
            # Entidades
            batch_ner_logits = torch.argmax(outputs['entity_logits'], dim=2)
            for i in range(len(batch['input_ids'])):
                label_ids = batch['entity_labels'][i].cpu().numpy()
                pred_ids = batch_ner_logits[i].cpu().numpy()
                
                lab_list, pred_list = [], []
                for lab, pred in zip(label_ids, pred_ids):
                    if lab != -100:
                        lab_list.append(id2tag[lab])
                        pred_list.append(id2tag[pred])
                ner_true.append(lab_list)
                ner_preds.append(pred_list)

    # Métricas
    print("\n" + "="*50)
    print(f"RELATÓRIO FINAL: {MODEL_NAME}")
    print("="*50)
    
    acc = accuracy_score(intent_true, intent_preds)
    f1_int = f1_score(intent_true, intent_preds, average='weighted')
    f1_ner = ner_f1_score(ner_true, ner_preds, mode='strict', scheme=IOB2)
    
    print(f"Intenção Acurácia: {acc:.4f}")
    print(f"Intenção F1: {f1_int:.4f}")
    print(f"Entidade F1 (Span): {f1_ner:.4f}")
    print("\nDETALHES DE ENTIDADES:")
    print(ner_classification_report(ner_true, ner_preds, mode='strict', scheme=IOB2))

    report = ner_classification_report(ner_true, ner_preds, mode='strict', scheme=IOB2)

    return acc, f1_int, f1_ner

# Executa Treino
train_and_evaluate(model, train_loader, val_loader, epochs=5)

# ==============================================================================
# 6. SALVAMENTO COMPLETO
# ==============================================================================
def save_model_complete(model, tokenizer, output_dir, intent2id, tag2id, model_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n💾 Salvando modelo em '{output_dir}'...")

    # 1. Salvar os pesos
    torch.save(model.state_dict(), f"{output_dir}/model_weights.bin")

    # 2. Salvar o Tokenizer
    tokenizer.save_pretrained(output_dir)

    # 3. Salvar configurações
    config_data = {
        "base_model_name": model_name,
        "intent2id": intent2id,
        "tag2id": tag2id,
        "num_intents": len(intent2id),
        "num_entities": len(tag2id)
    }
    
    with open(f"{output_dir}/training_config.json", "w", encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
        
    print("✅ Modelo e configurações salvos com sucesso!")

# Salva
save_model_complete(model, tokenizer, OUTPUT_DIR, intent2id, tag2id, MODEL_NAME)

# ==============================================================================
# 7. TESTE RÁPIDO (Playground)
# ==============================================================================
def predict_playground(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    
    intent = id2intent[torch.argmax(outputs['intent_logits']).item()]
    entity_ids = torch.argmax(outputs['entity_logits'], dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    print(f"\nFrase: '{text}'")
    print(f"Intenção: {intent}")
    print("Entidades:", end=" ")
    
    current_ent = None
    for token, idx in zip(tokens, entity_ids):
        if token in tokenizer.all_special_tokens: continue
        label = id2tag[idx]
        token_clean = token.replace(' ', ' ').replace('##', '').strip()
        
        if label.startswith("B-"):
            if current_ent: print(f"[{current_ent['tag']}: {current_ent['val']}]", end=" ")
            current_ent = {'tag': label[2:], 'val': token_clean}
        elif label.startswith("I-") and current_ent and label[2:] == current_ent['tag']:
            current_ent['val'] += token_clean
        else:
            if current_ent: print(f"[{current_ent['tag']}: {current_ent['val']}]", end=" "); current_ent = None
    if current_ent: print(f"[{current_ent['tag']}: {current_ent['val']}]", end=" ")
    print()

print("\n--- TESTE MANUAL ---")
predict_playground("Qual o elenco de Matrix?")
predict_playground("Quero ver filmes lançados em 2022")


def run_training_pipeline(
    train_file=TRAIN_FILE,
    val_file=VAL_FILE,
    output_dir=OUTPUT_DIR,
    model_name=MODEL_NAME,
    epochs=5
):
    print("\n================ PIPELINE DE TREINO INICIADO ================")

    # 1. Carregar dados
    print("\n📂 Carregando datasets...")
    raw_train_data = load_file(train_file)
    raw_val_data = load_file(val_file)

    print(f"Treino: {len(raw_train_data)} | Validação: {len(raw_val_data)}")

    # 2. Criar mapas de labels
    print("\n🧠 Criando mapas de classes...")
    all_data = raw_train_data + raw_val_data

    unique_intents = sorted(set(d['intent'] for d in all_data))
    unique_tags = {'O'}
    for item in all_data:
        for ent in item.get('entities', []):
            unique_tags.add(f"B-{ent['entity']}")
            unique_tags.add(f"I-{ent['entity']}")
    unique_tags = sorted(unique_tags)

    intent2id = {k: v for v, k in enumerate(unique_intents)}
    id2intent = {v: k for k, v in intent2id.items()}
    tag2id = {k: v for v, k in enumerate(unique_tags)}
    id2tag = {v: k for k, v in tag2id.items()}

    print(f"Intents: {len(intent2id)} | Tags NER: {len(tag2id)}")

    # 3. Processamento
    print("\n⚙️ Processando dados...")
    train_processed = process_data(raw_train_data, tokenizer, intent2id, tag2id)
    val_processed = process_data(raw_val_data, tokenizer, intent2id, tag2id)

    train_loader = DataLoader(
        JointDataset(train_processed), batch_size=16, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        JointDataset(val_processed), batch_size=32, collate_fn=collate_fn
    )

    # 4. Criar modelo
    print("\n🤖 Inicializando modelo...")
    model = JointTransformer(model_name, len(intent2id), len(tag2id)).to(device)

    # 5. Treinar
    print("\n🔥 Iniciando treinamento...")
    acc, f1_int, f1_ner = train_and_evaluate(model, train_loader, val_loader, epochs=epochs)

    # 6. Salvar resultado
    print("\n💾 Salvando artefatos...")
    save_model_complete(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        intent2id=intent2id,
        tag2id=tag2id,
        model_name=model_name
    )

    print("\n✅ PIPELINE FINALIZADO COM SUCESSO!")
    return model, acc, f1_int, f1_ner


if __name__ == "__main__":
    model, tokenizer, intent2id, tag2id, id2intent, id2tag = run_training_pipeline(epochs=5)
