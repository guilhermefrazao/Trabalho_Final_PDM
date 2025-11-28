import json
import time
import os
import sys
import torch
from torch.utils.data import  Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from seqeval.metrics import f1_score as ner_f1_score
from seqeval.metrics import classification_report as ner_classification_report
from seqeval.scheme import IOB2
from tqdm.auto import tqdm


# NOTA: Não importamos torch, transformers ou sklearn aqui no topo!
# Isso evita o Timeout do DagBag no Airflow.

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


def load_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo '{filepath}' não encontrado.")

def run_training_pipeline(epochs=1):
    # ==========================================================================
    # IMPORTS TARDIOS (LAZY IMPORTS) - O Segredo para o Airflow não travar
    # ==========================================================================

    
    # Importa as classes do arquivo vizinho que criamos
    # Adiciona o diretório atual ao path para garantir que ele ache o joint_model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from joint_model import JointTransformer

    # ==========================================================================
    # CONFIGURAÇÕES
    # ==========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_FILE = os.path.join(BASE_DIR, 'data', 'dataset_v3_train.json')
    VAL_FILE = os.path.join(BASE_DIR, 'data', 'dataset_v3_val.json')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'modelo_treinado_v3')
    MODEL_NAME = "microsoft/mdeberta-v3-base"

    # ==========================================================================
    # HELPER FUNCTIONS (Internas para ter acesso ao tokenizer/torch)
    # ==========================================================================
    def process_data(raw_list, tokenizer, intent2id, tag2id):
        processed = []
        for item in raw_list:
            text = item['text']
            intent = item['intent']
            entities = item.get('entities', [])
            
            tokenized = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)
            offset_mapping = tokenized['offset_mapping']
            labels = ['O'] * len(tokenized['input_ids'])
            
            for ent in entities:
                try:
                    start, end = ent['start'], ent['end']
                    label_name = ent['entity']
                    found = False
                    for idx, (os, oe) in enumerate(offset_mapping):
                        if os == 0 and oe == 0: continue
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

    def train_and_evaluate(model, train_loader, val_loader, epochs):
        optimizer = AdamW(model.parameters(), lr=2e-5)
        print(f"\n--- 🚀 INICIANDO TREINAMENTO ({epochs} épocas) ---")
        
        for epoch in range(epochs):
            model.train()
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        # Avaliação
        model.eval()
        intent_preds, intent_true = [], []
        ner_preds, ner_true = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                
                batch_int_preds = torch.argmax(outputs['intent_logits'], dim=1).cpu().numpy()
                intent_preds.extend(batch_int_preds)
                intent_true.extend(batch['intent_labels'].cpu().numpy())
                
                batch_ner_logits = torch.argmax(outputs['entity_logits'], dim=2)
                for i in range(len(batch['input_ids'])):
                    label_ids = batch['entity_labels'][i].cpu().numpy()
                    pred_ids = batch_ner_logits[i].cpu().numpy()
                    
                    lab_list = [id2tag[lab] for lab in label_ids if lab != -100]
                    pred_list = [id2tag[pred] for lab, pred in zip(label_ids, pred_ids) if lab != -100]
                    
                    ner_true.append(lab_list)
                    ner_preds.append(pred_list)

        acc = accuracy_score(intent_true, intent_preds)
        f1_int = f1_score(intent_true, intent_preds, average='weighted')
        f1_ner = ner_f1_score(ner_true, ner_preds, mode='strict', scheme=IOB2)
        return acc, f1_int, f1_ner

    def save_model_complete(model, tokenizer, output_dir, intent2id, tag2id):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        torch.save(model.state_dict(), f"{output_dir}/model_weights.bin")
        tokenizer.save_pretrained(output_dir)
        config_data = {
            "base_model_name": MODEL_NAME, "intent2id": intent2id, "tag2id": tag2id,
            "num_intents": len(intent2id), "num_entities": len(tag2id)
        }
        with open(f"{output_dir}/training_config.json", "w", encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

    # ==========================================================================
    # EXECUÇÃO DO PIPELINE
    # ==========================================================================
    print("\n================ PIPELINE DE TREINO INICIADO ================")
    
    print(f"Carregando Tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\n📂 Carregando datasets...")
    raw_train_data = load_file(TRAIN_FILE)
    raw_val_data = load_file(VAL_FILE)

    print("\n🧠 Criando mapas de classes...")
    all_data = raw_train_data + raw_val_data
    unique_intents = sorted(set(d['intent'] for d in all_data))
    unique_tags = {'O'}
    for item in all_data:
        for ent in item.get('entities', []):
            unique_tags.add(f"B-{ent['entity']}")
            unique_tags.add(f"I-{ent['entity']}")
    
    intent2id = {k: v for v, k in enumerate(unique_intents)}
    tag2id = {k: v for v, k in enumerate(sorted(unique_tags))}
    id2tag = {v: k for k, v in tag2id.items()}

    print("\n⚙️ Processando dados...")
    train_processed = process_data(raw_train_data, tokenizer, intent2id, tag2id)
    val_processed = process_data(raw_val_data, tokenizer, intent2id, tag2id)

    train_loader = DataLoader(JointDataset(train_processed), batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(JointDataset(val_processed), batch_size=16, collate_fn=collate_fn)

    print("\n🤖 Inicializando modelo...")
    model = JointTransformer(MODEL_NAME, len(intent2id), len(tag2id)).to(device)

    print("\n🔥 Iniciando treinamento...")
    acc, f1_int, f1_ner = train_and_evaluate(model, train_loader, val_loader, epochs)

    print("\n💾 Salvando artefatos...")
    save_model_complete(model, tokenizer, OUTPUT_DIR, intent2id, tag2id)

    print("\n✅ PIPELINE FINALIZADO COM SUCESSO!")
    
    # Retorna o que você precisar para o MLflow
    return model, tokenizer, acc, f1_int, f1_ner

if __name__ == "__main__":
    # Teste local apenas
    run_training_pipeline(epochs=1)