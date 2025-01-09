import torch

def token_to_text(embedding_result,tokenizer):
    embeddings = embedding_result['embeddings']
    input_ids = embedding_result['input_ids']
    attention_mask = embedding_result['attention_mask']
    print("Embeddings shape:", embeddings.shape)  # Output: torch.Size([1, sequence_length, 1024]) for bert-large
    print("Input IDs shape:", input_ids.shape)
    print("Attention Mask shape:", attention_mask.shape)

    # You can get the tokens back from the input IDs:
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return tokens

def get_bert_embedding(text, model, tokenizer, device):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512) # Important: Truncate if needed

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the model's output
    with torch.no_grad(): # Important: Disable gradient calculation for inference
        outputs = model(**inputs)

    # Extract the embeddings. There are several strategies:
    # 1. [CLS] token embedding (often a good general representation)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # 2. Mean of all token embeddings
    # mean_embedding = outputs.last_hidden_state.mean(dim=1)

    # 3. Max-pooling of all token embeddings
    # max_embedding = torch.max(outputs.last_hidden_state, dim=1).values

    return {
        'embeddings': cls_embedding,
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }