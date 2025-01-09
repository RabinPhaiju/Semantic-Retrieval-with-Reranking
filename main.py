import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import token_to_text,get_bert_embedding
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator

# model_name = "google-bert/bert-base-uncased"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
device = torch.device('mps')
model.to(device)

sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Example usage
text1 = "The quick brown fox jumps over the lazy dog."
text2 = "A swift brown fox leaps over the indolent canine."

# using sentence-transformers
# embeddings1 = sentence_model.encode(text1)
# embeddings2 = sentence_model.encode(text2)
# similarity_AB = cosine_similarity(embeddings1.reshape(1, -1), embeddings2.reshape(1, -1))[0][0]
# print(f"Similarity between A and B: {similarity_AB}")

# using bert
# embedding1 = get_bert_embedding(text1, model, tokenizer, device)
# embedding2 = get_bert_embedding(text2, model, tokenizer, device)
# print(embedding['embeddings'].shape)
# similarity = cosine_similarity(embedding1['embeddings'].cpu().numpy(), embedding2['embeddings'].cpu().numpy())
# print(f"Similarity bert: {similarity}")


# bert-token to text
# print(token_to_text(embedding_long_text, tokenizer))
