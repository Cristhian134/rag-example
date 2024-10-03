from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from config import MODEL_EMBEDDING_NAME, VECTORDB_PATH, DOCUMENTS_PATH

# Lista de modelos para embedding en HF hub: https://huggingface.co/spaces/mteb/leaderboard
Settings.embed_model = HuggingFaceEmbedding(model_name=MODEL_EMBEDDING_NAME)
Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25

documents = SimpleDirectoryReader(DOCUMENTS_PATH).load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)

persist_dir = VECTORDB_PATH
index.storage_context.persist(persist_dir=persist_dir)