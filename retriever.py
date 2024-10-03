from config import VECTORDB_PATH, MODEL_EMBEDDING_NAME
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine

class MyRetriever():

  query_engine = None
  top_k = 3

  def __init__(self):

    Settings.embed_model = HuggingFaceEmbedding(model_name=MODEL_EMBEDDING_NAME)
    Settings.llm = None
    Settings.chunk_size = 256
    Settings.chunk_overlap = 25

    index = self.load_index()

    # set number of docs to retreive
    top_k = 3

    # configure retriever
    retriever = VectorIndexRetriever(
      index=index,
      similarity_top_k=top_k,
    )

    # assemble query engine
    self.query_engine = RetrieverQueryEngine(
      retriever=retriever,
      node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )

  def load_index(self):
    storage_context = StorageContext.from_defaults(
      docstore=SimpleDocumentStore.from_persist_dir(persist_dir=VECTORDB_PATH),
      vector_store=SimpleVectorStore.from_persist_dir(
        persist_dir=VECTORDB_PATH
      ),
      index_store=SimpleIndexStore.from_persist_dir(persist_dir=VECTORDB_PATH),
    )
    index = load_index_from_storage(storage_context)
    return index

  def get_contexts(self, query):
    return self.query_engine.query(query)

  def get_context(self, query):
    # reformat response
    context = "Context:\n"
    response = self.get_contexts(query)
    for i in range(self.top_k):
        context = context + response.source_nodes[i].text + "\n\n"
    return context