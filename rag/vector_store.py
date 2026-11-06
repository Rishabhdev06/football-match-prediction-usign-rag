"""
Persistent Vector Store Management via ChromaDB
================================================
Manages persistent embeddings, idempotent document ingestion,
metadata filtering, and semantic similarity search.
"""

import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_COLLECTION = 'football_knowledge'
DEFAULT_PERSIST_DIR = os.environ.get('CHROMA_PERSIST_DIR') or (
    os.path.join(PROJECT_ROOT, 'chroma_db') if os.path.exists(os.path.join(PROJECT_ROOT, 'chroma_db'))
    else 'chroma_db'
)


def get_embedding_function():
    """
    Initializes the embedding function. Uses the locally cached
    all-MiniLM-L6-v2 model via ONNX.
    """
    return embedding_functions.DefaultEmbeddingFunction()


def get_chroma_client(persist_directory=None):
    """Returns a persistent ChromaDB client instance."""
    if persist_directory is None:
        persist_directory = DEFAULT_PERSIST_DIR
    os.makedirs(persist_directory, exist_ok=True)
    return chromadb.PersistentClient(path=persist_directory)


def get_collection(client=None, collection_name=DEFAULT_COLLECTION):
    """Retrieves or creates the primary football knowledge vector collection."""
    if client is None:
        client = get_chroma_client()
    ef = get_embedding_function()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={'description': 'EPL match records, team summaries, and reference docs'}
    )


def ingest_documents(chunks, client=None, collection_name=DEFAULT_COLLECTION, batch_size=100, force_reload=False):
    """
    Idempotent document ingestion:
    Checks for existing chunk IDs to prevent duplicate embedding computation.
    """
    col = get_collection(client, collection_name)
    total_chunks = len(chunks)
    if total_chunks == 0:
        return {'total': 0, 'added': 0, 'existing': 0}

    # Fetch existing IDs in collection
    existing_data = col.get()
    existing_ids = set(existing_data.get('ids', []))

    if force_reload and existing_ids:
        col.delete(ids=list(existing_ids))
        existing_ids = set()

    to_add = [c for c in chunks if c['id'] not in existing_ids]
    already_present = len(chunks) - len(to_add)

    if not to_add:
        return {
            'total': total_chunks,
            'added': 0,
            'existing': already_present
        }

    # Batch upsert
    for i in range(0, len(to_add), batch_size):
        batch = to_add[i:i + batch_size]
        ids = [b['id'] for b in batch]
        docs = [b['content'] for b in batch]
        metas = [b['metadata'] for b in batch]
        col.upsert(ids=ids, documents=docs, metadatas=metas)

    return {
        'total': total_chunks,
        'added': len(to_add),
        'existing': already_present
    }


def query_collection(query_text, top_k=5, where_filter=None, client=None, collection_name=DEFAULT_COLLECTION):
    """
    Executes semantic similarity search on the knowledge collection.
    """
    col = get_collection(client, collection_name)
    total_count = col.count()
    if total_count == 0:
        return []

    actual_k = min(top_k, total_count)
    kwargs = {
        'query_texts': [query_text],
        'n_results': actual_k
    }
    if where_filter:
        kwargs['where'] = where_filter

    results = col.query(**kwargs)
    passages = []
    if not results or not results['documents'] or not results['documents'][0]:
        return passages

    ids = results['ids'][0]
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    distances = results['distances'][0] if 'distances' in results and results['distances'] else [0.0]*len(ids)

    for doc_id, text, meta, dist in zip(ids, docs, metas, distances):
        # Convert distance to similarity score in [0.0, 1.0]
        similarity = round(max(0.0, 1.0 - (dist / 2.0)), 4)
        passages.append({
            'id': doc_id,
            'text': text,
            'metadata': meta,
            'distance': round(dist, 4),
            'similarity': similarity
        })

    return passages


def get_collection_stats(client=None, collection_name=DEFAULT_COLLECTION):
    """Returns summary statistics of the vector store."""
    col = get_collection(client, collection_name)
    count = col.count()
    if count == 0:
        return {'total_records': 0, 'doc_types': {}}

    data = col.get()
    doc_types = {}
    for m in data.get('metadatas', []):
        dt = m.get('doc_type', 'other')
        doc_types[dt] = doc_types.get(dt, 0) + 1

    return {
        'total_records': count,
        'doc_types': doc_types
    }
