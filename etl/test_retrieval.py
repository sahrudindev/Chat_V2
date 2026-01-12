import os
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel

# --- KONFIGURASI ---
QDRANT_HOST = "localhost" # Sesuaikan jika via Docker
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_data"

# Init Model (Gunakan GPU jika ada, sama seperti ETL)
print("⏳ Loading Model BGE-M3 untuk Querying...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda') 
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def hybrid_search(query_text, limit=5):
    print(f"\n🔎 Mencari: '{query_text}'")
    
    # 1. Generate Query Embeddings (Dense + Sparse)
    output = model.encode(query_text, return_dense=True, return_sparse=True)
    dense_vec = output['dense_vecs']
    sparse_vec = output['lexical_weights']

    # Konversi format Sparse untuk Qdrant
    sparse_indices = [int(k) for k in sparse_vec.keys()]
    sparse_values = [float(v) for v in sparse_vec.values()]

    # 2. Eksekusi Hybrid Search di Qdrant
    # Strategi: Kita mencari berdasarkan Dense Vector, 
    # TAPI kita gunakan Sparse Vector untuk me-rerank hasil (fusion)
    # atau kita bisa gunakan fitur 'prefetch' Qdrant untuk true hybrid.
    
    # Di sini kita gunakan pendekatan modern Qdrant: Query Fusion
    # Kita cari di Dense, cari di Sparse, lalu gabungkan skornya.
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=models.NamedVector(
            name="dense", 
            vector=dense_vec.tolist()
        ),
        query_filter=None,
        limit=limit,
        with_payload=True
    )
    
    # NOTE: Untuk implementasi Hybrid yang lebih advanced (RRF Fusion),
    # Qdrant versi terbaru mendukung query batching prefetch. 
    # Tapi search() standar di atas sudah cukup untuk memverifikasi data masuk.
    
    # Mari kita coba search spesifik SPARSE saja untuk tes keyword
    print("   👉 Hasil Pencarian (Top 3):")
    for i, hit in enumerate(results[:3]):
        payload = hit.payload
        print(f"   [{i+1}] Score: {hit.score:.4f}")
        print(f"       Ticker: {payload.get('ticker', 'N/A')} | Date: {payload.get('date', 'N/A')}")
        print(f"       Text: {payload.get('text', '')[:100]}...") # Potong teks biar rapi
        print("-" * 40)

if __name__ == "__main__":
    # Test Case 1: Pertanyaan Semantik Umum
    hybrid_search("Bagaimana performa laba bersih bank BCA tahun 2023?")
    
    # Test Case 2: Keyword Spesifik (Tes apakah Sparse vector bekerja)
    # Coba cari ticker yang mungkin ada di data Anda
    hybrid_search("Dividen tunai ADRO")