import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

index = None
metadata = [] 
def generate_adv_embedding(text):
    embedding = model.encode(text, convert_to_tensor=False)
    return np.array([embedding]).astype('float32')

def build_database_embeddings(adv_list):

    global index, metadata
    metadata = adv_list 
    
    print("جاري تحويل 70 ألف إعلان إلى Vectors... قد يستغرق ذلك دقائق")
    embeddings = model.encode(adv_list, show_progress_bar=True, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatIP(dimension) 
    
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print("تم بناء قاعدة البيانات بنجاح!")

def match_query(query_text, threshold=0.50, top_k=10):

    if index is None:
        return "قاعدة البيانات غير جاهزة"

    query_vector = generate_adv_embedding(query_text)
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, k=top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= threshold:
            results.append({
                "text": metadata[idx],
                "confidence": round(float(score) * 100, 2)
            })
            
    return results

old_data = ["شقة للبيع في المزة", "بيت عربي قديم", "سيارة مرسيدس للبيع"] 
build_database_embeddings(old_data)

search_results = match_query("بدي منزل بدمشق", threshold=0.50)
print(search_results)