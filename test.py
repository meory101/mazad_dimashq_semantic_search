

import arabic_reshaper

from bidi.algorithm import get_display

from sentence_transformers import SentenceTransformer, util



def arabic(text):

    reshaped = arabic_reshaper.reshape(text)

    return get_display(reshaped)



model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')



database_items = [

    "منزل كبير للبيع في دمشق",

    "سيارة مستعملة بحالة ممتازة",

    "شقة مفروشة للايجار بالرياض",

    "محل تجاري بمساحة واسعة"

]


user_query = "بدي بيت واسع للشراء"


query_embedding = model.encode(user_query, convert_to_tensor=True)

items_embeddings = model.encode(database_items, convert_to_tensor=True)



cosine_scores = util.cos_sim(query_embedding, items_embeddings)[0]



print(arabic(f"كلمة البحث: {user_query}"))

print()

print(arabic("النتائج المرتبة حسب الذكاء الدلالي:"))



results = []

for i in range(len(database_items)):

    results.append({'text': database_items[i], 'score': cosine_scores[i].item()})


results = sorted(results, key=lambda x: x['score'], reverse=True)


for res in results:

    similarity_pct = res['score'] * 100

    print(arabic(f"- {res['text']} (نسبة المطابقة: {similarity_pct:.2f}%)")) 