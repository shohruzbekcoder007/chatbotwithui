import os
import json
from typing import List, Dict, Any, Optional
import chromadb
import torch
from sentence_transformers import SentenceTransformer
import re
import uuid
import logging

from retriever.encoder import cross_encode_sort

# GPU mavjudligini tekshirish va device tanlash
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CustomEmbeddingFunction:
    def __init__(self, model_name='BAAI/bge-m3', device='cuda'):
        print(f"Model yuklanmoqda: {model_name}")
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
        
        # Local model papkasini yaratish
        local_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_models')
        os.makedirs(local_models_dir, exist_ok=True)
        
        # Model nomini local papka uchun moslashtirish
        model_folder_name = model_name.replace('/', '_')
        local_model_path = os.path.join(local_models_dir, model_folder_name)
        
        # Xotira cheklovlarini sozlash
        torch.cuda.set_per_process_memory_fraction(0.95)  # GPU xotirasining 95% ini ishlatish
        
        # Logging darajasini pasaytirish
        logging.getLogger("accelerate").setLevel(logging.WARNING)
        
        # Agar model local'da mavjud bo'lsa, local'dan yuklash
        if os.path.exists(local_model_path):
            print(f"Model local'dan yuklanmoqda: {local_model_path}")
            self.model = SentenceTransformer(local_model_path, device=str(self.device))
        else:
            print(f"Model internetdan yuklanmoqda va local'ga saqlanadi: {model_name}")
            self.model = SentenceTransformer(model_name, device=str(self.device))
            # Modelni local'ga saqlash
            self.model.save(local_model_path)
            print(f"Model local'ga saqlandi: {local_model_path}")
            
        self.max_length = 4096
        self.batch_size = 256  # Batch size ni belgilash

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Matnlarni tozalash
        texts = [self._preprocess_text(text) for text in input]
        
        # Batch processing
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            # Batch encoding with normalization and truncation
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                max_length=self.max_length,
                truncation=True,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings.tolist())
        
        return embeddings
    
    def _preprocess_text(self, text: str) -> str:
        """
        Matnni tozalash va formatlash
        
        Args:
            text: Kiruvchi matn
            
        Returns:
            str: Tozalangan matn
        """
        if not isinstance(text, str):
            text = str(text)
            
        # HTML teglarni olib tashlash
        text = re.sub(r'<[^>]+>', '', text)
        
        # Ko'p bo'sh joylarni bitta bo'sh joy bilan almashtirish
        text = ' '.join(text.split())
        
        return text
    
    def get_token_count(self, text: str) -> int:
        """
        Matndagi tokenlar sonini hisoblash
        
        Args:
            text: Kiruvchi matn
        
        Returns:
            int: Tokenlar soni
        """
        return len(self.model.tokenize(text))

class RegionNER:
    """O'zbekiston hududlari uchun Named Entity Recognition"""
    
    def __init__(self):
        # MHOBT kodlari pattern
        self.mhobt_pattern = r'\b(17|1703|1703\d{3,6})\b'
        
        # SOATO kodlari pattern  
        self.soato_pattern = r'\b\d{8,10}\b'
        
        # Viloyat nomlari
        self.regions = {
            # Viloyatlar
            'andijon': ['andijon', 'андижон', 'andijan'],
            'buxoro': ['buxoro', 'бухара', 'bukhara'],
            'fargona': ['fargona', 'фергана', 'fergana', 'farghona'],
            'jizzax': ['jizzax', 'жиззах', 'jizzakh'],
            'xorazm': ['xorazm', 'хорезм', 'khorezm'],
            'namangan': ['namangan', 'наманган'],
            'navoiy': ['navoiy', 'навои', 'navoi'],
            'qashqadaryo': ['qashqadaryo', 'кашкадарья', 'kashkadarya'],
            'qoraqalpogiston': ['qoraqalpogiston', 'каракалпакстан', 'karakalpakstan'],
            'samarqand': ['samarqand', 'самарканд', 'samarkand'],
            'sirdaryo': ['sirdaryo', 'сырдарья', 'syrdarya'],
            'surxondaryo': ['surxondaryo', 'сурхандарья', 'surkhandarya'],
            'toshkent': ['toshkent', 'ташкент', 'tashkent'],
            'toshkent_shahri': ['toshkent shahri', 'ташкент шахри', 'tashkent city']
        }
        
        # Tuman/shahar nomlari (asosiy tumanlar)
        self.districts = {
            # Andijon viloyati
            'oltinko\'l': ['oltinko\'l', 'oltinkol', 'олтинкол'],
            'andijon_shahri': ['andijon shahri', 'andijon city'],
            'asaka': ['asaka', 'асака'],
            'baliqchi': ['baliqchi', 'баликчи'],
            'bo\'z': ['bo\'z', 'boз', 'боз'],
            'buloqboshi': ['buloqboshi', 'булокбоши'],
            'izboskan': ['izboskan', 'избоскан'],
            'jalaquduq': ['jalaquduq', 'жалакудук'],
            'marhamat': ['marhamat', 'мархамат'],
            'pakhtaobod': ['pakhtaobod', 'пахтаобод'],
            'qo\'rg\'ontepa': ['qo\'rg\'ontepa', 'kurgontepa', 'кургонтепа'],
            'shahrixon': ['shahrixon', 'шахрихон'],
            'ulug\'nor': ['ulug\'nor', 'ulugnar', 'улугнор'],
            'xo\'jaobod': ['xo\'jaobod', 'khojaobod', 'хожаобод'],
            'xonobod': ['xonobod', 'khonobod', 'хонобод']
        }
        
        # Barcha nom variantlarini birlashtirish
        self.all_entities = {}
        self.all_entities.update(self.regions)
        self.all_entities.update(self.districts)
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        User querydan hududiy entitylarni ajratib olish
        
        Args:
            query: Foydalanuvchi so'rovi
            
        Returns:
            Dict: Ajratib olingan entitylar
        """
        entities = {
            'mhobt_codes': [],
            'soato_codes': [],
            'region_names': [],
            'district_names': [],
            'extracted_text': []
        }
        
        query_lower = query.lower().strip()
        
        # MHOBT kodlarini ajratib olish
        mhobt_matches = re.findall(self.mhobt_pattern, query)
        entities['mhobt_codes'] = list(set(mhobt_matches))
        
        # SOATO kodlarini ajratib olish
        soato_matches = re.findall(self.soato_pattern, query)
        # MHOBT kodlari bilan bir xil bo'lmaganlarini filtrlash
        soato_codes = [code for code in soato_matches if code not in mhobt_matches]
        entities['soato_codes'] = list(set(soato_codes))
        
        # Hudud va tuman nomlarini ajratib olish
        for entity_key, variants in self.all_entities.items():
            for variant in variants:
                if variant.lower() in query_lower:
                    if entity_key in self.regions:
                        if entity_key not in entities['region_names']:
                            entities['region_names'].append(entity_key)
                            entities['extracted_text'].append(variant)
                    elif entity_key in self.districts:
                        if entity_key not in entities['district_names']:
                            entities['district_names'].append(entity_key)
                            entities['extracted_text'].append(variant)
                    break
        
        return entities
    
    def create_enhanced_query(self, original_query: str, entities: Dict[str, List[str]]) -> str:
        """
        Ajratib olingan entitylar asosida kengaytirilgan qidiruv so'rovini yaratish
        
        Args:
            original_query: Asl so'rov
            entities: Ajratib olingan entitylar
            
        Returns:
            str: Kengaytirilgan qidiruv so'rovi
        """
        enhanced_parts = [original_query]
        
        # MHOBT kodlarini qo'shish
        for code in entities['mhobt_codes']:
            enhanced_parts.append(f"MHOBT: {code}")
            enhanced_parts.append(f"mhobt {code}")
        
        # SOATO kodlarini qo'shish  
        for code in entities['soato_codes']:
            enhanced_parts.append(f"SOATO: {code}")
            enhanced_parts.append(f"soato {code}")
        
        # Hudud nomlarini qo'shish
        for region in entities['region_names']:
            enhanced_parts.append(region)
            # Barcha variantlarni qo'shish
            if region in self.regions:
                enhanced_parts.extend(self.regions[region])
        
        # Tuman nomlarini qo'shish
        for district in entities['district_names']:
            enhanced_parts.append(district)
            if district in self.districts:
                enhanced_parts.extend(self.districts[district])
        
        return " ".join(enhanced_parts)

class ChromaManager:
    """ChromaDB wrapper"""
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.embeddings = CustomEmbeddingFunction()
        self.ner = RegionNER()  # NER instansiyasini qo'shish
        
        # ChromaDB klientini yaratish - persistent_directory bilan
        persistent_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chroma_db')
        os.makedirs(persistent_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persistent_dir)
        
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"Kolleksiyani olishda xatolik: {str(e)}")
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embeddings,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Kolleksiya yaratildi: {collection_name}")

    def get_all_documents(self) -> Dict:
        """
        Barcha hujjatlarni olish
        """
        try:
            # Kolleksiya hajmini olish
            count = self.collection.count()
            if count == 0:
                print("Kolleksiya bo'sh")
                return {"documents": [], "embeddings": [], "metadatas": []}
            
            print(f"Kolleksiyadan {count} ta hujjat olinmoqda...")            
            # Ma'lumotlarni olish
            result = self.collection.get()
            
            # Ma'lumotlarni tekshirish
            if not result or 'documents' not in result:
                raise ValueError("Noto'g'ri format: documents topilmadi")
            
            if not result['documents']:
                raise ValueError("Bo'sh ma'lumotlar qaytarildi")
            
            print(f"Muvaffaqiyatli olindi: {len(result['documents'])} ta hujjat")
            return result
            
        except Exception as e:
            print(f"Hujjatlarni olishda xatolik: {str(e)}")
            return {"documents": [], "embeddings": [], "metadatas": []}
            
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
        """
        Kolleksiyaga yangi hujjatlar qo'shish - hudud kodlari uchun soddalashtirilgan
        
        Args:
            texts: Matnlar ro'yxati
            metadatas: Metadatalar ro'yxati (ixtiyoriy)
            ids: Hujjat ID lari ro'yxati (ixtiyoriy)
        """
        try:
            if not texts:
                print("Qo'shish uchun hujjatlar yo'q")
                return
            
            # ID lar generatsiya qilish
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]
            
            # Metadatani tekshirish
            if metadatas is None:
                metadatas = [{"source": "unknown"} for _ in texts]
            elif len(metadatas) != len(texts):
                raise ValueError(f"Metadatalar soni ({len(metadatas)}) matnlar soniga ({len(texts)}) mos kelmaydi")
            
            # Har bir metadataga noyub ID qo'shish
            for i, metadata in enumerate(metadatas):
                metadata["doc_id"] = ids[i]
            
            # Hujjatlarni qo'shish
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"{len(texts)} ta hujjat muvaffaqiyatli qo'shildi")
            
        except Exception as e:
            print(f"Hujjatlarni qo'shishda xatolik: {e}")
    
    def search_documents(self, query: str, n_results: int = 5):
        """Hujjatlar orasidan qidirish"""
        return self.hybrid_search(query, n_results)

    def hybrid_search(self, query: str, n_results: int = 5):
        """
        Hybrid qidiruv - NER bilan kengaytirilgan semantic va keyword qidiruvlarni birlashtirib ishlatish
        """
        try:
            # 1. NER orqali entitylarni ajratib olish
            entities = self.ner.extract_entities(query)
            print(f"NER natijasi: {entities}")
            
            # 2. Kengaytirilgan qidiruv so'rovini yaratish
            enhanced_query = self.ner.create_enhanced_query(query, entities)
            print(f"Kengaytirilgan so'rov: {enhanced_query}")
            
            # 3. Agar aniq kodlar topilgan bo'lsa, ularni alohida qidirish
            specific_searches = []
            
            # MHOBT kodlari bo'yicha aniq qidiruv
            for mhobt_code in entities['mhobt_codes']:
                try:
                    mhobt_results = self.collection.query(
                        query_texts=[f"MHOBT: {mhobt_code}"],
                        n_results=10,
                        include=['documents', 'distances', 'metadatas']
                    )
                    if mhobt_results and mhobt_results.get('documents') and mhobt_results['documents'][0]:
                        specific_searches.extend([
                            {
                                'document': doc,
                                'metadata': mhobt_results['metadatas'][0][i] if mhobt_results.get('metadatas') else {},
                                'distance': mhobt_results['distances'][0][i],
                                'score': 0.95,  # Yuqori priority
                                'search_type': 'mhobt_exact'
                            }
                            for i, doc in enumerate(mhobt_results['documents'][0])
                        ])
                except Exception as e:
                    print(f"MHOBT qidiruv xatoligi: {e}")
            
            # SOATO kodlari bo'yicha aniq qidiruv
            for soato_code in entities['soato_codes']:
                try:
                    soato_results = self.collection.query(
                        query_texts=[f"SOATO: {soato_code}"],
                        n_results=10,
                        include=['documents', 'distances', 'metadatas']
                    )
                    if soato_results and soato_results.get('documents') and soato_results['documents'][0]:
                        specific_searches.extend([
                            {
                                'document': doc,
                                'metadata': soato_results['metadatas'][0][i] if soato_results.get('metadatas') else {},
                                'distance': soato_results['distances'][0][i],
                                'score': 0.9,  # Yuqori priority
                                'search_type': 'soato_exact'
                            }
                            for i, doc in enumerate(soato_results['documents'][0])
                        ])
                except Exception as e:
                    print(f"SOATO qidiruv xatoligi: {e}")
            
            # 4. Semantic search (kengaytirilgan so'rov bilan)
            semantic_results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=min(n_results * 3, 50),
                include=['documents', 'distances', 'metadatas']
            )

            # 5. Keyword search
            keyword_results = None
            try:
                # Enhanced query so'zlari bo'yicha qidirish
                query_words = enhanced_query.lower().split()
                if len(query_words) > 0:
                    keyword_results = self.collection.query(
                        query_texts=[enhanced_query],
                        n_results=min(n_results * 2, 30),
                        where_document={"$contains": query_words[0]},
                        include=['documents', 'distances', 'metadatas']
                    )
            except Exception as keyword_error:
                print(f"Keyword search xatoligi: {keyword_error}")

            # 6. Natijalarni birlashtirish va scoring
            combined_results = []
            seen_docs = set()

            # Aniq qidiruvlar (MHOBT/SOATO) - eng yuqori priority
            for result in specific_searches:
                doc_hash = hash(result['document'][:100])
                if doc_hash not in seen_docs:
                    combined_results.append({
                        'document': result['document'],
                        'metadata': result['metadata'],
                        'distance': result['distance'],
                        'semantic_score': 0,
                        'keyword_score': 0,
                        'exact_match_score': result['score'],
                        'final_score': result['score'],
                        'doc_hash': doc_hash,
                        'search_type': result['search_type']
                    })
                    seen_docs.add(doc_hash)

            # Semantic natijalarni qo'shish
            if semantic_results and semantic_results.get('documents') and semantic_results['documents'][0]:
                for i, doc in enumerate(semantic_results['documents'][0]):
                    doc_hash = hash(doc[:100])
                    if doc_hash not in seen_docs:
                        distance = semantic_results['distances'][0][i]
                        metadata = semantic_results['metadatas'][0][i] if semantic_results.get('metadatas') else {}
                        
                        semantic_score = max(0, 1 - distance)
                        
                        combined_results.append({
                            'document': doc,
                            'metadata': metadata,
                            'distance': distance,
                            'semantic_score': semantic_score,
                            'keyword_score': 0,
                            'exact_match_score': 0,
                            'final_score': semantic_score * 0.7,
                            'doc_hash': doc_hash,
                            'search_type': 'semantic'
                        })
                        seen_docs.add(doc_hash)

            # Keyword natijalarni qo'shish
            if keyword_results and keyword_results.get('documents') and keyword_results['documents'][0]:
                for i, doc in enumerate(keyword_results['documents'][0]):
                    doc_hash = hash(doc[:100])
                    
                    if doc_hash in seen_docs:
                        # Mavjud hujjatga keyword score qo'shish
                        for result in combined_results:
                            if result['doc_hash'] == doc_hash and result.get('search_type') != 'mhobt_exact' and result.get('search_type') != 'soato_exact':
                                keyword_distance = keyword_results['distances'][0][i]
                                keyword_score = max(0, 1 - keyword_distance)
                                result['keyword_score'] = keyword_score
                                result['final_score'] = (result['semantic_score'] * 0.7) + (keyword_score * 0.3)
                                break
                    else:
                        # Yangi hujjat qo'shish
                        distance = keyword_results['distances'][0][i]
                        metadata = keyword_results['metadatas'][0][i] if keyword_results.get('metadatas') else {}
                        keyword_score = max(0, 1 - distance)
                        
                        combined_results.append({
                            'document': doc,
                            'metadata': metadata,
                            'distance': distance,
                            'semantic_score': 0,
                            'keyword_score': keyword_score,
                            'exact_match_score': 0,
                            'final_score': keyword_score * 0.5,
                            'doc_hash': doc_hash,
                            'search_type': 'keyword'
                        })
                        seen_docs.add(doc_hash)

            # 7. Entity matching bonus
            for result in combined_results:
                doc_lower = result['document'].lower()
                
                # MHOBT/SOATO kodlari uchun bonus
                for code in entities['mhobt_codes']:
                    if code in doc_lower:
                        result['final_score'] += 0.3
                
                for code in entities['soato_codes']:
                    if code in doc_lower:
                        result['final_score'] += 0.25
                
                # Hudud nomlari uchun bonus
                for region in entities['region_names']:
                    if region in doc_lower:
                        result['final_score'] += 0.2
                
                # Tuman nomlari uchun bonus
                for district in entities['district_names']:
                    if district in doc_lower:
                        result['final_score'] += 0.15

            # 8. Query bilan exact match bonus
            for result in combined_results:
                doc_lower = result['document'].lower()
                query_lower = query.lower()
                
                if query_lower in doc_lower:
                    result['final_score'] += 0.1
                
                # Enhanced query so'zlari
                enhanced_words = enhanced_query.lower().split()
                doc_words = doc_lower.split()
                matching_words = sum(1 for word in enhanced_words if word in doc_words)
                if enhanced_words:
                    word_match_ratio = matching_words / len(enhanced_words)
                    result['final_score'] += word_match_ratio * 0.05

            # 9. Final score bo'yicha tartiblash
            combined_results.sort(key=lambda x: x['final_score'], reverse=True)

            # 10. n_results gacha cheklash
            top_results = combined_results[:n_results]

            # 11. Cross-encoder bilan qayta tartiblash
            documents = [result['document'] for result in top_results]
            if documents:
                try:
                    reranked_docs = cross_encode_sort(enhanced_query, documents, n_results)
                    reranked_results = []
                    for reranked_doc in reranked_docs:
                        for result in top_results:
                            if result['document'] == reranked_doc:
                                reranked_results.append(result)
                                break
                    top_results = reranked_results
                except Exception as cross_error:
                    print(f"Cross-encoder xatoligi: {cross_error}")

            # 12. Natijalarni formatlab qaytarish
            final_documents = [result['document'] for result in top_results]
            final_distances = [result['distance'] for result in top_results]
            final_metadatas = [result['metadata'] for result in top_results]

            # NER ma'lumotlarini qaytarish uchun qo'shimcha
            search_info = {
                "entities_found": entities,
                "enhanced_query": enhanced_query,
                "search_types": [result.get('search_type', 'unknown') for result in top_results]
            }

            return {
                "documents": [final_documents],
                "distances": [final_distances],
                "metadatas": [final_metadatas],
                "scores": [result['final_score'] for result in top_results],
                "search_info": search_info  # NER ma'lumotlari
            }

        except Exception as e:
            print(f"Qidirishda xatolik: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"documents": [[]], "distances": [[]], "metadatas": [[]], "scores": [], "search_info": {}}

    def delete_all_documents(self):
        """Barcha hujjatlarni o'chirish"""
        try:
            all_docs = self.collection.get()
            all_ids = all_docs["ids"]
            before = self.collection.count()
            if all_ids:
                self.collection.delete(ids=all_ids)
                after = self.collection.count()
                self.__init__(self.collection_name) 
                print("Hujjatlar o'chirildi.")
                print(f"O'chirishdan oldin: {before}, keyin: {after}")
            else:
                print("O'chirish uchun hujjat yo'q.")
            return True
        except Exception as e:
            print(f"Hujjatlarni o'chirishda xatolik: {e}")
            return False

    def create_collection(self, collection_name: Optional[str] = None):
        """ChromaDB collection yaratish"""
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name if collection_name else self.collection_name,
                embedding_function=self.embeddings,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity uchun
            )
            print(f"Yangi kolleksiya yaratildi: {collection_name if collection_name else self.collection_name}")
        except Exception as e:
            print(f"Kolleksiya yaratishda xatolik: {str(e)}")

def add_documents_from_json(json_file_path: str):
    """JSON fayldan ma'lumotlarni o'qib ChromaDB ga qo'shish"""
    try:
        # 1. JSON faylni o'qish
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print(f"JSON fayl o'qildi. Elementlar soni: {len(data)}")
        
        # 2. Agar yakka element bo'lsa, listga o'tkazish
        if not isinstance(data, list):
            data = [data]
            print("Yakka element listga o'tkazildi")
        
        # 3. Har bir elementni alohida qo'shish
        added_count = 0
        for i, doc in enumerate(data):
            print(f"\nElement {i+1}/{len(data)} qayta ishlanmoqda...")
            
            if "content" in doc:
                # 4. Matnni tozalash
                text = doc["content"].strip()
                
                # 5. Metadata yaratish
                metadata = {k: v for k, v in doc.items() if k != "content"}
                    
                # 6. Har bir hujjat uchun noyub ID yaratish
                doc_id = str(uuid.uuid4())  # UUID yordamida noyub ID

                print(f"Matn uzunligi: {len(text)}")
                print(f"Metadata: {metadata}")
                print(f"Hujjat ID: {doc_id}")
                
                # 7. ChromaDB ga qo'shish
                try:
                    chroma_manager.add_documents(
                        texts=[text],  # Bitta matn
                        metadatas=[{**metadata, 'id': doc_id}],  # Bitta metadata, ID qo'shildi
                        ids=[doc_id]  # Hujjatni noyub ID bilan qo'shish
                    )
                    print(f"Element {i+1} muvaffaqiyatli qo'shildi")
                    added_count += 1
                except Exception as e:
                    print(f"Element {i+1} ni qo'shishda xatolik: {e}")
            else:
                print(f"Element {i+1} da 'content' maydoni topilmadi")
        
        print(f"\nBarcha elementlar qayta ishlandi. {added_count} ta yangi hujjat qo'shildi.")
        return True
    except Exception as e:
        print(f"JSON faylni qayta ishlashda xatolik: {e}")
        return False

# ChromaManager instansiyasini yaratib
chroma_manager = ChromaManager("documents")

# Qustionlar uchun alohida ChromaManager yaratish
questions_manager = ChromaManager(collection_name="questions")  # Uncommenting this line

# Funksiyalarni eksport qilish
def count_documents():
    """Bazadagi hujjatlar sonini qaytaradi"""
    return chroma_manager.collection.count()

def remove_all_documents():
    """Barcha hujjatlarni o'chirish"""
    return chroma_manager.delete_all_documents()

def search_documents(query: str, n_results: int = 5):
    """Hujjatlar orasidan qidirish"""
    return chroma_manager.search_documents(query, n_results)

def create_collection():
    """ChromaDB collection yaratish"""
    return chroma_manager.create_collection()
