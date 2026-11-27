"""
FAISS Index + Multi-Query + Hybrid Search + RAG Fusion 파이프라인
===================================================================

[아키텍처 및 알고리즘 개요]
1. Multi-Query Layer: 사용자의 단일 입력을 LLM을 통해 서로 다른 5가지 관점의 질의로 확장
   - 시간 복잡도: O(M * L_token) (M=5: 확장 질의 수, L_token: 생성 토큰 수)
2. Hybrid Search Layer:
   - Dense: FAISS IndexFlatL2 (L2 거리 / 유클리디안 거리 기반 Exact Flat Index)
     - 시간 복잡도: O(M * N * D) (N: 청크 수, D: 임베딩 차원)
   - Sparse: BM25Okapi (키워드 정확 매칭)
     - 시간 복잡도: O(M * |q| * avg_doc_len)
3. RAG Fusion & RRF Layer: Reciprocal Rank Fusion 공식 적용
   - 공식: RRF(d) = sum_{q in Q} [ w_dense / (k + r_dense(d)) + w_sparse / (k + r_sparse(d)) ]
   - 시간 복잡도: O(M * K log K) (K: 검색 후보 수)
4. LLM Generation: RRF 상위 문맥을 기반으로 최종 답변 스트리밍 생성
"""

import os
import re
import math
import time
from typing import List, Dict, Any, Tuple, Optional, Generator
from dataclasses import dataclass, asdict
import numpy as np

# FAISS 라이브러리 임포트
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# BM25 라이브러리 임포트
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Google GenAI SDK 임포트
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


@dataclass
class DocumentChunk:
    """문서 분할 청크 및 메타데이터 데이터 구조"""
    chunk_id: str
    text: str
    source: str
    page: int
    char_start: int
    char_end: int


@dataclass
class SearchResult:
    """검색 및 재정렬 결과 데이터 구조"""
    chunk: DocumentChunk
    rrf_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    query_matches: Optional[List[str]] = None


class TextTokenizer:
    """한글 및 영문 혼합 텍스트 토크나이저 (BM25 전처리용)"""
    @staticmethod
    def tokenize(text: str) -> List[str]:
        # 영문, 숫자, 한글 단어 추출 (소문자 변환)
        cleaned = text.lower()
        tokens = re.findall(r'[a-zA-Z0-9가-힣]+', cleaned)
        # 1글자 초과 단어 또는 주요 한글 단어 보존
        return [t for t in tokens if len(t) > 1] or tokens


class EmbeddingService:
    """
    고차원 벡터 생성을 위한 임베딩 서비스
    Gemini API (`gemini-embedding-001`)를 우선 사용하며,
    API 키 미설정 시에도 파이프라인 테스트가 가능하도록 결정론적 대체 임베딩을 제공합니다.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self.dimension = 768  # gemini-embedding-001 기본 차원

        if self.api_key and GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[EmbeddingService] Gemini Client 초기화 실패 (Fallback 사용): {e}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 목록을 고차원 벡터(L2 정규화)로 변환
        공간 복잡도: O(N * D * 4 bytes)
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.client:
            try:
                embeddings = []
                # 배치 처리
                for text in texts:
                    res = self.client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=text
                    )
                    # res.embedding.values 또는 SDK 반환값 처리
                    if hasattr(res, "embedding") and hasattr(res.embedding, "values"):
                        embeddings.append(res.embedding.values)
                    elif hasattr(res, "embeddings") and res.embeddings:
                        embeddings.append(res.embeddings[0].values)
                    else:
                        raise ValueError("임베딩 결과 파싱 실패")
                
                arr = np.array(embeddings, dtype=np.float32)
                # Cosine Similarity를 위해 L2 단위 벡터로 정규화
                faiss.normalize_L2(arr)
                return arr
            except Exception as e:
                print(f"[EmbeddingService] Gemini API 임베딩 호출 실패, Fallback 임베딩으로 전환: {e}")

        # Fallback: 결정론적 시드 기반 벡터 생성 (API 키 미설정 또는 네트워크 오류 시 로컬 작동 보장)
        return self._generate_fallback_embeddings(texts)

    def _generate_fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """단어 해시 기반 의사 임베딩 (L2 정규화 적용)"""
        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            tokens = TextTokenizer.tokenize(text)
            for token in tokens:
                h = abs(hash(token)) % self.dimension
                vec[h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            else:
                vec[0] = 1.0
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


class FAISSIndexManager:
    """
    FAISS 기반 고차원 벡터 인덱스 관리자
    - IndexFlatL2 (L2 거리 / 유클리디안 거리 기반 Exact Flat Index)
    - IndexHNSWFlat (계층적 소규모 그래프 근사 탐색)
    - 로컬 디스크 파일 영구 저장(faiss_flat.index) 및 복원(load) 완벽 지원
    """
    def __init__(self, dimension: int = 768, index_type: str = "IndexFlatL2", data_dir: Optional[str] = None):
        self.dimension = dimension
        self.index_type = index_type  # "IndexFlatL2" 또는 "IndexHNSWFlat"
        self.chunks: List[DocumentChunk] = []
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.index_file_path = os.path.join(self.data_dir, "faiss_flat.index")
        self.metadata_file_path = os.path.join(self.data_dir, "chunks_metadata.json")

        self._init_index()

    def _init_index(self):
        """인덱스 구조 초기화"""
        if FAISS_AVAILABLE:
            if self.index_type == "IndexHNSWFlat":
                # M=32 개의 링크를 갖는 HNSW 그래프 인덱스
                self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_L2)
            else:
                # 기본값: 표준 L2 거리 기반 IndexFlatL2
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = None
            self._vectors: List[np.ndarray] = []

    def clear(self):
        """인덱스 초기화"""
        self.chunks.clear()
        self._init_index()
        if os.path.exists(self.index_file_path):
            try:
                os.remove(self.index_file_path)
            except Exception:
                pass
        if os.path.exists(self.metadata_file_path):
            try:
                os.remove(self.metadata_file_path)
            except Exception:
                pass

    def add_chunks(self, chunks: List[DocumentChunk], vectors: np.ndarray):
        """문서 청크와 임베딩 벡터 적재 후 디스크(faiss_flat.index)에 저장"""
        if len(chunks) != len(vectors):
            raise ValueError("청크 수와 벡터 수가 일치하지 않습니다.")
        self.chunks.extend(chunks)

        if FAISS_AVAILABLE:
            if not vectors.flags['C_CONTIGUOUS']:
                vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            self.index.add(vectors)
            self.save_to_disk()
        else:
            for v in vectors:
                self._vectors.append(v)
            self.save_to_disk()

    def save_to_disk(self):
        """FAISS Flat DB 및 메타데이터를 디스크에 영구 저장"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            if FAISS_AVAILABLE and self.index is not None:
                faiss.write_index(self.index, self.index_file_path)
            
            # 메타데이터 JSON 저장
            meta_list = [asdict(c) for c in self.chunks]
            import json
            with open(self.metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(meta_list, f, ensure_ascii=False, indent=2)
            print(f"[FAISSIndexManager] Flat DB 저장 완료: {self.index_file_path} (청크 {len(self.chunks)}개)")
        except Exception as e:
            print(f"[FAISSIndexManager] Flat DB 저장 중 오류: {e}")

    def load_from_disk(self) -> bool:
        """디스크에 저장된 faiss_flat.index 및 메타데이터 복원"""
        import json
        if not os.path.exists(self.index_file_path) or not os.path.exists(self.metadata_file_path):
            return False

        try:
            # 1. 메타데이터 로드
            with open(self.metadata_file_path, "r", encoding="utf-8") as f:
                meta_list = json.load(f)
            self.chunks = [DocumentChunk(**m) for m in meta_list]

            # 2. FAISS 인덱스 로드
            if FAISS_AVAILABLE:
                self.index = faiss.read_index(self.index_file_path)
                print(f"[FAISSIndexManager] 디스크에서 Flat DB 로드 완료: {self.index_file_path} (총 {len(self.chunks)}개)")
                return True
            else:
                return False
        except Exception as e:
            print(f"[FAISSIndexManager] Flat DB 로드 실패: {e}")
            return False

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Dense 벡터 검색 수행 (L2 거리를 유사도 점수로 변환: 1 / (1 + distance))
        반환: [(chunk_idx, similarity_score), ...]
        시간 복잡도: O(N * D) (IndexFlatL2) / O(log N) (IndexHNSWFlat)
        """
        if not self.chunks:
            return []

        top_k = min(top_k, len(self.chunks))
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        if FAISS_AVAILABLE and self.index is not None:
            if not query_vector.flags['C_CONTIGUOUS']:
                query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
            distances, indices = self.index.search(query_vector, top_k)
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0 and idx < len(self.chunks):
                    # L2 거리는 작을수록 유사하므로 1 / (1 + dist) 유사도로 변환
                    sim_score = float(1.0 / (1.0 + max(0.0, float(dist))))
                    results.append((int(idx), sim_score))
            return results
        else:
            # Fallback Euclidean distance
            scores = []
            for v in self._vectors:
                dist = float(np.linalg.norm(query_vector[0] - v))
                scores.append(1.0 / (1.0 + dist))
            sorted_indices = np.argsort(scores)[::-1][:top_k]
            return [(int(i), scores[i]) for i in sorted_indices]


class BM25IndexManager:
    """
    BM25 기반 키워드(Sparse) 검색기
    정확한 단어 매칭 및 빈도-역문서 빈도(TF-IDF) 가중치 적용
    """
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def clear(self):
        self.chunks.clear()
        self.tokenized_corpus.clear()
        self.bm25 = None

    def build_index(self, chunks: List[DocumentChunk]):
        """문서 청크들로부터 BM25 역색인 구축"""
        self.chunks = list(chunks)
        self.tokenized_corpus = [TextTokenizer.tokenize(c.text) for c in self.chunks]
        if self.tokenized_corpus and BM25_AVAILABLE:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Sparse 키워드 검색 수행
        시간 복잡도: O(|q| * avg_doc_len)
        """
        if not self.chunks or not self.bm25:
            return []

        top_k = min(top_k, len(self.chunks))
        tokens = TextTokenizer.tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in sorted_indices if scores[i] > 0]


class MultiQueryExpander:
    """
    다중 쿼리 확장 계층 (Multi-Query Expansion Layer)
    단일 입력 쿼리를 LLM을 통해 5가지 서로 다른 관점의 파생 쿼리로 확장합니다.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        if self.api_key and GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[MultiQueryExpander] Gemini Client 초기화 실패: {e}")

    def expand(self, query: str) -> List[str]:
        """
        사용자 질의를 5가지 관점으로 확장
        1. 원본 쿼리
        2. 핵심 키워드/동의어 기반 질의
        3. 세부 기술 메커니즘 관점 질의
        4. 원리 및 상위 배경 관점 질의
        5. 실무 적용 및 해결책 관점 질의
        """
        expanded = [query]  # 항상 원본 쿼리 포함

        if self.client:
            prompt = f"""당신은 AI 정보 검색(RAG) 전문가입니다. 사용자의 질문을 다양한 관점에서 검색할 수 있도록 4개의 서로 다른 파생 검색어(총 5개 관점 완성용)를 작성해 주세요.
각 파생 검색어는 다음 관점을 반영해야 합니다:
1. 핵심 용어 및 동의어 중심 검색어
2. 세부 기술적 원리 및 메커니즘 검색어
3. 배경 원리 및 상위 개념 검색어
4. 실무 적용, 활용 사례 및 해결책 관점 검색어

반드시 다른 설명 없이 한 줄에 하나의 검색어만 총 4줄로 출력하세요.

원래 질문: {query}"""
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )
                text = response.text or ""
                lines = [l.strip().lstrip("0123456789.-) ") for l in text.strip().split("\n") if l.strip()]
                for line in lines[:4]:
                    if line and line not in expanded:
                        expanded.append(line)
            except Exception as e:
                print(f"[MultiQueryExpander] LLM 쿼리 확장 실패 (Fallback 규칙 적용): {e}")

        # 5개 미만인 경우 규칙 기반 확장으로 5개 보충
        if len(expanded) < 5:
            tokens = TextTokenizer.tokenize(query)
            token_str = " ".join(tokens)
            candidates = [
                f"{query} 원리와 개념",
                f"{token_str} 핵심 정의 및 특징",
                f"{query} 사용 방법 및 적용 사례",
                f"{token_str} 장단점 및 세부 구조"
            ]
            for c in candidates:
                if c not in expanded and len(expanded) < 5:
                    expanded.append(c)

        return expanded[:5]


class RRFFusion:
    """
    Reciprocal Rank Fusion (상호 순위 융합) 계층
    공식: RRF(d) = sum_{q in Q} [ w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d)) ]
    기본 k = 60 (RRF 표준 하이퍼파라미터)
    """
    def __init__(self, k: int = 60, dense_weight: float = 0.6, sparse_weight: float = 0.4):
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse(
        self,
        dense_results_per_query: List[List[Tuple[int, float]]],
        sparse_results_per_query: List[List[Tuple[int, float]]],
        chunks: List[DocumentChunk],
        top_n: int = 5
    ) -> List[SearchResult]:
        """
        다중 쿼리의 Dense 및 Sparse 검색 결과 목록을 집계하여 최적 순위로 정렬
        시간 복잡도: O(M * K log K)
        """
        rrf_scores: Dict[int, float] = {}
        best_dense_rank: Dict[int, int] = {}
        best_sparse_rank: Dict[int, int] = {}

        # 1. Dense 검색 순위 집계
        for query_idx, res_list in enumerate(dense_results_per_query):
            for rank, (chunk_idx, _) in enumerate(res_list, start=1):
                # RRF 가중치 점수 합산
                score = self.dense_weight / (self.k + rank)
                rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + score
                if chunk_idx not in best_dense_rank or rank < best_dense_rank[chunk_idx]:
                    best_dense_rank[chunk_idx] = rank

        # 2. Sparse 검색 순위 집계
        for query_idx, res_list in enumerate(sparse_results_per_query):
            for rank, (chunk_idx, _) in enumerate(res_list, start=1):
                score = self.sparse_weight / (self.k + rank)
                rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + score
                if chunk_idx not in best_sparse_rank or rank < best_sparse_rank[chunk_idx]:
                    best_sparse_rank[chunk_idx] = rank

        # 3. RRF 점수 기준 내림차순 정렬
        sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        results: List[SearchResult] = []
        for chunk_idx, score in sorted_chunks[:top_n]:
            if 0 <= chunk_idx < len(chunks):
                results.append(SearchResult(
                    chunk=chunks[chunk_idx],
                    rrf_score=round(score, 5),
                    dense_rank=best_dense_rank.get(chunk_idx),
                    sparse_rank=best_sparse_rank.get(chunk_idx)
                ))

        return results


class RAGPipeline:
    """
    통합 RAG 파이프라인
    - PDF 문서 수집 및 청킹
    - FAISS 및 BM25 하이브리드 색인 구축
    - Multi-Query 확장 -> 하이브리드 검색 -> RRF 랭킹 융합 -> 답변 생성
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.embedding_service = EmbeddingService(api_key=self.api_key)
        self.faiss_manager = FAISSIndexManager(dimension=768, index_type="IndexFlatL2")
        self.bm25_manager = BM25IndexManager()
        self.query_expander = MultiQueryExpander(api_key=self.api_key)
        self.rrf_fusion = RRFFusion(k=60, dense_weight=0.6, sparse_weight=0.4)
        self.all_chunks: List[DocumentChunk] = []
        self.client = None

        if self.api_key and GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[RAGPipeline] Gemini Client 초기화 실패: {e}")

        # 1. 디스크에 저장된 faiss_flat.index가 있다면 자동 복원
        if self.faiss_manager.load_from_disk():
            self.all_chunks = list(self.faiss_manager.chunks)
            self.bm25_manager.build_index(self.all_chunks)
            print(f"[RAGPipeline] 디스크에서 기존 Flat DB 자동 로드 완료: {len(self.all_chunks)}개 청크")
        else:
            # 2. 디스크에 DB가 없는 경우 기본 FAISS Flat DB 자동 생성 (DB 부재 에러 방지)
            self._seed_default_flat_db()

    def _seed_default_flat_db(self):
        """기본 초기 FAISS Flat DB 생성 (최초 실행 시 DB 부재 에러 방지)"""
        sample_pages = [
            (1, "인공지능(AI)과 자연어 처리 분야에서 검색 증강 생성(RAG, Retrieval-Augmented Generation)은 LLM의 환각(Hallucination) 현상을 방지하고 최신 지식을 주입하기 위한 핵심 기법입니다. FAISS IndexFlatL2는 정확한 유클리디안 거리(L2) 기반의 고차원 벡터 검색을 수행합니다."),
            (2, "하이브리드 검색(Hybrid Search)은 Dense Vector 검색(FAISS)과 Sparse 키워드 검색(BM25)을 결합하여 문맥적 의미와 고유명사 정확도를 동시에 만족합니다."),
            (3, "다중 쿼리 확장(Multi-Query Expansion)은 5가지 관점으로 검색어를 확장하며, Reciprocal Rank Fusion(RRF, k=60) 알고리즘을 적용하여 최적의 상위 문서들을 재정렬(Re-ranking)합니다.")
        ]
        self.index_pdf_text("FAISS_Default_Knowledge_Base.pdf", sample_pages)
        print("[RAGPipeline] 초기 FAISS Flat DB (IndexFlatL2) 구축 완료.")

    def index_pdf_text(self, filename: str, pages_text: List[Tuple[int, str]], chunk_size: int = 500, chunk_overlap: int = 100) -> Dict[str, Any]:
        """
        PDF 페이지 텍스트를 수신하여 청킹 및 벡터/키워드 색인 생성
        """
        start_time = time.time()
        new_chunks: List[DocumentChunk] = []

        # 1. 텍스트 청킹 (Chunking)
        for page_num, text in pages_text:
            text = text.strip()
            if not text:
                continue

            start = 0
            chunk_idx = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_str = text[start:end].strip()
                if chunk_str:
                    c = DocumentChunk(
                        chunk_id=f"{filename}_p{page_num}_c{chunk_idx}",
                        text=chunk_str,
                        source=filename,
                        page=page_num,
                        char_start=start,
                        char_end=end
                    )
                    new_chunks.append(c)
                    chunk_idx += 1
                start += (chunk_size - chunk_overlap)

        if not new_chunks:
            return {"status": "error", "message": "유효한 텍스트 청크를 생성하지 못했습니다."}

        # 2. 임베딩 계산 및 FAISS 적재
        chunk_texts = [c.text for c in new_chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)
        self.faiss_manager.add_chunks(new_chunks, embeddings)

        # 3. BM25 인덱스 갱신
        self.all_chunks.extend(new_chunks)
        self.bm25_manager.build_index(self.all_chunks)

        elapsed = time.time() - start_time
        return {
            "status": "success",
            "filename": filename,
            "total_chunks_created": len(new_chunks),
            "total_indexed_chunks": len(self.all_chunks),
            "elapsed_seconds": round(elapsed, 3),
            "complexity": {
                "chunking": f"O(P * L)",
                "embedding": f"O(N * D) where N={len(new_chunks)}, D={self.embedding_service.dimension}",
                "faiss_insertion": f"O(N * D)",
                "bm25_indexing": f"O(Total_Words)"
            }
        }

    def search_hybrid_rrf(self, query: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Multi-Query + Hybrid Search (FAISS + BM25) + RRF 전체 검색 파이프라인 실행
        """
        start_time = time.time()

        # 1단계: Multi-Query 5가지 관점 확장
        t0 = time.time()
        expanded_queries = self.query_expander.expand(query)
        mq_time = time.time() - t0

        dense_results_per_query = []
        sparse_results_per_query = []

        # 2단계: 각 쿼리별 Dense(FAISS) & Sparse(BM25) 동시 검색
        t1 = time.time()
        for q in expanded_queries:
            # Dense Vector Search
            q_emb = self.embedding_service.embed_texts([q])[0]
            d_res = self.faiss_manager.search(q_emb, top_k=10)
            dense_results_per_query.append(d_res)

            # Sparse BM25 Search
            s_res = self.bm25_manager.search(q, top_k=10)
            sparse_results_per_query.append(s_res)
        search_time = time.time() - t1

        # 3단계: Reciprocal Rank Fusion(RRF) 상호 순위 융합
        t2 = time.time()
        final_results = self.rrf_fusion.fuse(
            dense_results_per_query=dense_results_per_query,
            sparse_results_per_query=sparse_results_per_query,
            chunks=self.all_chunks,
            top_n=top_n
        )
        rrf_time = time.time() - t2

        total_time = time.time() - start_time

        return {
            "original_query": query,
            "expanded_queries": expanded_queries,
            "ranked_documents": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "source": r.chunk.source,
                    "page": r.chunk.page,
                    "text": r.chunk.text,
                    "rrf_score": r.rrf_score,
                    "dense_rank": r.dense_rank,
                    "sparse_rank": r.sparse_rank
                }
                for r in final_results
            ],
            "timings": {
                "multi_query_expansion_sec": round(mq_time, 3),
                "hybrid_search_sec": round(search_time, 3),
                "rrf_fusion_sec": round(rrf_time, 3),
                "total_retrieval_sec": round(total_time, 3)
            },
            "complexity": {
                "multi_query": "O(M * L_token) (M=5)",
                "faiss_dense_search": f"O(M * N * D) (N={len(self.all_chunks)}, D={self.embedding_service.dimension})",
                "bm25_sparse_search": "O(M * |q| * avg_doc_len)",
                "rrf_fusion": "O(M * K log K) where K is candidate pool size"
            }
        }

    def generate_answer_stream(self, query: str, top_contexts: List[Dict[str, Any]]) -> Generator[str, None, None]:
        """
        RRF로 선별된 문맥을 바탕으로 LLM 스트리밍 답변 생성
        """
        context_str = "\n\n".join([
            f"[문서: {c['source']} | 페이지: {c['page']} | RRF점수: {c['rrf_score']}]\n{c['text']}"
            for c in top_contexts
        ])

        system_instruction = (
            "당신은 주어진 문서 컨텍스트에 기반하여 정확하고 신뢰성 있는 답변을 제공하는 전문 AI 어시스턴트입니다.\n"
            "반드시 아래 제공된 [참고 문서]의 내용을 기반으로 질문에 명확하게 답변하십시오.\n"
            "답변 끝에는 반드시 인용한 출처(파일명 및 페이지)를 명확하게 밝히십시오."
        )

        user_prompt = f"### [참고 문서]\n{context_str}\n\n### [질문]\n{query}\n\n위 문서를 참고하여 체계적으로 답변하세요:"

        if self.client:
            try:
                # Gemini 스트리밍 호출
                stream_res = self.client.models.generate_content_stream(
                    model="gemini-3.8-flash",
                    contents=[system_instruction, user_prompt]
                )
                for chunk in stream_res:
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as e:
                yield f"\n[오류] Gemini 스트리밍 생성 중 오류 발생: {str(e)}\n(Fallback 로컬 요약 답변으로 대체합니다)\n\n"

        # Fallback 답변 생성
        yield f"### [검색 요약 결과]\n질문 '{query}'에 대해 상위 {len(top_contexts)}개의 관련 문서 청크가 RRF 융합을 통해 성공적으로 검색되었습니다.\n\n"
        for i, c in enumerate(top_contexts, start=1):
            yield f"- **출처 {i}**: `{c['source']}` (p.{c['page']}, RRF 점수: {c['rrf_score']})\n  > {c['text'][:120]}...\n\n"
        yield "※ 현재 Gemini API 키가 설정되지 않아 로컬 검색 결과 요약으로 응답하였습니다. 환경변수 `GEMINI_API_KEY`를 설정하시면 완전한 실시간 LLM 답변이 생성됩니다."
