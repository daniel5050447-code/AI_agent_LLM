"""
Advanced RAG 파이프라인 단위 테스트 스크립트 (test_local.py)
=============================================================
이 스크립트는 서버를 띄우지 않고도 파이프라인의 핵심 컴포넌트들을 직접 검증합니다:
1. 텍스트 청킹 (Chunking)
2. FAISS 인덱스 적재 (Dense Vector)
3. BM25 역색인 적재 (Sparse Keyword)
4. Multi-Query 5가지 관점 질의 확장
5. 하이브리드 동시 검색
6. Reciprocal Rank Fusion (RRF) 상호 순위 융합
7. 최종 답변 스트리밍 생성
"""

import os
import sys

# 현재 디렉토리를 모듈 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGPipeline


def run_test():
    print("=" * 65)
    print("  Advanced RAG (FAISS + BM25 + MultiQuery + RRF) 파이프라인 테스트")
    print("=" * 65)

    # 1. 파이프라인 인스턴스 생성
    print("\n[Step 1] RAG 파이프라인 초기화...")
    pipeline = RAGPipeline()
    print("  -> 초기화 완료 (FAISS 차원: 768, RRF k=60)")

    # 2. 가상 PDF 문서 텍스트 적재
    sample_pages = [
        (1, """인공지능(AI)과 자연어 처리 분야에서 검색 증강 생성(RAG, Retrieval-Augmented Generation)은
거대 언어 모델(LLM)이 가진 환각(Hallucination) 현상을 극복하고 최신 외부 지식을 주입하기 위한 핵심 기법입니다.
전통적인 RAG는 단일 쿼리 벡터 검색에만 의존하여 어휘 불일치(Vocabulary Mismatch) 문제를 자주 겪습니다."""),
        (2, """하이브리드 검색(Hybrid Search)은 Dense Vector 검색과 Sparse 키워드 검색(BM25)을 결합합니다.
FAISS는 고차원 벡터 공간에서 L2 정규화 내적(IndexFlatIP) 또는 HNSW 그래프를 통해 빠른 의미적 유사도 검색을 수행합니다.
BM25는 정확한 키워드 매칭과 역문서 빈도 가중치를 제공하여 고유명사와 전문 용어 검색에 강점을 보입니다."""),
        (3, """Multi-Query 확장은 사용자의 단일 입력 질문을 다양한 시각(기술 원리, 활용 사례, 동의어 등)으로 5가지로 확장합니다.
이후 각 검색 결과의 순위를 취합하기 위해 Reciprocal Rank Fusion(RRF) 알고리즘을 사용합니다.
RRF 공식: RRF_Score = sum( weight / (k + rank) ). 이를 통해 서로 다른 척도의 점수를 순위 기반으로 공정하게 융합합니다.""")
    ]

    print("\n[Step 2] 샘플 문서 적재 및 색인 구축 (FAISS & BM25)...")
    res = pipeline.index_pdf_text("AI_RAG_Overview.pdf", sample_pages, chunk_size=250, chunk_overlap=50)
    print(f"  -> 색인 결과: 생성 청크 {res['total_chunks_created']}개, 총 누적 청크 {res['total_indexed_chunks']}개 ({res['elapsed_seconds']}초)")

    # 3. 질문 질의 및 Multi-Query + Hybrid Search + RRF 실행
    test_query = "RAG에서 하이브리드 검색과 RRF 알고리즘이 왜 중요한가요?"
    print(f"\n[Step 3] 질의 실행: '{test_query}'")
    
    search_res = pipeline.search_hybrid_rrf(test_query, top_n=3)

    print("\n  [확장된 5가지 다중 쿼리]:")
    for i, q in enumerate(search_res["expanded_queries"], start=1):
        print(f"    {i}. {q}")

    print("\n  [RRF 순위 융합 최적 문서 Top-3]:")
    for i, doc in enumerate(search_res["ranked_documents"], start=1):
        print(f"    #{i} [RRF 점수: {doc['rrf_score']}] (Dense: #{doc['dense_rank']}, Sparse: #{doc['sparse_rank']})")
        print(f"       출처: {doc['source']} (p.{doc['page']})")
        print(f"       본문: {doc['text'][:90]}...")

    print("\n  [소요 시간]:", search_res["timings"])
    print("  [알고리즘 복잡도]:", search_res["complexity"])

    # 4. LLM 답변 스트리밍 생성
    print("\n[Step 4] LLM 답변 생성 스트리밍:")
    print("-" * 65)
    for token in pipeline.generate_answer_stream(test_query, search_res["ranked_documents"]):
        print(token, end="", flush=True)
    print("\n" + "-" * 65)

    print("\n[테스트 완료] 모든 RAG 파이프라인 단계가 정상적으로 실행되었습니다!\n")


if __name__ == "__main__":
    run_test()
