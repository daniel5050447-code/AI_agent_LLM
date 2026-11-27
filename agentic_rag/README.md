# Advanced RAG: FAISS + Multi-Query + Hybrid Search + RAG Fusion

고차원 벡터 검색(FAISS), 키워드 검색(BM25), 5가지 관점 다중 쿼리 확장(Multi-Query Expansion), 그리고 상호 순위 융합(Reciprocal Rank Fusion; RRF)을 결합한 프로덕션급 RAG 파이프라인 실습 프로젝트입니다.

---

## 1. 아키텍처 및 파이프라인 흐름

```
[사용자 질문 Q]
       │
       ▼
[1. Multi-Query Expansion Layer] ──> 5가지 관점 파생 쿼리 생성 (Q1 ~ Q5)
       │
       ├───> [2. Dense Vector Search (FAISS IndexFlatL2)] ──> Top-10 의미적 유사도 청크
       │
       └───> [3. Sparse Keyword Search (BM25Okapi)]     ──> Top-10 키워드 매칭 청크
       │
       ▼
[4. RRF Re-ranking Layer] ──> RRF(d) = Σ [ w_dense / (60 + r_dense) + w_sparse / (60 + r_sparse) ]
       │
       ▼
[5. Top-5 최적 문서 컨텍스트 선정 & LLM 스트리밍 (StreamHandler)]
       │
       ▼
[Google Apps Script (index.html & Code.gs) 프론트엔드 실시간 렌더링]
```

### 알고리즘 복잡도
| 단계 | 알고리즘 | 시간 복잡도 | 공간 복잡도 |
| :--- | :--- | :--- | :--- |
| **Multi-Query** | LLM 5관점 확장 | $O(M \cdot L_{token})$ ($M=5$) | $O(M \cdot L_{token})$ |
| **Dense Search** | FAISS `IndexFlatL2` | $O(M \cdot N \cdot D)$ ($D=768$) | $O(N \cdot D \cdot 4\text{B})$ |
| **Sparse Search** | BM25 Okapi | $O(M \cdot \|q\| \cdot \text{avg\_len})$ | $O(V + \sum \|d\|)$ |
| **RRF Re-ranking** | Reciprocal Rank Fusion | $O(M \cdot K \log K)$ ($k=60$) | $O(K)$ |
| **Generation** | Gemini 3.8 Flash Streaming | $O(L_{output})$ | $O(L_{output})$ |
