"""
arq_rag package — ARQ = ADC Reranking + QJL (TurboQuant)
==========================================================
Thuật toán cao cấp nhất trong hệ thống:
- PQ Encoding (giống rag_pq) làm tầng nén
- ADC Reranking (từ rag_pq)
- QJL (Johnson-Lindenstrauss) projection để sketch thêm metadata
  → giảm phương sai ước lượng khoảng cách (TurboQuant)
"""
