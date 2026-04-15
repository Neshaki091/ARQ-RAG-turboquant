import re

def filter_relevant_contexts(query, contexts, top_n=3):
    """
    Lọc ngữ cảnh dựa trên các từ khóa quan trọng của câu hỏi để khử nhiễu.
    """
    if not contexts:
        return []
    
    # 1. Trích xuất từ khóa từ query
    stop_words = {"là", "gì", "của", "và", "trong", "so", "với", "như", "thế", "nào", "có", "đặc", "điểm", "khác", "biệt", "đóng", "vai", "trò"}
    words = re.findall(r'\w+', query.lower())
    
    # Từ khóa "Elite" - Nếu xuất hiện thì chunk đó cực kỳ quan trọng
    elite_map = {
        "johnson": 100, "lindenstrauss": 100, "jl": 100, 
        "buffon": 100, "kim": 100, "lượng": 50, "tử": 50,
        "l2": 30, "l1": 30, "sai": 10, "số": 10
    }
    
    keywords = [w for w in words if len(w) > 2 and w not in stop_words]
    
    if not keywords:
        return contexts[:top_n]
    
    # 2. Tính điểm cho mỗi context
    scored_contexts = []
    for ctx in contexts:
        score = 0
        ctx_lower = ctx.lower()
        
        # Thưởng lớn cho Elite keywords
        for elite, weight in elite_map.items():
            if elite in ctx_lower:
                score += weight * (1 + ctx_lower.count(elite) * 0.1)
        
        # Điểm cho các từ khóa khác
        for kw in keywords:
            if kw not in elite_map and kw in ctx_lower:
                score += 5
        
        scored_contexts.append((score, ctx))
    
    # 3. Sắp xếp và lấy top_n
    scored_contexts.sort(key=lambda x: x[0], reverse=True)
    
    # Chỉ lấy những chunk có nội dung thực sự liên quan (score > 20)
    final_selection = [ctx for score, ctx in scored_contexts if score > 20]
    
    if not final_selection:
        return contexts[:top_n]
        
    return final_selection[:top_n]
