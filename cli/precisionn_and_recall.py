'''
1. Dùng PRECISION khi bạn muốn GIẢM RÁC (Less Junk)
Câu hỏi cốt lõi: "Trong những gì tìm được, có bao nhiêu cái đúng?

- "Khi nào dùng: Khi trải nghiệm người dùng phụ thuộc vào việc 
    kết quả hiển thị phải cực kỳ sạch sẽ và chuẩn xác. 
    Người dùng chỉ xem trang đầu (Top K) nên họ không muốn 
    thấy kết quả sai lầm, lạc đề.
    
- Ví dụ: Search Engine thương mại điện tử, Google Search, 
hoặc ứng dụng tìm phim giải trí thông thường.




2. Dùng RECALL khi bạn muốn ĐỦ, KHÔNG SÓT (Completeness)
Câu hỏi cốt lõi: "Trong những cái đúng có trong kho, 
    mình đã tìm ra được bao nhiêu cái?
    
- "Khi nào dùng:Trong các ứng dụng nhạy cảm, việc bỏ sót thông tin 
    quan trọng sẽ gây hậu quả nghiêm trọng (Dù có lẫn một chút kết quả 
    rác cũng chấp nhận được, miễn là không sót).
    Trong giai đoạn đầu (Retrieval) của hệ thống multi-stage RAG. 
    Nếu bước đầu bạn lấy thiếu dữ liệu (Low Recall), 
    thì mô hình Re-ranker đắt tiền ở sau cũng không thể 
    sửa chữa được vì dữ liệu đúng đã bị bỏ lại.
    
- Ví dụ: Tìm kiếm tài liệu y khoa để chữa bệnh, 
    tra cứu hồ sơ pháp lý tòa án, phát hiện lỗi bảo mật hệ thống.
    
    
    
    
Dùng F1-SCORE khi bạn muốn CÂN BẰNG, KHÔNG LỆCH (Harmonic Balance)
Câu hỏi cốt lõi: "Hệ thống tìm kiếm tổng thể đang hoạt động 
    tốt đến mức nào khi phải vừa giữ kết quả sạch (Precision) 
    vừa không bỏ sót tài liệu (Recall)?
    
- "Khi nào dùng:Trong các ứng dụng mà Precision và Recall có vai trò 
    quan trọng ngang nhau, bạn không muốn tối ưu một bên mà làm bên 
    còn lại tệ đi thảm hại.
    Khi bạn cần một con số duy nhất (Single-number metric) 
    để chấm điểm nhanh, so sánh và đánh giá hiệu năng giữa các thuật toán, 
    mô hình Embedding, hoặc các chiến lược Prompt Expand khác nhau.
    Khi bạn muốn phạt nặng những hệ thống cực đoan (ví dụ: hệ thống 
    chỉ lấy bừa 1 kết quả đúng duy nhất để đạt Precision 100% nhưng 
    Recall gần như bằng 0%, hoặc hệ thống vớt sạch cả database 
    để đạt Recall 100% nhưng tràn ngập rác).
- Ví dụ: Đánh giá chất lượng 
    bộ máy tìm kiếm phim (Movie Search) của bạn, 
    kiểm thử chatbot RAG trả lời tự động cho khách hàng, 
    tối ưu bộ phân loại thư rác (Spam Filter).
'''
