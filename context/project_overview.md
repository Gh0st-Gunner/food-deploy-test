# PROJECT OVERVIEW: MUNCHIN' APP

Munchin' là một ứng dụng Web toàn diện hỗ trợ người dùng theo dõi và quản lý dinh dưỡng hàng ngày. Ứng dụng tích hợp công nghệ Trí tuệ Nhân tạo (AI) tiên tiến để phân tích món ăn từ hình ảnh, tự động bóc tách các thành phần nguyên liệu, tính toán kích thước khẩu phần dưới dạng mô hình 3D và hiển thị chi tiết hàm lượng dinh dưỡng.

---

## 1. TÍNH NĂNG DÀNH CHO NGƯỜI DÙNG (USER-FACING FEATURES)

### A. Khảo sát & Tính toán Calorie mục tiêu (Onboarding & BMR Calculator)
*   **Khảo sát thông tin**: Thu thập giới tính, tuổi, cân nặng (hỗ trợ kg/lbs), chiều cao và mục tiêu sức khỏe (Giảm cân, Duy trì, Tăng cân).
*   **Tính toán tự động**: Áp dụng công thức Harris-Benedict để tính chỉ số BMR và TDEE, từ đó tự động xác lập mục tiêu Calo hàng ngày cùng tỷ lệ các chất đa lượng (Carbs: 45%, Protein: 25%, Fats: 30%).

### B. Hệ thống Xác thực Bảo mật (Authentication & Security)
*   **Đăng ký & Đăng nhập**: Hỗ trợ đăng nhập linh hoạt bằng cả Username hoặc địa chỉ Email.
*   **Xác thực Email**: Gửi mã OTP xác nhận tài khoản ngay sau khi đăng ký để nâng cao tính bảo mật.
*   **Quên mật khẩu (Forgot Password)**: Nhận mã OTP qua email để đặt lại mật khẩu mới, đồng thời tự động vô hiệu hóa tất cả các phiên làm việc hiện hoạt để ngăn chặn truy cập trái phép.

### C. Bảng điều khiển Dinh dưỡng Hàng ngày (Daily Calorie Dashboard)
*   **Theo dõi trực quan**: Hiển thị biểu đồ vòng cung tiến trình calo nạp vào so với mục tiêu trong ngày.
*   **Thống kê chất đa lượng**: Theo dõi chi tiết lượng Carbs, Protein và Fats đã nạp theo thời gian thực.
*   **Nhật ký bữa ăn**: Liệt kê các món ăn đã ăn theo từng nhóm bữa ăn (Bữa sáng, Bữa trưa, Bữa tối).

### D. Bộ quét món ăn AI thông minh (AI Scanner Panel)
*   **Nhận diện & Bóc tách (Classification & Detection)**: Nhận diện tên món ăn Việt Nam và bóc tách các thành phần nguyên liệu cấu thành (ví dụ: bún, thịt bò, rau thơm trong bát Bún bò Huế).
*   **Ước lượng khẩu phần 3D (Portion Sizing)**:
    *   *Plate Ellipse Fitting*: Phát hiện hình dạng đĩa/bát để hiệu chỉnh góc nghiêng camera và tính toán tỉ lệ pixel-sang-cm.
    *   *ZoeDepth*: Dựng bản đồ chiều sâu 3D của đĩa thức ăn, tính toán độ cao vật lý của từng thành phần và tích hợp thể tích ($Volume = Area \times Height$) để suy ra khối lượng (gam).
*   **Tra cứu Dinh dưỡng tự động**: Kết nối với API USDA và FatSecret để quy đổi khối lượng món ăn thành chỉ số calo và dinh dưỡng chính xác nhất.

### E. Gợi ý Thực đơn & Khám phá Công thức (Explore & Recommendation)
*   **Khám phá món ăn**: Tra cứu công thức nấu ăn, thành phần nguyên liệu và thông tin dinh dưỡng của hàng trăm món ăn Việt Nam truyền thống.
*   **Đề xuất cá nhân hóa (Recommendation Engine)**: Phân tích lịch sử ăn uống gần đây và mục tiêu calo để đưa ra gợi ý món ăn thông minh giúp tối ưu hóa tiến trình dinh dưỡng của người dùng.

---

## 2. QUẢN TRỊ VIÊN (ADMIN DASHBOARD)
*   **Thống kê tổng quan**: Xem tổng số người dùng, số lượng phiên hoạt động và tổng số tác vụ phân tích AI trong hệ thống.
*   **Giám sát hiệu năng**: Theo dõi trạng thái kết nối Cơ sở dữ liệu và Redis Cache.
*   **Quản lý người dùng**: Xem danh sách tài khoản, ngày tham gia, vai trò (User/Admin) và quyền kích hoạt/khóa tài khoản.

---

## 3. CÔNG NGHỆ & THƯ VIỆN SỬ DỤNG (TECH STACK & LIBRARIES)

### Backend API
*   **FastAPI**: Web framework tốc độ cao, hỗ trợ tài liệu hóa API tự động bằng Swagger.
*   **SQLAlchemy**: Công cụ ORM quản lý mô hình dữ liệu, kết nối PostgreSQL/SQLite.
*   **Celery**: Hệ thống hàng đợi tác vụ bất đồng bộ phân tán để xử lý các mô hình ML nặng.

### AI / Deep Learning Models
*   **PyTorch & Torchvision**: Thư viện nền tảng chạy suy diễn mô hình AI.
*   **Grounding DINO**: Phát hiện vật thể dạng Zero-shot để bóc tách nguyên liệu từ văn bản đầu vào.
*   **SAM 2 (Segment Anything Model 2)**: Cắt phân vùng (segmentation) chính xác các thành phần món ăn theo tọa độ hộp giới hạn (bounding box).
*   **ZoeDepth**: Mô hình ước lượng chiều sâu đơn ảnh kết hợp đo khoảng cách vật lý (metric depth estimation) để xây dựng mô hình 3D cho khẩu phần ăn.

### Frontend Client
*   **Vanilla HTML5 & CSS3**: Xây dựng giao diện Responsive, hiệu ứng Glassmorphic hiện đại, thiết kế tối giản, tông màu tối/sáng hài hòa.
*   **Vanilla JavaScript (ES6)**: Xử lý tương tác SPA (Single Page Application), biểu đồ báo cáo tiến trình (Calories, Weight, Macros, Active Burn) tích hợp bộ lọc thời gian động (1, 3, 7, 30 ngày), hỗ trợ cơ chế bật/tắt mặt nạ nguyên liệu phân tích AI (mặc định ẩn khi tải trang kết quả), kết nối Websocket theo dõi tiến trình AI.
