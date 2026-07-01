# HỆ THỐNG MUNCHIN' - DANH SÁCH TÍNH NĂNG CHI TIẾT (FEATURES LIST)

Tài liệu này tổng hợp toàn bộ các tính năng người dùng (User-facing Features) và các cơ chế kỹ thuật (Technical Features) được triển khai trong ứng dụng **Munchin' - Track Your Diet Journey** (Hệ thống phân loại món ăn Việt Nam & Ước lượng dinh dưỡng tự động).

---

## 1. Khảo sát & Khởi tạo Chỉ số Sức khỏe (User Onboarding & BMR Survey)
Giúp thu thập dữ liệu người dùng ban đầu nhằm cá nhân hóa hoàn toàn mục tiêu dinh dưỡng hàng ngày (Calories, Protein, Carbs, Fats) của mỗi cá nhân.

*   **Chọn Giới tính (Gender Selection)**: Giao diện trực quan chọn Nam (Male), Nữ (Female) hoặc Khác (Other) để tính toán tỷ lệ trao đổi chất cơ bản (BMR) tương ứng.
*   **Chọn Tuổi (Age Selection)**: Bộ chọn dạng cuộn tròn (Wheel Picker) mượt mà cho phép chọn độ tuổi chính xác.
*   **Cân nặng & Đơn vị (Weight & Units Toggle)**: Nhập cân nặng sử dụng thanh trượt (slider) và nút chuyển đổi đơn vị linh hoạt giữa `Kg` và `Lbs`.
*   **Mục tiêu Sức khỏe (Health Goals)**:
    *   *Giảm cân (Lose weight)*: Tự động thiết lập chế độ thâm hụt calo (calorie deficit).
    *   *Giữ cân (Maintain weight)*: Cân bằng năng lượng nạp và tiêu thụ.
    *   *Tăng cân (Gain weight)*: Tự động thiết lập chế độ thặng dư calo (calorie surplus).
*   **Tính toán BMR & TDEE Tự động**: Tích hợp công thức tính toán lượng calo mục tiêu dựa trên giới tính, cân nặng, độ tuổi và mục tiêu đã chọn.

*Mã nguồn liên quan*:
*   UI Onboarding Flow: [index.html:L31-182](file:///c:/Users/Home/Desktop/vn%20food/front-end/static/index.html#L31-182)
*   JS Logic: [index.js](file:///c:/Users/Home/Desktop/vn%20food/front-end/static/index.js)

---

## 2. Quản lý Tài khoản & Bảo mật (User Authentication & Session Management)
Cung cấp khả năng đăng ký, bảo mật tài khoản người dùng và lưu trữ dữ liệu cá nhân một cách an toàn.

*   **Đăng ký & Đăng nhập (Register/Login)**: Đăng ký nhanh bằng tên hiển thị, tên đăng nhập và mật khẩu được mã hóa an toàn.
*   **Quản lý phiên (Session Management)**: Sử dụng Token phiên tạm thời được lưu trữ và đính kèm trong Header yêu cầu (`Authorization: Bearer <token>`).
*   **Mật khẩu bảo mật**: Mật khẩu được mã hóa một chiều qua thuật toán băm bảo mật PBKDF2-HMAC-SHA256 trước khi lưu vào cơ sở dữ liệu.
*   **Phân quyền vai trò (Role-based Access Control - RBAC)**: Phân cấp rõ ràng giữa người dùng thông thường (`user`) và quản trị viên (`admin`).

*Mã nguồn liên quan*:
*   API Routes: [routes.py:L411-462](file:///c:/Users/Home/Desktop/vn%20food/back-end/api/routes.py#L411-462)
*   Auth Helpers: [auth.py](file:///c:/Users/Home/Desktop/vn%20food/back-end/api/auth.py)
*   User Models: [User](file:///c:/Users/Home/Desktop/vn%20food/back-end/core/database.py#L56) và [UserSession](file:///c:/Users/Home/Desktop/vn%20food/back-end/core/database.py#L67)

---

## 3. Bảng Điều khiển Dinh dưỡng Hàng ngày (Daily Nutrition Dashboard)
Trang chủ đóng vai trò là trung tâm theo dõi và ghi nhận chỉ số dinh dưỡng hàng ngày của người dùng.

*   **Date Slider & Bộ chọn Lịch (Calendar Modal)**: 
    *   Thanh trượt ngày nằm ngang giúp chuyển nhanh qua các ngày gần nhất.
    *   Tích hợp bộ chọn lịch tùy biến để xem hoặc ghi nhận lịch sử ăn uống của bất kỳ ngày nào trong quá khứ hoặc tương lai.
*   **Biểu đồ Calo dạng vòng bán nguyệt (Semi-Circular Calorie Gauge)**:
    *   Vẽ tiến trình nạp calo thời gian thực bằng SVG gradient đẹp mắt.
    *   Hiển thị chi tiết: **Budget** (Lượng calo định mức), **Left** (Calo còn lại), và **Exercise** (Calo bù trừ từ vận động).
*   **Thống kê chất dinh dưỡng đa lượng (Macronutrient Progress)**:
    *   Theo dõi tỉ lệ và khối lượng Protein, Fats, và Carbs đã nạp so với mục tiêu đề ra qua các thanh tiến trình nhiều màu sắc.
*   **Nhật ký bữa ăn theo buổi (Meal Category Tabs)**:
    *   Hiển thị các bữa ăn đã lưu phân chia theo: **Bữa sáng (Breakfast)**, **Bữa trưa (Lunch)**, và **Bữa tối (Dinner)**.
    *   Cho phép xóa bữa ăn cũ ra khỏi hành trình hoặc xem chi tiết.
*   **Nhật ký cân nặng và hoạt động thể chất nhanh**: Hiển thị nhanh các chỉ số cân nặng hiện tại trong ngày và các bài tập đã hoàn thành.

*Mã nguồn liên quan*:
*   UI Dashboard: [index.html:L274-410](file:///c:/Users/Home/Desktop/vn%20food/front-end/static/index.html#L274-410)
*   JS Render & Calculations: [index.js](file:///c:/Users/Home/Desktop/vn%20food/front-end/static/index.js)

---

## 4. Quét Ảnh Món Ăn Bằng AI (AI Food Scanner)
Tính năng cốt lõi của ứng dụng, cho phép chụp hoặc tải ảnh đĩa thức ăn lên để AI tự động phân tích chi tiết.

*   **Chụp ảnh / Tải tệp lên (Drag & Drop Zone)**: Vùng nhận diện thả ảnh hoặc chạm để kích hoạt camera điện thoại tiện lợi.
*   **Lựa chọn Mô hình AI (Model Selector)**: Người dùng có thể chọn sử dụng mô hình phù hợp trong danh sách:
    *   Các mô hình AI cục bộ dạng ONNX/PyTorch (EfficientNet-B0, ResNet-50).
    *   Mô hình Cloud tiên tiến (Google Gemini `gemini-1.5-flash`).
    *   Mô hình Ollama cục bộ (LLaVA, LLaVA-Phi3, BakLLaVA) được quét tự động qua API của Ollama.
*   **Quy trình Xử lý Ảnh Chế độ Chính xác (Accurate Mode)**:
    1.  **Phân loại Món ăn (Food Classification)**: Phân loại món ăn tổng thể (như Phở bò, Bánh mì, Bún chả...) thông qua mô hình đã chọn.
    2.  **Tra cứu Dinh dưỡng Cơ sở (USDA/FatSecret Lookup)**: Tra cứu nhanh lượng calo/macros định mức của món ăn từ các nguồn uy tín.
    3.  **Phát hiện Nguyên liệu (Ingredient Detection - Grounding DINO)**: Nhận diện tọa độ Bounding Box của từng nguyên liệu nhỏ có trong đĩa thức ăn (ví dụ: thịt bò, hành lá, bánh phở, giá đỗ).
    4.  **Cắt Phân vùng Chi tiết (Segmentation - SAM 2)**: Cắt viền đa giác cực kỳ chính xác cho từng thành phần nguyên liệu. Kết quả được lưu dưới dạng mặt nạ (mask) đè lên ảnh gốc trên Frontend.
    5.  **Tính toán Khẩu phần (Portion Area-Ratio)**: So sánh diện tích mặt nạ của nguyên liệu với đĩa chuẩn 25cm để ước lượng khối lượng (gram) thực tế của món ăn.
    6.  **Bản đồ Độ sâu (Depth Map V2)**: Sử dụng mô hình `Depth Anything V2` ước lượng chiều sâu 3D của đĩa ăn để hỗ trợ hiển thị cấu trúc.
    7.  **Hội tụ Dinh dưỡng (Nutrients Aggregation)**: Tính toán tổng lượng calo và macros thực tế của cả đĩa ăn dựa trên khối lượng nguyên liệu ước lượng được.
*   **Xử lý Bất đồng bộ & WebSocket Streaming**:
    *   Tác vụ nặng được gửi vào hàng đợi xử lý bất đồng bộ Celery (hoặc FastAPI BackgroundTasks fallback nếu Redis Broker ngoại tuyến).
    *   Cập nhật tiến trình phân tích theo thời gian thực (0% - 100%) trực tiếp lên giao diện người dùng thông qua kết nối **WebSocket**.

*Mã nguồn liên quan*:
*   API Analyze: [analyze](file:///c:/Users/Home/Desktop/vn%20food/back-end/api/routes.py#L65)
*   WebSocket Server: [websocket_jobs_stream](file:///c:/Users/Home/Desktop/vn%20food/back-end/api/routes.py#L340)
*   AI Pipeline Workers: [workers](file:///c:/Users/Home/Desktop/vn%20food/back-end/workers/)
*   Database Schema: [Job](file:///c:/Users/Home/Desktop/vn%20food/back-end/core/database.py#L13)

---

## 5. Khám Phá & Gợi Ý Món Ăn Thông Minh (Explore & Recommendation Engine)
Hỗ trợ người dùng trong việc tìm kiếm các món ăn tốt cho sức khỏe và gợi ý các bữa ăn khoa học tiếp theo.

*   **Nhập món ăn thủ công (Manual Log)**: Cho phép ghi chép nhanh các món ăn tùy chỉnh bằng cách nhập tay Tên món, Calories và Macros tương ứng.
*   **Khám phá món ăn tốt cho sức khỏe (AI Scraped Dishes)**:
    *   Sử dụng API tìm kiếm web của Ollama (`https://ollama.com/api/web_search`) để cào dữ liệu thực tế các món ăn lành mạnh theo mục tiêu dinh dưỡng vĩ mô của người dùng.
    *   Hiển thị thông tin bao gồm: Tiêu đề món, mô tả, hàm lượng dinh dưỡng, nguyên liệu và hướng dẫn chế biến chi tiết.
*   **Bộ Lên Kế Hoạch Bữa Ăn (AI Recipe Generator)**:
    *   Người dùng chỉ cần nhập các nguyên liệu hiện có trong tủ lạnh (ví dụ: `thịt gà, gừng, nấm`), AI sẽ tự động sáng tạo ra công thức nấu ăn chuẩn Việt, tính calo phù hợp mục tiêu.
*   **Hệ thống Gợi ý Cá nhân hóa (Flavor AI Engine)**:
    *   Xếp hạng (Rank) và đánh giá các món ăn dựa trên độ tương đồng của chất dinh dưỡng đa lượng (cosine similarity).
    *   **Cơ chế Tránh lặp món (Fatigue Penalty)**: Tự động phân tích các món ăn đã nạp gần đây của người dùng. Nếu phát hiện người dùng ăn quá nhiều một loại đạm/hương vị (ví dụ: Bò, Gà, Heo, Trứng, Tôm/Hải sản, Đậu hũ), hệ thống sẽ phạt điểm giảm độ ưu tiên (tối đa giảm 35% điểm) nhằm khuyến khích đổi vị và cân bằng dinh dưỡng.
    *   **Lời khuyên dinh dưỡng bằng tiếng Việt (Rationales)**: Đưa ra nhận xét cụ thể vì sao món ăn này phù hợp với bạn (ví dụ: *"Tỉ lệ dinh dưỡng (Macros) khớp hoàn hảo...", "Cung cấp lượng Protein dồi dào...", "Giúp thay đổi khẩu vị mới mẻ so với món heo bạn ăn gần đây"*).

*Mã nguồn liên quan*:
*   Explore Routes: [explore](file:///c:/Users/Home/Desktop/vn%20food/back-end/api/routes.py#L279) và [explore_recommend](file:///c:/Users/Home/Desktop/vn%20food/back-end/api/routes.py#L315)
*   Scraper & Generator: [explore_scraper.py](file:///c:/Users/Home/Desktop/vn%20food/back-end/api/explore_scraper.py)
*   Flavor Recommendation: [recommendation_engine.py](file:///c:/Users/Home/Desktop/vn%20food/back-end/api/recommendation_engine.py)

---

## 6. Ghi Nhận Hoạt Động & Cân Nặng (Activity & Weight Logging)
Hệ thống quản lý lượng calo tiêu thụ từ vận động và lịch sử theo dõi thể trạng cơ thể.

*   **Nhật ký Cân nặng (Weight Log)**: Ghi chép cân nặng hàng ngày và xem lại danh sách lịch sử đo đếm theo thời gian để dễ dàng kiểm soát quá trình tăng/giảm cân.
*   **Nhật ký Hoạt động (Exercise Log)**:
    *   Hỗ trợ ghi nhanh các loại bài tập phổ biến: Đi bộ (Walking), Chạy bộ (Running), Đạp xe (Cycling), Bơi lội (Swimming), Tập tạ (Strength Training), Yoga, hoặc Tự chọn (Custom).
    *   Tự động tính toán lượng Calo đốt cháy dựa trên tỷ lệ chuẩn hóa cho mỗi phút vận động.
    *   Cho phép ghi chép các bài tập tự thiết kế với lượng calo đốt cháy nhập tay.
*   **Xóa bản ghi**: Người dùng có thể xóa các bản ghi cân nặng hoặc vận động bị nhầm lẫn trực tiếp trên giao diện qua các hộp thoại xác nhận (Confirm Modals).

*Mã nguồn liên quan*:
*   Modals UI: [index.html:L875-945](file:///c:/Users/Home/Desktop/vn%20food/front-end/static/index.html#L875-945)
*   Modals Logic: [index.js](file:///c:/Users/Home/Desktop/vn%20food/front-end/static/index.js)

---

## 7. Báo Cáo Tiến Trình & Nhận Định Dinh Dưỡng (Diet Progress Reports & Insights)
Cung cấp cái nhìn trực quan và lời khuyên hữu ích để người dùng duy trì lối sống lành mạnh.

*   **Biểu đồ intake hàng tuần (Chart.js Line Chart)**:
    *   Vẽ biểu đồ đường biểu diễn xu hướng calo nạp vào hàng ngày trong tuần.
    *   Hiển thị lượng calo nạp trung bình (Average kcal) trực tiếp trên biểu đồ.
*   **Nhận định Dinh dưỡng (Nutrition Insights)**:
    *   *Chuỗi Đạt Mục Tiêu Calo (Calorie Goal Streak)*: Đưa ra phản hồi tích cực khi duy trì nạp calo ổn định.
    *   *Cân Bằng Dinh Dưỡng (Macronutrient Balance)*: Nhắc nhở người dùng nạp đủ protein nếu lượng protein thực tế chưa chạm mốc mục tiêu.

*Mã nguồn liên quan*:
*   Reports Panel: [index.html:L674-708](file:///c:/Users/Home/Desktop/vn%20food/front-end/static/index.html#L674-708)
*   Chart Configuration: [index.js](file:///c:/Users/Home/Desktop/vn%20food/front-end/static/index.js)

---

## 8. Bảng Quản Trị Hệ Thống (Admin User & System Console)
Giao diện quản lý hoàn toàn độc lập dành riêng cho Quản trị viên (Admin) để giám sát trạng thái toàn bộ phần mềm.

*   **Thống kê thời gian thực (Real-time Stats)**:
    *   Tổng số người dùng đã đăng ký.
    *   Tổng số phiên hoạt động hiện tại (Active Sessions).
    *   Tổng số yêu cầu AI đã xử lý (Tổng số Jobs, Jobs thành công, Jobs lỗi).
    *   Trạng thái kết nối Cơ sở dữ liệu (Database Health).
*   **Bảng quản lý người dùng (User Management Table)**:
    *   Liệt kê toàn bộ người dùng kèm theo ngày tạo và vai trò.
    *   **Thêm người dùng mới**: Cho phép tạo nhanh tài khoản người dùng/quản trị viên mới.
    *   **Chỉnh sửa thông tin**: Sửa tên đăng nhập, thay đổi mật khẩu hoặc cập nhật vai trò trực tiếp.
    *   **Khóa / Mở khóa tài khoản (Block/Unblock Status)**: Ngắt kích hoạt tài khoản của người dùng vi phạm. Sau khi khóa, toàn bộ các phiên hoạt động (sessions) của người dùng đó sẽ bị xóa lập tiếp để ngăn cản việc truy cập hệ thống.
    *   **Xóa người dùng (Delete User)**: Xóa vĩnh viễn tài khoản (ngoại trừ tài khoản `admin` mặc định hoặc chính quản trị viên đang đăng nhập).

*Mã nguồn liên quan*:
*   Admin UI: [admin.html](file:///c:/Users/Home/Desktop/vn%20food/front-end/static/admin.html)
*   Admin Style: [admin.css](file:///c:/Users/Home/Desktop/vn%20food/front-end/static/admin.css)
*   Admin Logic: [admin.js](file:///c:/Users/Home/Desktop/vn%20food/front-end/static/admin.js)
*   Admin API: [routes.py:L465-593](file:///c:/Users/Home/Desktop/vn%20food/back-end/api/routes.py#L465-593)

---

## 9. Thiết Kế Trực Quan & Trải Nghiệm (Aesthetics & Dark Mode)
*   **Pitch-Black Dark Mode**: Chuyển đổi giao diện sang chế độ tối hoàn toàn (AMOLED Black) giúp tiết kiệm pin và dễ nhìn hơn vào ban đêm.
*   **Thiết kế Phong cách Figma Di động**: Toàn bộ ứng dụng được bọc trong một bộ khung mô phỏng giao diện điện thoại thông minh cực kỳ mượt mà, màu sắc được phối hợp hài hòa bằng các biến CSS (variables).
*   **Micro-animations**: Hiệu ứng chuyển động mượt mà khi di chuột, mở các ô modal, hay khi tải dữ liệu phân tích.
