# USER EXPERIENCE TEST CASES & MANUAL TESTING GUIDE

Tài liệu này cung cấp hướng dẫn thiết lập môi trường và danh sách các kịch bản kiểm thử (Test Cases) chi tiết cho tất cả các tính năng trên ứng dụng Munchin'.

---

## 1. HƯỚNG DẪN THIẾT LẬP & KHỞI CHẠY HỆ THỐNG

Trước khi tiến hành kiểm thử, hãy đảm bảo hệ thống đã được khởi chạy đầy đủ theo các bước sau:

### Bước 1: Khởi động toàn bộ hệ thống
Đảm bảo phần mềm Docker Desktop đã được mở. Nhấp đúp vào tệp tin **[start.bat](file:///c:/Users/Home/Desktop/vn%20food/start.bat)** ở thư mục gốc để khởi chạy tất cả các dịch vụ (bao gồm cơ sở dữ liệu, Redis cache, MinIO S3, FastAPI gateway và các Celery workers).

### Bước 2: Dừng toàn bộ hệ thống khi hoàn tất
Sau khi thực hiện xong kiểm thử, nhấp đúp vào tệp tin **[stop.bat](file:///c:/Users/Home/Desktop/vn%20food/stop.bat)** ở thư mục gốc để dừng toàn bộ container và dọn dẹp các tài nguyên nền.

---

## 2. DANH SÁCH CÁC KỊCH BẢN KIỂM THỬ (TEST CASES)

### A. Nhóm tính năng khảo sát & xác thực (Onboarding & Authentication)

| ID | Tính năng kiểm thử | Các bước thực hiện | Kết quả mong đợi |
| :--- | :--- | :--- | :--- |
| **TC-01** | Khảo sát Onboarding mới | 1. Truy cập `http://localhost:10800` trên trình duyệt.<br>2. Chọn Giới tính $\to$ Tuổi $\to$ Cân nặng (kg/lbs) $\to$ Mục tiêu.<br>3. Ấn nút Tiếp tục để đến màn hình Đăng ký. | Hệ thống lưu tạm thời các chỉ số khảo sát. Nút tiếp tục hoạt động mượt mà và chuyển màn hình chính xác. |
| **TC-02** | Đăng ký có xác thực Email | 1. Nhập Username, Mật khẩu và một địa chỉ Email hợp lệ.<br>2. Nhấp chọn Đăng ký.<br>3. Kiểm tra terminal của `run_local.bat` để lấy mã xác thực OTP 6 số.<br>4. Nhập mã OTP vào ô xác nhận trên màn hình. | Tài khoản được tạo thành công với trạng thái `is_verified = True`. Người dùng được tự động đăng nhập vào Dashboard. |
| **TC-03** | Đăng nhập bằng Username | 1. Đăng xuất khỏi hệ thống.<br>2. Tại màn hình đăng nhập, nhập Username và Mật khẩu vừa tạo.<br>3. Nhấp Đăng nhập. | Đăng nhập thành công, token được lưu vào `localStorage`, hiển thị đúng giao diện Dashboard. |
| **TC-04** | Đăng nhập bằng Email | 1. Đăng xuất.<br>2. Tại màn hình đăng nhập, nhập địa chỉ Email thay thế cho Username.<br>3. Nhập Mật khẩu và ấn Đăng nhập. | Hệ thống khớp thông tin và đăng nhập thành công vào Dashboard. |
| **TC-05** | Khôi phục mật khẩu (Forgot) | 1. Tại màn hình Đăng nhập, nhấp "Forgot Password".<br>2. Nhập email của tài khoản đã đăng ký.<br>3. Lấy mã OTP đặt lại mật khẩu từ console của `run_local.bat`.<br>4. Nhập mã OTP và mật khẩu mới.<br>5. Đăng nhập lại bằng mật khẩu mới. | Mật khẩu được cập nhật thành công. Người dùng đăng nhập được bằng mật khẩu mới. Các phiên làm việc cũ bị vô hiệu hóa. |

---

### B. Bảng điều khiển & Nhập nhật ký (Dashboard & Meals Diary)

| ID | Tính năng kiểm thử | Các bước thực hiện | Kết quả mong đợi |
| :--- | :--- | :--- | :--- |
| **TC-06** | Calorie Target & Progress Ring | 1. Kiểm tra chỉ số calo mục tiêu hiển thị trên Dashboard.<br>2. Thay đổi mục tiêu sức khỏe trong Onboarding (Tăng cân/Giảm cân) và kiểm tra lại Calo mục tiêu. | Lượng calo mục tiêu thay đổi chính xác tương ứng với mục tiêu (Tăng cân $\approx$ +400kcal, Giảm cân $\approx$ -500kcal). Vòng tròn tiến trình SVG vẽ đúng tỉ lệ. |
| **TC-07** | Thêm món ăn thủ công | 1. Trên Dashboard, nhấp biểu tượng dấu cộng (+) tại nhóm bữa ăn.<br>2. Chọn món ăn và khối lượng.<br>3. Nhấp Thêm. | Món ăn hiển thị trong danh mục bữa ăn. Thanh tiến trình Calo và Carbs/Protein/Fat tự động cập nhật tăng lên tương ứng. |
| **TC-08** | Xóa món ăn khỏi nhật ký | 1. Nhấp nút xóa (biểu tượng thùng rác/dấu trừ) cạnh món ăn trong danh sách bữa ăn. | Món ăn biến mất khỏi danh sách. Chỉ số calo và dinh dưỡng trên Dashboard tự động giảm trừ ngay lập tức. |

---

### C. Quét món ăn AI (AI Scanner - Camera & Estimator)

| ID | Tính năng kiểm thử | Các bước thực hiện | Kết quả mong đợi |
| :--- | :--- | :--- | :--- |
| **TC-09** | Quét ảnh món ăn chế độ Fast | 1. Truy cập tab Scan (biểu tượng camera ở giữa thanh điều hướng).<br>2. Tải lên một hình ảnh đĩa thức ăn (ví dụ: đĩa cơm tấm thịt sườn).<br>3. Chọn chế độ "Fast Mode".<br>4. Ấn "Start Analysis". | API khởi động tác vụ bất đồng bộ. Kết quả trả về tên món ăn nhanh chóng nhờ mô hình ONNX/PyTorch tối ưu phân loại. |
| **TC-10** | Quét ảnh chế độ Accurate (3D) | 1. Tải lên hình ảnh thức ăn.<br>2. Chọn chế độ "Accurate Mode" và điền đường kính đĩa chuẩn (mặc định 25cm).<br>3. Ấn "Start Analysis" và theo dõi tiến trình chạy của `run_worker.bat`. | Celery worker chạy thành công Grounding DINO, cắt phân vùng SAM 2, dựng bản đồ chiều sâu GPU Depth Anything V2 và ước lượng thể tích đĩa ăn. Kết quả trả về khối lượng (g) chi tiết. |
| **TC-11** | Hiệu chỉnh góc nghiêng (Tilt) | 1. Tải lên ảnh chụp đĩa thức ăn ở góc nghiêng lớn (~45 độ).<br>2. Chạy quét ở chế độ Accurate. | OpenCV khớp thành công Ellipse đĩa ăn. Trục lớn và trục nhỏ khớp tỉ lệ góc nghiêng chính xác, hiệu chỉnh diện tích thức ăn để không bị sai lệch khối lượng thực tế. |

---

### D. Gợi ý thực đơn & Quản trị (Explore & Admin Management)

| ID | Tính năng kiểm thử | Các bước thực hiện | Kết quả mong đợi |
| :--- | :--- | :--- | :--- |
| **TC-12** | Đề xuất món ăn thông minh | 1. Truy cập tab Explore.<br>2. Xem danh sách gợi ý món ăn thông minh dựa trên calo còn lại trong ngày của bạn. | Thuật toán gợi ý trả về các món ăn phù hợp với lượng calo mục tiêu còn thiếu của người dùng. |
| **TC-13** | Quản trị tài khoản (Admin) | 1. Đăng nhập bằng tài khoản Admin mặc định (`admin`/`admin123`).<br>2. Truy cập trang quản trị tại `http://localhost:10800/admin.html`. | Hiển thị bảng điều khiển quản trị viên với đầy đủ thống kê người dùng, phiên hoạt động, trạng thái kết nối DB/Redis và nút khóa/mở khóa tài khoản người dùng. |
