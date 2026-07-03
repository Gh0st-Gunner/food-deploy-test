# 5. Hướng Dẫn Kiểm Thử & Kịch Bản Test Cases (Testing Guide & Test Cases)

Tài liệu này đóng vai trò là cẩm nang kiểm thử thủ công và QA cho dự án Munchin', bao gồm hướng dẫn thiết lập môi trường, vận hành hệ thống container và danh sách 13 kịch bản kiểm thử chi tiết.

---

## 1. Thiết Lập Môi Trường & Khởi Chạy Hệ Thống

Để chuẩn bị môi trường kiểm thử, máy tính kiểm thử cần có sẵn **Docker Desktop** (hoặc Docker Engine) và kết nối internet để nạp trước mô hình từ HuggingFace (chỉ cần chạy lần đầu).

*   **Khởi động hệ thống:** Nhấp đúp chuột vào tệp tin **[start.bat](file:///c:/Users/Home/Desktop/vn%20food/start.bat)** ở thư mục gốc của dự án. File batch này sẽ thực hiện:
    1.  Kiểm tra Docker Desktop đã chạy chưa, nếu chưa sẽ tự động kích hoạt.
    2.  Kiểm tra dịch vụ Cloudflare Tunnel và khởi chạy một tunnel người dùng dưới nền (background user-space tunnel) nếu dịch vụ hệ thống chưa chạy.
    3.  Tự động kích hoạt toàn bộ các container dịch vụ bằng lệnh `docker compose up -d`.
    4.  Mở cổng dịch vụ giao diện chính tại địa chỉ: `http://localhost:10800` (hoặc tên miền ngoài `https://munchin.thegunner.uk`).
*   **Tải dữ liệu hình ảnh kiểm thử mẫu:** Bạn có thể tự động tải về 20 tệp ảnh món ăn truyền thống Việt Nam phục vụ việc kiểm thử bằng cách chạy lệnh:
    ```bash
    python scripts/download_test_images.py
    ```
    Các ảnh tải xuống sẽ được lưu vào thư mục `test-image/` (ví dụ: `banh-mi.jpg`, `pho.jpg`, `bun-bo-hue.jpg`...).
*   **Tắt hệ thống:** Nhấp đúp chuột vào tệp tin **[stop.bat](file:///c:/Users/Home/Desktop/vn%20food/stop.bat)** để dọn dẹp và tắt toàn bộ container chạy ẩn.

---

## 2. Danh Sách Các Kịch Bản Kiểm Thử (Test Cases)

### Nhóm A: Khảo Sát & Xác Thực (Onboarding & Authentication)

#### TC-01: Khảo sát Onboarding mới
*   **Các bước thực hiện:**
    1.  Truy cập giao diện tại `http://localhost:10800` trên trình duyệt.
    2.  Trình duyệt tự động chuyển hướng đến giao diện khảo sát. Chọn Giới tính $\to$ Tuổi $\to$ Cân nặng (kg/lbs) $\to$ Chiều cao $\to$ Tần suất hoạt động $\to$ Mục tiêu sức khỏe.
    3.  Nhấp chọn nút "Tiếp tục" để đến màn hình Đăng ký.
*   **Kết quả mong đợi:** Hệ thống lưu tạm các chỉ số khảo sát vào session/local storage. Nút "Tiếp tục" hoạt động trơn tru, hiển thị đúng màn hình Đăng ký tài khoản.

#### TC-02: Đăng ký tài khoản mới & Xác thực OTP Email
*   **Các bước thực hiện:**
    1.  Tại màn hình Đăng ký, điền Username, Mật khẩu và một địa chỉ Email hợp lệ.
    2.  Nhấp chọn nút "Đăng ký".
    3.  Xem console hoặc log stdout của máy chủ API (hoặc tệp tin logs của API container) để lấy mã xác thực OTP 6 số tự động được sinh ra.
    4.  Nhập mã OTP này vào ô xác nhận trên màn hình và ấn "Xác minh".
*   **Kết quả mong đợi:** Đăng ký thành công. Người dùng xác minh OTP thành công sẽ chuyển trạng thái `is_verified` thành `True` trong database, tự động đăng nhập và chuyển về trang Dashboard.

#### TC-03: Đăng nhập bằng tên tài khoản (Username)
*   **Các bước thực hiện:**
    1.  Nhấp nút "Đăng xuất" trên thanh menu để thoát khỏi phiên làm việc.
    2.  Tại màn hình đăng nhập, nhập chính xác Username và Mật khẩu vừa tạo.
    3.  Nhấp chọn "Đăng nhập".
*   **Kết quả mong đợi:** Đăng nhập thành công. Token phiên mới được ghi vào `localStorage` của trình duyệt và hiển thị đúng thông tin Dashboard người dùng.

#### TC-04: Đăng nhập bằng địa chỉ Email
*   **Các bước thực hiện:**
    1.  Nhấp nút "Đăng xuất".
    2.  Tại màn hình đăng nhập, nhập địa chỉ Email thay thế cho Username.
    3.  Nhập mật khẩu và nhấp chọn "Đăng nhập".
*   **Kết quả mong đợi:** Hệ thống tự động khớp địa chỉ email với tài khoản người dùng và đăng nhập thành công vào Dashboard.

#### TC-05: Khôi phục mật khẩu (Forgot Password)
*   **Các bước thực hiện:**
    1.  Tại màn hình Đăng nhập, nhấp chọn liên kết "Forgot Password?".
    2.  Nhập địa chỉ email của tài khoản đã đăng ký.
    3.  Lấy mã OTP đặt lại mật khẩu từ log của máy chủ API.
    4.  Nhập mã OTP và điền mật khẩu mới.
    5.  Thử đăng nhập lại bằng mật khẩu cũ (để kiểm tra xem có bị chặn không) và sau đó đăng nhập bằng mật khẩu mới.
*   **Kết quả mong đợi:** Mật khẩu cũ bị từ chối. Đăng nhập thành công bằng mật khẩu mới. Toàn bộ các token phiên hoạt động trước đó bị xóa bỏ khỏi cơ sở dữ liệu.

---

### Nhóm B: Bảng Điều Khiển & Nhật Ký Bữa Ăn (Dashboard & Meals Diary)

#### TC-06: Báo cáo dinh dưỡng, tiến trình & Bộ lọc thời gian biểu đồ
*   **Các bước thực hiện:**
    1.  Xem lượng Calorie mục tiêu trên Dashboard và biểu đồ báo cáo tiến trình (ở thẻ Weekly Calorie Intake).
    2.  Nhấp vào ô chọn thời gian (mặc định 7 ngày) trên biểu đồ để thay đổi khoảng thời gian lọc (1 ngày, 3 ngày, 7 ngày, 30 ngày).
    3.  Nhấp vào các chỉ số dinh dưỡng/sức khỏe khác nhau trên Dashboard (như Calo nạp vào, Cân nặng, Macros, Active Burn) để xem biểu đồ đổi màu chủ đề và hiển thị dữ liệu lịch sử/trung bình tương ứng.
    4.  Thay đổi thông tin chiều cao/cân nặng hoặc mục tiêu sức khỏe (ví dụ: chuyển từ Giảm cân sang Tăng cân).
*   **Kết quả mong đợi:** Lượng calo mục tiêu tăng thêm tương ứng ($\approx$ tăng thêm 900 kcal khi đổi từ Giảm cân sang Tăng cân). Biểu đồ vẽ đúng tỉ lệ dữ liệu, đổi màu sắc chủ đề tương ứng với chỉ số được chọn (ví dụ: Màu đỏ cam cho Calories, Xanh lá cho Carbs, Xanh dương cho Weight) và cập nhật số liệu trung bình (Avg) chính xác theo khoảng thời gian được lọc.

#### TC-07: Thêm món ăn thủ công vào nhật ký
*   **Các bước thực hiện:**
    1.  Tại Dashboard, nhấp chọn biểu tượng dấu cộng (+) tại danh mục "Bữa trưa".
    2.  Chọn một món ăn trong danh sách (ví dụ: Cơm tấm) và nhập khối lượng 400g. Nhấp chọn "Thêm".
*   **Kết quả mong đợi:** Món cơm tấm hiển thị trong danh mục bữa trưa. Chỉ số Calorie và Carbs/Protein/Fat trên Dashboard tự động cộng dồn và vòng tròn tiến trình tăng lên tương ứng.

#### TC-08: Xóa món ăn khỏi nhật ký
*   **Các bước thực hiện:**
    1.  Nhấp chọn biểu tượng thùng rác (xóa) cạnh món cơm tấm vừa thêm trong danh sách bữa trưa.
*   **Kết quả mong đợi:** Món ăn bị xóa khỏi danh sách bữa ăn ngay lập tức. Tổng lượng calo và macros trên bảng điều khiển tự động giảm trừ đi lượng dinh dưỡng của món ăn đó.

---

### Nhóm C: Bộ Quét Món Ăn AI (AI Scanner - Fast & Accurate Mode)

#### TC-09: Quét ảnh món ăn chế độ nhanh (Fast Mode)
*   **Các bước thực hiện:**
    1.  Truy cập tab Scan (biểu tượng Camera).
    2.  Tải lên một hình ảnh đĩa thức ăn từ thư mục `test-image/` (ví dụ: `banh-mi.jpg`).
    3.  Chọn chế độ quét **Fast Mode**.
    4.  Kiểm tra danh sách mô hình phân loại (mô hình mặc định được chọn sẵn là `eff_b0`).
    5.  Nhấp chọn "Start Analysis".
*   **Kết quả mong đợi:** Thời gian phản hồi nhanh chóng (dưới 3 giây) do hệ thống chỉ chạy mô hình phân loại (ONNX/PyTorch) mà không kích hoạt các mô hình phân đoạn và chiều sâu 3D. Kết quả hiển thị tên món ăn chính xác.

#### TC-10: Quét ảnh chế độ chính xác (Accurate Mode - 3D)
*   **Các bước thực hiện:**
    1.  Tải lên hình ảnh món ăn từ thư mục `test-image/` (ví dụ: `com-tam.jpg`).
    2.  Chọn chế độ quét **Accurate Mode** và nhập đường kính đĩa ăn tham chiếu (mặc định 25cm).
    3.  Nhấp chọn "Start Analysis".
    4.  Theo dõi logs của container `worker-detection` để xác nhận các mô hình DINO, SAM 2 và ZoeDepth hoạt động.
    5.  Khi kết quả hiển thị, nhấp vào nút "Xem ảnh phân tích" để bật hiển thị các mặt nạ nguyên liệu (SAM 2 mask overlay) vẽ đè lên ảnh. Nhấp lại lần nữa ("Xem ảnh gốc") để ẩn mặt nạ đi.
*   **Kết quả mong đợi:** Hệ thống phản hồi thành công sau 5-8 giây. Trả về kết quả phân tích: Nhận diện bát/đĩa cơm tấm, bóc tách các nguyên liệu (thịt heo, cơm, rau củ...), hiển thị ảnh gốc ban đầu với thông tin mặt nạ ẩn, cho phép bật/tắt hiển thị ảnh overlay vẽ viền màu quanh nguyên liệu, dựng bản đồ chiều sâu 3D và tính toán khối lượng chi tiết từng thành phần.

#### TC-11: Hiệu chỉnh góc nghiêng camera (Tilt Correction)
*   **Các bước thực hiện:**
    1.  Tải lên một hình ảnh đĩa thức ăn được chụp ở góc nghiêng lớn ($\approx 45^\circ$ đến $50^\circ$).
    2.  Chạy phân tích ở chế độ **Accurate Mode**.
    3.  Kiểm tra thuộc tính `tilt_ratio` và `measured_diameter_px` trả về trong chi tiết kết quả.
*   **Kết quả mong đợi:** Thuật toán OpenCV ellipse fitting phát hiện đúng hình dạng bầu dục của đĩa thức ăn, tính được tỷ lệ nghiêng camera và hiệu chỉnh diện tích thức ăn chính xác để không bị bóp méo khối lượng.

---

### Nhóm D: Khám Phá & Quản Trị (Explore & Admin Dashboard)

#### TC-12: Khám phá công thức & Gợi ý thực đơn thông minh
*   **Các bước thực hiện:**
    1.  Truy cập vào tab Explore.
    2.  Xem danh sách gợi ý đề xuất món ăn ở đầu trang.
    3.  Nạp một lượng thức ăn có hàm lượng calo cao vào Dashboard (để giảm lượng calo còn thiếu trong ngày xuống thấp).
    4.  Quay lại tab Explore để xem sự thay đổi của danh sách gợi ý món ăn.
*   **Kết quả mong đợi:** Danh sách đề xuất thay đổi thông minh. Hệ thống tự động lọc bỏ các món ăn có calo quá lớn vượt ngưỡng năng lượng còn thiếu của người dùng trong ngày, ưu tiên gợi ý các món ăn nhẹ, lành mạnh.

#### TC-13: Trang quản trị hệ thống (Admin Panel)
*   **Các bước thực hiện:**
    1.  Đăng xuất tài khoản hiện tại.
    2.  Đăng nhập bằng tài khoản Admin mặc định: Username: `admin`, Password: `admin123`.
    3.  Truy cập vào địa chỉ quản trị: `http://localhost:10800/admin.html`.
    4.  Kiểm tra hiển thị biểu đồ thống kê hệ thống và bảng quản lý người dùng. Thực hiện thử nghiệm khóa hoạt động của một tài khoản người dùng khác và thử dùng tài khoản đó đăng nhập lại.
*   **Kết quả mong đợi:** Hiển thị trang quản trị thành công với các biểu đồ thống kê tài khoản, phiên làm việc hoạt động và kết nối DB/Redis ở trạng thái "ok". Tài khoản bị khóa sẽ bị chặn không thể đăng nhập lại vào hệ thống.
