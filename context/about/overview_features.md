# 1. Tổng Quan Ứng Dụng & Các Tính Năng Dành Cho Người Dùng (Munchin' Overview & User Features)

Tài liệu này cung cấp cái nhìn chi tiết về mục tiêu thiết kế, tính năng chức năng và trải nghiệm người dùng của ứng dụng **Munchin'** - ứng dụng theo dõi và quản lý dinh dưỡng thông minh sử dụng Trí tuệ Nhân tạo.

---

## 1. Mục Tiêu Dự Án (Project Goal)
Munchin' được xây dựng nhằm giải quyết bài toán kiểm soát năng lượng nạp vào cơ thể (Calorie Tracking) một cách tự động và tối giản nhất cho người dùng. Thay vì bắt người dùng phải tra cứu thủ công từng nguyên liệu và tự ước lượng cân nặng món ăn, Munchin' áp dụng các mô hình học sâu (Deep Learning) hàng đầu về thị giác máy tính (Computer Vision) để:
*   Nhận diện chính xác tên món ăn Việt Nam từ hình ảnh chụp thực tế.
*   Bóc tách từng thành phần nguyên liệu riêng biệt cấu thành món ăn.
*   Ước lượng thể tích và khối lượng (gam) của từng nguyên liệu bằng bản đồ chiều sâu 3D.
*   Quy đổi trực tiếp ra lượng Calorie và các chất đa lượng (Macronutrients) mà không cần tra cứu thủ công.

---

## 2. Các Tính Năng Dành Cho Người Dùng (User-Facing Features)

### A. Quy Trình Onboarding & Bộ Tính Toán Calorie Mục Tiêu (BMR & TDEE Calculator)
Khi người dùng truy cập ứng dụng lần đầu tiên, hệ thống sẽ dẫn dắt qua một biểu mẫu khảo sát sức khỏe trực quan:
*   **Thông tin thu thập:** Giới tính, Tuổi, Cân nặng (hỗ trợ chuyển đổi đơn vị kg $\leftrightarrow$ lbs), Chiều cao, Tần suất hoạt động thể chất (Ít vận động, Vận động nhẹ, Vận động vừa, Vận động nhiều, Vận động nặng) và Mục tiêu cá nhân (Giảm cân, Duy trì cân nặng, Tăng cân).
*   **Tính toán BMR (Basal Metabolic Rate - Tỷ lệ trao đổi chất cơ bản):** Hệ thống sử dụng công thức **Harris-Benedict** chuẩn để tính toán năng lượng tiêu thụ tối thiểu:
    *   *Nam giới:* $BMR = 88.362 + (13.397 \times W_{kg}) + (4.799 \times H_{cm}) - (5.677 \times A_{years})$
    *   *Nữ giới:* $BMR = 447.593 + (9.247 \times W_{kg}) + (3.098 \times H_{cm}) - (4.330 \times A_{years})$
*   **Tính toán TDEE (Total Daily Energy Expenditure - Tổng năng lượng tiêu thụ hàng ngày):** BMR được nhân với hệ số hoạt động thể chất tương ứng ($1.2 \dots 1.9$).
*   **Xác lập Calorie mục tiêu hàng ngày:**
    *   *Giảm cân:* $Target = TDEE - 500 \text{ kcal}$ (Giới hạn tối thiểu là 1200 kcal đối với Nữ và 1500 kcal đối với Nam để đảm bảo an toàn sức khỏe).
    *   *Tăng cân:* $Target = TDEE + 400 \text{ kcal}$.
    *   *Duy trì:* $Target = TDEE$.
*   **Tỷ lệ chất đa lượng (Macronutrients Ratio):** Lượng calo mục tiêu được phân bổ theo tỷ lệ chuẩn:
    *   **Carbohydrates (Carbs):** 45% tổng lượng calo ($1 \text{g Carbs} = 4 \text{ kcal}$).
    *   **Protein:** 25% tổng lượng calo ($1 \text{g Protein} = 4 \text{ kcal}$).
    *   **Fats (Chất béo):** 30% tổng lượng calo ($1 \text{g Fat} = 9 \text{ kcal}$).

### B. Hệ Thống Xác Thực Bảo Mật (Authentication & Security)
Hệ thống xác thực được thiết kế theo tiêu chuẩn Zero Trust để bảo vệ dữ liệu người dùng:
*   **Đăng ký đa chức năng:** Người dùng có thể đăng ký tài khoản mới bằng Username, Email và Mật khẩu. Mật khẩu được băm (hash) bằng thuật toán PBKDF2-HMAC-SHA256 với muối ngẫu nhiên (salt) 16-byte trước khi ghi vào cơ sở dữ liệu.
*   **Xác thực Email bằng mã OTP:** Ngay sau khi đăng ký thành công, tài khoản ở trạng thái chưa kích hoạt (`is_verified = False`). Hệ thống sẽ gửi một mã OTP ngẫu nhiên 6 chữ số có thời hạn 10 phút đến email người dùng. Người dùng phải nhập mã OTP này để kích hoạt tài khoản.
*   **Cơ chế Đăng nhập linh hoạt:** Hỗ trợ đăng nhập bằng cả tên người dùng (Username) hoặc địa chỉ email. Khi đăng nhập thành công, hệ thống sẽ cấp một token phiên làm việc (`session_token`) lưu trữ an toàn trong cơ sở dữ liệu và chuyển về phía client.
*   **Khôi phục mật khẩu an toàn (Forgot Password):** Khi yêu cầu đặt lại mật khẩu, mã OTP xác minh 6 số được gửi về email. Việc đổi mật khẩu thành công sẽ **vô hiệu hóa lập tức toàn bộ các session hoạt động cũ** của tài khoản đó để ngăn chặn hacker tiếp tục truy cập trái phép.

### C. Bảng Điều Khiển Nhật Ký Dinh Dưỡng Hàng Ngày (Dashboard & Meals Diary)
Giao diện Dashboard chính được thiết kế theo phong cách Glassmorphic hiện đại, cung cấp cái nhìn trực quan về cán cân năng lượng:
*   **Vòng tiến trình SVG (Progress Ring):** Biểu diễn tỷ lệ phần trăm Calo đã nạp vào so với mục tiêu trong ngày. Màu sắc chuyển đổi mượt màng theo mức độ nạp calo.
*   **Theo dõi chất đa lượng:** 3 thanh tiến trình nhỏ biểu diễn Carbs, Protein và Fats đã nạp so với định mức mục tiêu.
*   **Nhật ký bữa ăn (Meals Diary):** Phân chia khẩu phần ăn trong ngày thành 3 nhóm bữa ăn chính: Bữa sáng (Breakfast), Bữa trưa (Lunch) và Bữa tối (Dinner). Người dùng có thể:
    *   Thêm món ăn thủ công (chọn tên món ăn và khối lượng ước tính).
    *   Xóa món ăn khỏi nhật ký (hệ thống lập tức trừ chỉ số calo và dinh dưỡng theo thời gian thực).
    *   Sử dụng camera quét ảnh món ăn bằng AI.

### D. Hệ Thống Khám Phá Món Ăn & Đề Xuất Cá Nhân Hóa (Explore & Recommendations)
*   **Khám phá công thức nấu ăn:** Người dùng có thể duyệt qua danh sách các món ăn Việt Nam truyền thống (như Phở bò, Cơm tấm, Bún chả, Bánh mì...) để xem thành phần nguyên liệu tiêu chuẩn, thông tin dinh dưỡng trung bình trên 100g và hướng dẫn chế biến tốt cho sức khỏe.
*   **Flavor AI Recommendation Engine:** Hệ thống tích hợp một thuật toán lọc đề xuất thông minh:
    *   **Phân tích sự thiếu hụt calo/macros:** Đọc lượng Calo, Carbs, Protein và Fats còn thiếu trong ngày của người dùng.
    *   **Phân tích lịch sử bữa ăn:** Tránh đề xuất trùng lặp món ăn người dùng vừa ăn trong ngày.
    *   **Xếp hạng ứng viên món ăn (Ranking candidates):** Lọc danh sách món ăn từ cơ sở dữ liệu phù hợp với nhu cầu calo còn thiếu và sắp xếp theo điểm số phù hợp (Match Score).
