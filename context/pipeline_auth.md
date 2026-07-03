# Pipeline Xác Thực & Quản Lý Phiên Làm Việc (Authentication & Session Management)

Tài liệu này đặc tả chi tiết về hệ thống xác thực người dùng, bảo mật mật khẩu, xác minh email qua mã OTP, và cơ chế quản lý phiên làm việc trong dự án Munchin'.

---

## 1. Các Tính Năng & Trường Hợp Sử Dụng (Use Cases)

Hệ thống cung cấp các chức năng bảo mật cho hai đối tượng chính: **Thành viên (User)** và **Quản trị viên (Admin)**.

### Biểu đồ Use Case tổng quan
```plantuml
@startuml
left to right direction
skinparam packageStyle rectangle

actor "Người dùng (User)" as user
actor "Quản trị viên (Admin)" as admin
actor "Hệ thống Email" as email_svc

rectangle "Hệ thống Xác thực & Bảo mật (Auth System)" {
    usecase "Đăng ký tài khoản mới" as UC_register
    usecase "Nhận mã OTP xác thực" as UC_send_otp
    usecase "Xác thực email qua OTP" as UC_verify_email
    usecase "Đăng nhập (Username/Email)" as UC_login
    usecase "Đăng xuất tài khoản" as UC_logout
    usecase "Yêu cầu khôi phục mật khẩu" as UC_forgot_pass
    usecase "Đặt lại mật khẩu mới" as UC_reset_pass
    
    usecase "Quản lý tài khoản người dùng" as UC_manage_users
    usecase "Khóa / Mở khóa tài khoản" as UC_block_user
}

user --> UC_register
user --> UC_verify_email
user --> UC_login
user --> UC_logout
user --> UC_forgot_pass
user --> UC_reset_pass

UC_register ..> UC_send_otp : <<include>>
UC_forgot_pass ..> UC_send_otp : <<include>>
UC_send_otp --> email_svc : Gửi OTP thực tế/mock

admin --> UC_manage_users
admin --> UC_block_user
admin --|> user : Kế thừa quyền đăng nhập/xuất
@enduml
```

---

## 2. Quy Trình Hoạt Động (Flow of Operation)

### A. Luồng Đăng Ký & Xác Thực Email qua OTP
Khi đăng ký tài khoản, trạng thái mặc định của tài khoản sẽ là chưa xác thực (`is_verified = False`). Một mã OTP gồm 6 chữ số ngẫu nhiên được sinh ra và gửi tới email của người dùng.

```plantuml
@startuml
autonumber
actor User as "Người dùng"
participant Client as "Frontend SPA"
participant API as "FastAPI Gateway"
database DB as "Cơ sở dữ liệu"
participant Email as "SMTP / Mock Email Service"

User -> Client: Điền form Đăng ký (Username, Email, Password)
Client -> API: Gửi request POST /auth/register
API -> API: Băm mật khẩu (PBKDF2-HMAC-SHA256, 100,000 vòng)
API -> DB: Kiểm tra Username / Email trùng lặp
alt Tài khoản đã tồn tại
    API -->> Client: Trả về lỗi 400 (Already Exists)
else Hợp lệ
    API -> DB: Khởi tạo tài khoản (is_verified = False)
    API -> API: Sinh mã OTP ngẫu nhiên (6 chữ số)
    API -> DB: Lưu OTP & Hạn hết hạn (15 phút) vào tài khoản
    API -> Email: Gửi mail OTP xác nhận
    API -->> Client: Trả về thông tin User thành công
    Client -> User: Yêu cầu nhập mã OTP xác thực
end

User -> Client: Nhập mã OTP
Client -> API: Gửi request POST /auth/verify-email (gửi kèm Session Token)
API -> DB: So sánh OTP và thời gian hết hạn
alt Mã không chính xác hoặc hết hạn
    API -->> Client: Trả về lỗi 400 (Invalid/Expired)
else Xác minh thành công
    API -> DB: Cập nhật is_verified = True, xóa OTP
    API -->> Client: Trả về thành công (Email verified)
    Client -> User: Chuyển hướng về Dashboard chính
end
@enduml
```

### B. Luồng Quên & Đặt Lại Mật Khẩu (Forgot / Reset Password)
Trường hợp người dùng quên mật khẩu, hệ thống cho phép khôi phục qua Email OTP. Việc đặt lại mật khẩu thành công sẽ đi kèm cơ chế bảo mật tự động vô hiệu hóa toàn bộ các phiên đăng nhập (Session) cũ.

```plantuml
@startuml
autonumber
actor User as "Người dùng"
participant Client as "Frontend SPA"
participant API as "FastAPI Gateway"
database DB as "Cơ sở dữ liệu"
participant Email as "SMTP / Mock Email Service"

User -> Client: Chọn "Forgot Password" & nhập Email
Client -> API: Gửi request POST /auth/forgot-password
API -> DB: Tìm kiếm Email trong hệ thống
alt Email không tồn tại
    API -->> Client: Trả về thông điệp giả lập thành công (Bảo mật thông tin)
else Hợp lệ
    API -> API: Sinh mã OTP đặt lại mật khẩu (6 chữ số)
    API -> DB: Lưu OTP & Hạn hết hạn vào bản ghi User
    API -> Email: Gửi mã OTP khôi phục mật khẩu
    API -->> Client: Trả về thành công
    Client -> User: Yêu cầu nhập OTP & Mật khẩu mới
end

User -> Client: Điền OTP và mật khẩu mới
Client -> API: Gửi request POST /auth/reset-password
API -> DB: Kiểm tra khớp mã OTP và Email
alt Không khớp hoặc hết hạn
    API -->> Client: Trả về lỗi 400 (Invalid code/email)
else Hợp lệ
    API -> API: Băm mật khẩu mới (PBKDF2-HMAC-SHA256)
    API -> DB: Cập nhật mật khẩu mới, xóa mã OTP
    API -> DB: Xóa sạch các phiên làm việc đang hoạt động (UserSession.delete)
    API -->> Client: Trả về đặt lại mật khẩu thành công
    Client -> User: Yêu cầu đăng nhập lại bằng mật khẩu mới
end
@enduml
```

---

## 3. Cách Hoạt Động & Cơ Chế Kỹ Thuật

### A. Mã hóa mật khẩu (Password Hashing)
Hệ thống không bao giờ lưu trữ mật khẩu dưới dạng văn bản thuần (plain text).
- **Thuật toán**: Sử dụng thuật toán chuẩn công nghiệp `PBKDF2-HMAC-SHA256`.
- **Muối ngẫu nhiên (Salt)**: Mỗi mật khẩu được trộn với một chuỗi muối 16 byte sinh ngẫu nhiên trước khi băm, ngăn chặn các cuộc tấn công Rainbow Table.
- **Vòng lặp**: Áp dụng $100,000$ vòng lặp băm để làm chậm các nỗ lực tấn công dò mật khẩu (Brute-force).
- **Định dạng lưu trữ**: Chuỗi hash được lưu trong DB dưới dạng `<salt_hex>:<key_hex>`.

### B. Quản lý Phiên (Session Management)
- **Token Phiên**: Khi người dùng đăng nhập thành công qua `POST /auth/login`, hệ thống tạo ra một mã token 64 ký tự hex ngẫu nhiên (`secrets.token_hex(32)`).
- **Lưu trữ phiên**: Token được lưu trong bảng cơ sở dữ liệu `user_sessions`, liên kết trực tiếp với `user_id` và định nghĩa thời gian hết hạn (`expires_at`).
- **Thời hạn sử dụng**: Mặc định là **7 ngày** kể từ thời điểm tạo.
- **Xác thực API (Authorization Header)**: Client phải gửi Token này trong mỗi Request thông qua Header `Authorization: Bearer <session_token>`.
- **Đăng xuất**: Khi gọi `POST /auth/logout` hoặc thực hiện đổi mật khẩu, hệ thống thực thi xóa bản ghi phiên khỏi database, ngay lập tức vô hiệu hóa token đó.
- **Kiểm tra trạng thái kích hoạt**: Tại dependency kiểm tra token `get_current_user`, nếu trường `is_active` của User bằng `False` (do bị Admin khóa), API sẽ lập tức chặn quyền truy cập và trả về lỗi 403 Forbidden.
