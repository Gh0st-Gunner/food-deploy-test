# 2. Kiến Trúc Hệ Thống & Sơ Đồ Luồng (System Architecture & Workflows)

Tài liệu này chi tiết hóa kiến trúc phần cứng, phân tách dịch vụ, thiết kế cơ sở dữ liệu, phân vùng hàng đợi Celery, cấu trúc lưu trữ S3 và luồng truyền thông thời gian thực bằng WebSockets.

---

## 1. Thiết Kế Hệ Thống Tổng Quan (System Architecture)
Munchin' được thiết kế dưới dạng kiến trúc hướng dịch vụ microservices thu nhỏ, tối ưu cho các hệ thống AI yêu cầu kết hợp giữa các tác vụ API nhanh (CPU-bound/IO-bound) và các tác vụ tính toán học sâu nặng (GPU-bound).

Các thành phần hệ thống bao gồm:
1.  **Frontend Client:** Giao diện SPA thuần (Vanilla HTML/CSS/JS) chạy trên trình duyệt người dùng.
2.  **FastAPI Gateway (API Service):** Cổng API chính chịu trách nhiệm tiếp nhận request, xác thực người dùng, điều phối tác vụ và truyền dữ liệu thời gian thực.
3.  **Database (PostgreSQL / SQLite fallback):** Lưu trữ thông tin người dùng, phiên làm việc và trạng thái/kết quả của các tác vụ phân tích (Jobs).
4.  **Redis Cache & Message Broker:**
    *   Lưu trữ hàng đợi Celery.
    *   Lưu trữ kết quả cache dinh dưỡng bên ngoài.
    *   Đóng vai trò kênh Pub/Sub truyền tin nhắn cập nhật tiến trình phân tích.
5.  **Celery Workers (Mạng lưới xử lý bất đồng bộ):**
    *   *Worker Classification:* Phân loại tên món ăn (chạy CPU).
    *   *Worker Detection:* Chạy Grounding DINO + SAM 2 + Depth Anything V2 (chạy GPU/CUDA).
    *   *Worker Nutrition:* Tra cứu API dinh dưỡng bên ngoài (chạy CPU/Network).
    *   *Worker Default:* Thực hiện gom kết quả (Aggregation) và các tác vụ nhẹ khác.
6.  **MinIO Object Storage (S3-compatible):** Lưu trữ hình ảnh người dùng tải lên, ảnh mặt nạ phân đoạn trung gian, ảnh vẽ đè overlay và bản đồ chiều sâu.

---

## 2. Các Thành Phần Công Nghệ Lõi

### A. Cơ Chế Kết Nối Cơ Sở Dữ Liệu Tự Phục Hồi (DB Fallback Mechanism)
Hệ thống sử dụng SQLAlchemy để quản lý phiên kết nối:
*   Mặc định hệ thống sẽ cố gắng kết nối tới cơ sở dữ liệu sản xuất **PostgreSQL**.
*   **SQLite Fallback:** Nếu kết nối tới PostgreSQL thất bại do sự cố mạng hoặc lỗi Docker, hệ thống tự động bắt ngoại lệ và khởi tạo một tệp tin SQLite cục bộ (`vnfood_backup.db`) làm backup. Cơ chế này đảm bảo máy chủ API vẫn có thể khởi động và phục vụ người dùng trong mọi tình huống.

### B. Redis Cache & Khả Năng Chống Chịu Lỗi (Redis Fault Tolerance)
Redis lưu trữ thông tin dinh dưỡng tạm thời của nguyên liệu và ánh xạ món ăn để tiết kiệm chi phí gọi API bên thứ ba.
*   **Runtime Fallback:** Khi Redis mất kết nối đột ngột lúc hệ thống đang chạy, toàn bộ lệnh đọc/ghi cache được bao bọc trong khối `try-except` để tự động chuyển sang lưu trữ cục bộ dạng từ điển trong RAM (`_memory_cache`). Khi Redis hoạt động trở lại, hệ thống sẽ tự phục hồi kết nối và tiếp tục ghi đè cache lên Redis.

### C. Phân Vùng Hàng Đợi Celery (Worker Task Routing)
Mỗi worker Celery được cấu hình để lắng nghe trên các hàng đợi (queues) cụ thể, tối ưu hóa việc phân phối phần cứng:
*   `queue="classification"`: Định tuyến về CPU worker để chạy mô hình phân loại nhỏ hoặc gọi API đám mây (Gemini/Ollama).
*   `queue="detection"`: Định tuyến về GPU worker có liên kết CUDA để xử lý các mô hình thị giác máy tính rất nặng (SAM 2, DINO, Depth Anything).
*   `queue="nutrition"`: Định tuyến về CPU worker để xử lý các truy vấn mạng HTTP đến API USDA hoặc FatSecret.
*   `queue="default"`: Định tuyến về CPU worker để thực hiện việc tổng hợp dữ liệu cuối cùng.

---

## 3. Bản Đồ Lưu Trữ Object Storage (S3 Key Layout)
Mọi file hình ảnh và nhị phân của một Job đều được tổ chức có cấu trúc trên S3/MinIO thông qua UUID của Job:
*   **Ảnh gốc người dùng tải lên:** `images/{job_id}/original.jpg`
*   **Mặt nạ kết hợp (Combined Mask):** `results/{job_id}/combined_mask.png` (Ảnh PNG nhị phân lưu trữ mask tổng của các nguyên liệu, dùng để Portion Worker đọc trực tiếp và bỏ qua bước phân đoạn SAM 2 lần hai).
*   **Ảnh vẽ đè nguyên liệu (Overlay Image):** `results/{job_id}/overlay.png` (Ảnh hiển thị cho người dùng với các nhãn nguyên liệu được vẽ viền màu).
*   **Bản đồ chiều sâu 3D (Depth Map):** `results/{job_id}/depth_map.png` (Bản đồ chiều sâu trực quan dạng thang độ xám/màu).

---

## 4. Truyền Thông Thời Gian Thực Bằng WebSockets
Khi người dùng bắt đầu quét món ăn, client sẽ khởi tạo kết nối WebSocket trực tiếp đến API Gateway tại địa chỉ `/api/v1/jobs/{job_id}/stream`:
1.  API Gateway đăng ký lắng nghe (Subscribe) trên kênh Redis Pub/Sub có tên `job_updates:{job_id}`.
2.  Mỗi khi một worker hoàn thành một công đoạn (phân loại xong, bóc tách nguyên liệu xong, tính thể tích xong), worker đó sẽ cập nhật DB và đẩy một bản tin trạng thái JSON vào kênh Redis Pub/Sub tương ứng.
3.  API Gateway nhận được bản tin từ Redis Pub/Sub và lập tức đẩy (push) thông tin tiến trình (`classification: completed`, `detection: running`...) về trình duyệt client.
4.  Client cập nhật giao diện (vòng tròn loading, thanh tiến trình) theo thời gian thực mà không cần dùng cơ chế Polling (liên tục gửi yêu cầu HTTP GET) gây quá tải server. Khi status đạt trạng thái `completed`, WebSocket sẽ tự động đóng kết nối sau khi gửi kết quả cuối cùng.
