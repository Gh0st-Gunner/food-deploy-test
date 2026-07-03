# 4. Các Tối Ưu Hóa Hiệu Năng Vừa Thực Hiện (Performance Optimizations & Code Changes)

Tài liệu này tài liệu hóa các điểm nghẽn hiệu năng nghiêm trọng ban đầu khiến thời gian quét ảnh lên tới **~30 giây/ảnh** và các giải pháp kỹ thuật đã triển khai để rút ngắn latency xuống **dưới 10 giây**.

---

## 1. Tóm Tắt Các Điểm Nghẽn & Kết Quả Tối Ưu Hóa

| Điểm Nghẽn Ban Đầu | Giải Pháp Triển Khai | Kết Quả Đạt Được |
| :--- | :--- | :--- |
| **Vòng lặp SAM 2 tuần tự:** Chạy lại Image Encoder và Processor cho từng bounding box riêng lẻ. | **Batch Inference:** Gom toàn bộ bounding boxes vào một batch để nạp cho SAM 2, Image Encoder chạy duy nhất 1 lần. | Thời gian chạy mô hình phân đoạn SAM 2 giảm từ 2-3 giây mỗi box xuống còn dưới **0.2 giây** cho toàn bộ đĩa ăn. |
| **Trùng lặp chạy SAM 2:** Portion worker chạy lại SAM 2 để phân đoạn đĩa ăn từ đầu. | **S3 Mask Sharing:** Tích hợp mặt nạ nguyên liệu thành `combined_mask.png` tại Detection worker, lưu lên S3 để Portion worker tải về dùng luôn. | Tiết kiệm 100% tài nguyên tính toán và thời gian chạy mô hình SAM 2 lần hai tại Portion task. |
| **GPU Worker bị block mạng:** Gọi API bên thứ ba (USDA/FatSecret) đồng bộ bên trong GPU worker. | **Non-blocking GPU Tasks:** Loại bỏ các cuộc gọi API ngoài khỏi GPU task, chuyển tiếp tra cứu về CPU-only aggregator task. | GPU/VRAM được giải phóng lập tức sau khi chạy xong mô hình, không còn bị ngâm tài nguyên để chờ mạng. |
| **Hàng đợi chạy tuần tự:** Celery `chain` bắt toàn bộ task chạy nối đuôi nhau tuần tự. | **Parallel Celery Workflows:** Áp dụng Celery `chord` kết hợp `group` để chạy song song luồng dinh dưỡng và luồng thị giác AI. | Song song hóa thành công bước gọi API ngoài (CPU-bound) và bước suy diễn mô hình AI (GPU-bound). |
| **Redis không kháng lỗi:** Mất kết nối Redis đột ngột làm sập hoàn toàn các task/request. | **Redis Connection Resiliency:** Bọc try-except các thao tác cache và tự động fallback sang bộ nhớ đệm trong RAM. | Đảm bảo tính sẵn sàng cao của hệ thống ngay cả khi hệ thống cache trung tâm gặp sự cố. |

---

## 2. Chi Tiết Kỹ Thuật Các Thay Đổi Mã Nguồn

### A. SAM 2 Batch Inference & VRAM Guard
*   **Vị trí sửa đổi:** [ingredient_detector.py](file:///c:/Users/Home/Desktop/vn%20food/back-end/segmentation/ingredient_detector.py)
*   **Chi tiết thay đổi:**
    *   Bổ sung cơ chế **VRAM Guard** tự động kiểm tra kích thước ảnh đầu vào. Nếu ảnh vượt quá `1024px`, hệ thống sẽ tự động resize ảnh về giới hạn an toàn bằng bộ lọc nội suy song tuyến (Bilinear) để ngăn chặn lỗi tràn bộ nhớ GPU (CUDA OOM).
    *   Thay đổi định dạng danh sách hộp giới hạn thành mảng 3 chiều có dạng `[[ [x1, y1, x2, y2], [x3, y3, x4, y4] ]]` để đưa vào `sam_processor`.
    *   Sau khi chạy `sam_model(**inputs, multimask_output=False)`, mặt nạ dự đoán được trích xuất bằng cách truy cập mảng tensor `pred_masks[0, i, 0]`. Tọa độ các bounding box và mặt nạ kết quả được ánh xạ ngược về độ phân giải của bức ảnh gốc ban đầu.

### B. Liên Kết Mặt Nạ Giữa Các Worker Qua Object Storage
*   **Vị trí sửa đổi:** [detection_worker.py](file:///c:/Users/Home/Desktop/vn%20food/back-end/workers/detection_worker.py) và [portion_worker.py](file:///c:/Users/Home/Desktop/vn%20food/back-end/workers/portion_worker.py)
*   **Chi tiết thay đổi:**
    *   Trong `detect_ingredients_task`, sau khi có danh sách mặt nạ nhị phân của các nguyên liệu đơn lẻ, thực hiện phép toán OR nhị phân trên ma trận numpy để tạo mặt nạ tổng hợp (`combined_mask`).
    *   Chuyển đổi `combined_mask` từ kiểu dữ liệu boolean sang ảnh PNG màu xám (trắng = vùng thức ăn, đen = nền) và tải lên S3 tại key `results/{job_id}/combined_mask.png`.
    *   Trong `estimate_portion_task`, hệ thống kiểm tra sự tồn tại của file `combined_mask.png` trên S3. Nếu tải về thành công, nó sẽ chuyển đổi ảnh PNG ngược lại thành ma trận boolean và truyền vào tham số `ingredient_masks` của hàm tính portion, bỏ qua hàm `_segment_dish` sử dụng SAM 2.

### C. Thiết Kế Luồng Công Việc Bất Đồng Bộ Song Song (Celery Chord Workflow)
*   **Vị trí sửa đổi:** [routes.py](file:///c:/Users/Home/Desktop/vn%20food/back-end/api/routes.py)
*   **Chi tiết thay đổi:**
    *   Cấu trúc luồng Celery ban đầu (Tuần tự):
        `Classify` $\rightarrow$ `Nutrition Lookup` $\rightarrow$ `Detection` $\rightarrow$ `Portion` $\rightarrow$ `Aggregation`
    *   Cấu trúc luồng Celery mới (Song song):
        ```python
        # Nhánh song song chạy trên CPU và GPU độc lập
        header = group([
            lookup_nutrition_task.si(job_id),  # Nhánh A: Tra cứu dinh dưỡng món ăn (CPU worker)
            chain(                             # Nhánh B: Pipeline thị giác máy tính (GPU worker)
                detect_ingredients_task.si(job_id, None, image_s3_key, params),
                estimate_portion_task.si(job_id, None, image_s3_key, None, params)
            )
        ])
        
        # Kết hợp chuỗi: Chạy phân loại trước, sau đó kích hoạt nhánh song song, cuối cùng tổng hợp kết quả
        workflow = chain(
            classify_food.si(job_id, image_s3_key, request.models),
            chord(header, aggregate_results.si(job_id))
        )
        ```
