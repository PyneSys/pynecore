---
sidebar_position: 2
---


# SDK Versioning

DNSE áp dụng cơ chế **Semantic Versioning** để quản lý version cho bộ SDK, đảm bảo sự đồng bộ chặt chẽ với lộ trình phát triển của nền tảng DNSE OpenAPI.
Tham khảo bộ SDK tại đây: [GitHub DNSE OpenAPI](https://github.com/dnse-tech/openapi-sdk)

### Tổng quan

Phiên bản SDK được quản lý theo cấu trúc: **`MAJOR.MINOR.PATCH`**
- **MAJOR**: Có breaking changes không tương thích ngược với major version trước đó
- **MINOR**: Bổ sung tính năng mới nhưng vẫn đảm bảo backward compatibility
- **PATCH**: Sửa lỗi hoặc tối ưu hiệu năng (đảm bảo tương thích ngược).
- VD: 2.3.1

<details>
  <summary>Ví dụ minh họa</summary>

| Phiên bản | Loại thay đổi | 	Nội dung chi tiết                                                |
|-----------|---------------|-------------------------------------------------------------------|
| 1.0.0	    | Initial	      | Phát hành bộ SDK đầu tiên                                         |
| 1.1.0	    | Minor	        | Bổ sung API "Truy vấn lịch sử khớp lệnh"                          |
| 1.1.3	    | Patch	        | Sửa lỗi không nhận diện được ký tự đặc biệt response dữ liệu lệnh |
| 1.2.0	    | Minor	        | Thêm field `matchedQuantity` trong response dữ liệu lệnh          |
| 2.0.0	    | Major	        | Đổi kiểu dữ liệu `price` từ string sang number                    |
| 2.0.1	    | Patch	        | Tối ưu tốc độ parse JSON trong các phản hồi từ Websocket          |

*Các thông tin trên chỉ mang tính chất minh họa*
</details>


### Quy tắc hoạt động

SDK versioning được quản lý độc lập với API versioning.
Một SDK version có thể hỗ trợ nhiều API versions khác nhau, miễn là các API versions đó vẫn tương thích với SDK.

| SDK Version | API Version | Ghi chú |
| --- | --- | --- |
| **1.3.0** | `2026-05-07` | Hỗ trợ đầy đủ Date-based Versioning & toàn bộ endpoints hiện tại |

> **Lưu ý:** Người dùng có thể chủ động ghi đè phiên bản API thông qua request Header `version` trong cấu hình SDK.

### Backward Compatibility

DNSE ưu tiên duy trì backward compatibility trong các SDK version mới nhằm giảm thiểu ảnh hưởng tới hệ thống tích hợp hiện tại.

Một số thay đổi được xem là breaking changes trong SDK, sẽ nâng cấp Major version:

- Đổi tên method, class hoặc interface
- Thay đổi kiểu dữ liệu đầu vào hoặc đầu ra
- Xóa method hoặc model đã công bố
- Thay đổi cơ chế xác thực (authentication) hay cơ chế chữ ký số (signature)

Các thay đổi sau không được xem là breaking changes:

- Thêm helper methods mới
- Thêm các tham số tùy chọn (optional parameters)
- Bổ sung model hoặc enum values mới
- Sửa lỗi hoặc tối ưu performance không làm thay đổi API contract

<details>
  <summary>Ví dụ về phát hành version</summary>

| Thời điểm  | Thay đổi                                                                  | SDK Version | API Version             |
|------------|---------------------------------------------------------------------------|-------------|-------------------------|
| 07-05-2026 | Phát hành version API đầu tiên                                            | `1.3.0`     | `2026-05-07`            |
| 15-06-2026 | Thêm Endpoint mới lấy thông tin sự kiện quyền                             | `1.4.0`     | Giữ nguyên `2026-05-07` |
| 20-06-2026 | Tối ưu performance cho websocket                                          | `1.4.1`     | Giữ nguyên `2026-05-07` |
| 20-08-2026 | Thêm field `matchedQuantity` trong response dữ liệu lệnh *(non-breaking)* | `1.5.0`     | Giữ nguyên `2026-05-07` |
| 10-10-2026 | Breaking change: Thay đổi URL và request Header của API đặt lệnh          | `2.0.0`     | `2026-10-10`            |

- Giả sử client hiện tại đang sử dụng SDK version `1.3.0` và API version `2026-05-07`
- Tại thời điểm `2026-10-10`, DNSE phát hành API version mới với breaking changes:
  - Nếu client **không nâng version SDK**: SDK 1.3.0 vẫn gọi đến các URL cũ và xử lý dữ liệu theo cấu trúc cũ. Không có rủi ro về mặt vận hành, nhưng người dùng sẽ không sử dụng được các tính năng mới trong version `2026-10-10`
  - Nếu client **nâng version SDK** lên `2.0.0` và API version `2026-10-10`: SDK hỗ trợ các thay đổi mới của nền tảng OpenAPI, có thể sử dụng Endpoint đặt lệnh mới nhất

*Các thông tin trên chỉ mang tính chất minh họa*
</details>

:::warning[Khuyến nghị tích hợp]

- Một API có breaking changes có thể dẫn tới SDK Major version mới nếu thay đổi đó ảnh hưởng tới public SDK interfaces.
- DNSE khuyến khích người dùng luôn sử dụng SDK Major version mới nhất để không bỏ lỡ các tính năng mới phát hành.
- Theo dõi và cập nhật các thay đổi mới nhất của chúng tôi tại <a href="https://developers.dnse.com.vn/docs/changelog">Changelog.</a>

:::