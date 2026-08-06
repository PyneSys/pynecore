---
sidebar_position: 1
---


import DownloadLink from '@site/src/components/DownloadLink';

# API Versioning

DNSE OpenAPI sử dụng cơ chế **Date-based Versioning** nhằm giúp clients chủ động kiểm soát quá trình nâng cấp hệ thống và đảm bảo backward compatibility giữa các phiên bản API.

Người dùng có thể tiếp tục sử dụng API version hiện tại để duy trì tính ổn định hoặc nâng cấp lên version mới để cập nhật các thay đổi và tính năng mới của nền tảng.

### Tổng quan
-	Phiên bản API được truyền thông qua Request Header `version`
-	Định dạng ngày: YYYY-MM-DD (VD: 2026-05-07)
-	Một API Version mới chỉ được tạo khi hệ thống có các **breaking changes** ảnh hưởng đến backward compatibility với các clients đang tích hợp.
- Version được xác định tại thời điểm xử lý từng request và không gắn cố định với API Key, tài khoản hoặc ứng dụng của người dùng.
- Các request khác nhau hoàn toàn có thể sử dụng các version khác nhau trong cùng một hệ thống tích hợp.

Ví dụ Header Request:

```http
x-api-key: lB58g6iWzyrNx2EhwwQXeYeoAnkzlaXkJWi  // APIKey được cấp khi đăng ký dịch vụ
x-Signature: Signature keyId="lB58g6iWzyrNx2EhwwQXeYeoAnkzlaXkJWi",algorithm="hmac-sha256",headers="(request-target) date",signature="U7NOnhIlAlsWJviOqtlRZajLmZmbq0Bb2T1EVsHm3%2Bg%3D",nonce="26c4b530cf12427d95bf691e39aa8d74"  // Chữ ký số theo thuật toán HMAC SHA256
Date: Fri, 15 May 2026 07:11:30 +0000     // Thời gian tạo yêu cầu (UTC)
version: 2026-05-07    // Phiên bản API
```

- Trường hợp giá trị `version` truyền lên sai định dạng `YYYY-MM-DD` hoặc không tồn tại, hệ thống trả về lỗi:

  ```json
  {
    "status": "error",
    "code": "OA-401",
    "message": "This API version does not seem to exist"
  }
  ```

### Quy tắc hoạt động

#### Cơ chế nhất quán phiên bản (Global Versioning)

Hệ thống áp dụng một phiên bản duy nhất cho toàn bộ nền tảng OpenAPI. Khi người dùng chỉ định Header version cụ thể (VD `version: 2027-05-07`)

- Đối với API có Breaking Changes tại ngày đó: Hệ thống kích hoạt xử lý theo logic mới.
- Đối với API không có thay đổi: Logic được giữ nguyên. Việc client truyền version cũ hay mới không làm ảnh hưởng đến hành vi của các API này.

:::tip[Lợi ích]

Người dùng không cần quản lý thủ công từng version riêng lẻ cho mỗi Endpoint khác nhau. Chỉ cần một Header duy nhất cho toàn bộ kết nối, giúp việc tích hợp và quản lý source code trở nên đơn giản hơn.

:::

#### Quy tắc Mapping phiên bản

Hệ thống tự động điều hướng Request dựa trên hai quy tắc sau:

- Phiên bản mặc định (Default version):
    - Áp dụng khi request **không truyền** Header `version`.
    - Hệ thống sẽ tự động fallback về version phát hành đầu tiên của nền tảng (mặc định là 2026-05-07) để đảm bảo backward compatibility cho các client hiện hữu.
    - Version này là cố định cho toàn bộ hệ thống và hoàn toàn không phụ thuộc vào thời điểm tài khoản của khách hàng được khởi tạo.
-	Chỉ định phiên bản:
- Áp dụng khi request **có truyền** Header `version`.
- Hệ thống sẽ mapping về phiên bản chính thức có ngày phát hành gần nhất trước đó hoặc bằng phiên bản client gửi lên.

<details>
  <summary>Nguyên tắc xử lý Version</summary>

1. Nếu Header `version` không được truyền → Sử dụng default version
2. Nếu version nhỏ hơn version đầu tiên → Mapping về version đầu tiên
3. Nếu version nằm giữa các đợt phát hành → Mapping về version gần nhất trước đó
4. Nếu version lớn hơn release mới nhất → Mapping về release mới nhất hiện có

    **Giả sử** DNSE có 2 phiên bản chính thức như sau:

    - 2026-05-07: Phát hành phiên bản đầu tiên (default version)
    - 2026-10-10: Phát hành phiên bản thứ 2 (Có breaking changes)

| Phiên bản client | Phiên bản thực tế | Mô tả                                                             |
|------------------|-------------------|-------------------------------------------------------------------|
| *(Không truyền)* | `2026-05-07`      | Tự động nhận diện phiên bản mặc định (Bản phát hành đầu tiên)     |
| `2025-12-01`     | `2026-05-07`      | Ngày gửi nhỏ hơn phiên bản đầu tiên, khớp về phiên bản mặc định   |
| `2026-09-15`     | `2026-05-07`      | Ngày nằm giữa 2 phiên bản, khớp về phiên bản gần nhất trước đó    |
| `2026-11-01`     | `2026-10-10`      | Ngày lớn hơn phiên bản thứ 2, khớp về phiên bản gần nhất trước đó |

*Các thông tin trên chỉ mang tính chất minh họa*
</details>

### Quản lý tương thích ngược (Backward Compatibility)

Để giúp người dùng chủ động lên kế hoạch nâng cấp source code, DNSE phân loại hai dạng thay đổi của hệ thống:

| Thay đổi làm tăng Version mới (Breaking Changes)     | Cập nhật trực tiếp, không tăng Version (Non-breaking)         |
|------------------------------------------------------|---------------------------------------------------------------|
| Thay đổi API Endpoint hiện tại                       | Bổ sung API Endpoint mới hoàn toàn                            |
| Thay đổi logic kiểm tra dữ liệu đầu vào (Validation) | Thêm các tham số tùy chọn (optional parameters) trong request |
| Đổi tên trường hoặc xóa trường trong dữ liệu trả về  | Bổ sung trường dữ liệu mới trong dữ liệu trả về               |
| Thay đổi kiểu dữ liệu cấu trúc (Data type)           | Bổ sung giá trị enum mới có tương thích ngược                 |

> Client nên triển khai parser theo hướng forward-compatible và bỏ qua các field không nhận diện trong response payload để đảm bảo khả năng tương thích với các thay đổi non-breaking trong tương lai.

<details>
  <summary>Ví dụ về phát hành version</summary>

| Thời điểm  | Nội dung thay đổi                                                         | API Version             |
|------------|---------------------------------------------------------------------------|-------------------------|
| 07-05-2026 | Triển khai API Versioning lần đầu                                         | `2026-05-07`            |
| 15-06-2026 | Thêm Endpoint mới lấy thông tin sự kiện quyền *(non-breaking)*            | Giữ nguyên `2026-05-07` |
| 20-08-2026 | Thêm field `matchedQuantity` trong response dữ liệu lệnh *(non-breaking)* | Giữ nguyên `2026-05-07` |
| 10-10-2026 | Thay đổi URL và request Header của API đặt lệnh (breaking changes)        | `2026-10-10`            |
| 15-01-2027 | Thay đổi cấu trúc response payload của API lấy vị thế (breaking changes)  | `2027-01-15`            |

- Kịch bản hệ thống: Giả sử client hiện tại đang sử dụng version `2026-05-07`. Vào ngày `2026-10-10`, DNSE phát hành API version mới với breaking changes:
    - Nếu client **giữ nguyên `version` cũ** hoặc không truyền Header `version`, hệ thống vẫn xử lý request theo đúng hành vi, cấu trúc URL cũ của bản `2026-05-07`.
    - Nếu client **nâng cấp` version`**, giá trị Header `version` đổi thành `2026-10-10` và người dùng cần cập nhật lại logic tích hợp tương ứng với cấu trúc mới.

*Các thông tin trên chỉ mang tính chất minh họa*
</details>

:::warning[Khuyến nghị tích hợp]

- Mặc dù hệ thống luôn fallback về bản phát hành đầu tiên, DNSE vẫn khuyến khích người dùng luôn ghim (Pin) một giá trị ngày phiên bản cụ thể thay vì để trống Header, nhằm kiểm soát source code một cách tường minh nhất.
- Việc không truyền `version` có thể khiến người dùng bỏ lỡ các tính năng mới hoặc hành vi cập nhật của hệ thống.
- Theo dõi và cập nhật các thay đổi mới nhất của chúng tôi tại <a href="https://developers.dnse.com.vn/docs/changelog">Changelog.</a>

:::

### Chính sách API Version

- Hiện tại DNSE chưa áp dụng cơ chế sunset version tự động.
- Các API version cũ vẫn tiếp tục được hỗ trợ nhằm đảm bảo tính ổn định cho các hệ thống đang tích hợp.
- Trong trường hợp có thay đổi về chính sách hỗ trợ version trong tương lai, DNSE sẽ thông báo chính thức thông qua Changelog và các kênh truyền thông kỹ thuật liên quan.

## API Versions

Danh sách dưới đây bao gồm các API version của DNSE OpenAPI.

---

### 2026-07-23

Phiên bản mở rộng hỗ trợ lệnh điều kiện (`orderCategory=STOP, OCO`) và lệnh thường Trái phiếu (`marketType=BOND`)

#### Non-breaking updates (áp dụng ngay, không cần nâng cấp version):

| Tên tính năng                        | Endpoint                                                    | They đổi                                                                                                                                                                    |
|--------------------------------------|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Sửa lệnh                             | `PUT /accounts/:accountNo/orders/:orderId`                  | Hỗ trợ thêm `marketType=BOND`                                                                                                                                               |
| Sức mua, sức bán                     | `GET /accounts/:accountNo/ppse`                             | Hỗ trợ thêm `marketType=BOND`                                                                                                                                               |
| Danh sách gói vay                    | `GET /accounts/:accountNo/loan-packages`                    | Hỗ trợ thêm `marketType=BOND`                                                                                                                                               |
| Đặt lệnh                             | `POST /accounts/:accountNo/orders` *(Endpoint mới)*         | Hỗ trợ thêm `marketType=BOND` với lệnh thường. Mở rộng loại lệnh STOP, OCO.<br/> Endpoint cũ `POST /accounts/orders` giữ nguyên chỉ hỗ trợ lệnh thường cho Cơ sở, Phái sinh |
| Chi tiết vị thế theo ID              | `GET /positions/{positionId}` *(Endpoint mới)*              | Chi tiết vị thế cổ phiếu và phái sinh<br/> Endpoint cũ `GET /accounts/positions/{positionId}` giữ nguyên                                                                    |
| Cấu hình chốt lời, cắt lỗ của vị thế | `GET /positions/{positionId}/pnl-configs` *(Endpoint mới)*  | Cấu hình chốt lời, cắt lỗ cho vị thế phái sinh<br/> Endpoint cũ `GET /accounts/positions/{positionId}/pnl-configs` giữ nguyên                                               |
| Cài đặt chốt lời, cắt lỗ cho vị thế  | `POST /positions/{positionId}/pnl-configs` *(Endpoint mới)* | Cài đặt cấu hình chốt lời, cắt lỗ cho vị thế phái sinh<br/> Endpoint cũ `POST /accounts/positions/{positionId}/pnl-configs` giữ nguyên                                      |
| Đóng vị thế                          | `POST /positions/{positionId}/close` *(Endpoint mới)*       | Đóng vị thế phái sinh<br/> Endpoint cũ `POST /accounts/positions/{positionId}/close` giữ nguyên                                                                             |

#### Breaking Changes (áp dụng từ version này):

**`DELETE /accounts/:accountNo/orders/:orderId` (Hủy lệnh)**
- Hỗ trợ thêm hủy lệnh thường Trái phiếu (`marketType=BOND`)
- Hỗ trợ thêm hủy lệnh STOP Cơ sở/ Phái sinh (`orderCategory=STOP`)
- Hỗ trợ thêm hủy lệnh OCO Phái sinh (`orderCategory=OCO`)
- **Breaking:** Trường `orderId` trong path params và response thay đổi kiểu dữ liệu từ `integer` → `string`

**`GET /accounts/:accountNo/orders` (Sổ lệnh)**
- Hỗ trợ phân trang
- Hỗ trợ thêm sổ lệnh Trái phiếu (`marketType=BOND`)
- Hỗ trợ thêm sổ lệnh STOP Cơ sở/ Phái sinh (`orderCategory=STOP`)
- Hỗ trợ thêm sổ lệnh OCO Phái sinh (`orderCategory=OCO`)
- **Breaking:** Trường `orderId` trong response thay đổi kiểu dữ liệu từ `integer` → `string`

**`GET /accounts/:accountNo/orders/:orderId` (Chi tiết lệnh theo ID)**
- Hỗ trợ truy vấn chi tiết lệnh thường Trái phiếu (`marketType=BOND`)
- **Breaking:** Response bỏ trường `reports`

> **Lưu ý:** Lệnh thường (`orderCategory=NORMAL`) vẫn sử dụng `orderId` dạng integer. Tuy nhiên, do cùng tích hợp trên một Endpoint với STOP/OCO, kiểu dữ liệu được thống nhất sang `string` để đảm bảo tính nhất quán.

> Client giữ nguyên hoặc không truyền Header `version` sẽ không bị ảnh hưởng bởi breaking changes, nhưng không thể sử dụng các tính năng mới. Để sử dụng, nâng cấp lên `version: 2027-07-23` và cập nhật logic parse `orderId` từ `integer` sang `string`.

**Supported SDKs:**
[DNSE sample SDK](https://github.com/dnse-tech/openapi-sdk) phiên bản **2.0.0** trở lên.

**Resources:**
💎 <DownloadLink href="https://cdn.entrade.com.vn/dnse-openapi/doc/dnse-openapi-2026-07-23.yaml" filename="dnse-openapi-2026-05-07.yaml">dnse-openapi-2026-07-23.yaml</DownloadLink>

---

### 2026-05-07
Phiên bản đầu tiên triển khai cơ chế API Versioning.

Giữ nguyên cấu trúc và logic của toàn bộ API đã được phát hành trước đây, đảm bảo các hệ thống hiện tại tiếp tục hoạt động mà không cần thay đổi tích hợp.

- Các request không truyền Header `version` sẽ sử dụng phiên bản mặc định
- Hỗ trợ tương thích ngược (backward compatibility) giữa các phiên bản API
- Hỗ trợ mapping version theo ngày phát hành (release date)

**Resources:**
💎 <DownloadLink href="https://cdn.entrade.com.vn/dnse-openapi/doc/dnse-openapi-2026-05-07.yaml" filename="dnse-openapi-2026-05-07.yaml">dnse-openapi-2026-05-07.yaml</DownloadLink>

