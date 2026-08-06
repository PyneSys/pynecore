# Error Codes
---

## Error Response

DNSE OpenAPI trả về lỗi thông qua hai thành phần: `HTTP status code` ở response header và `code` ở response body.
Tùy loại lỗi, response có thể kèm thêm `status` và `message` theo từng trường hợp cụ thể tương ứng.

Ví dụ:

```json lines
{
  "status": "error",
  "code": "OA-400",
  "message": "Authorization field missing, malformed or invalid"
}
```
```json lines
{
  "code": "INVALID_MARKET_TYPE"
}
```
```json lines
{
  "success": false,
  "code": "FORBIDDEN",
  "message": "You do not have access to this account"
}
```
```json lines
{
  "status": 400,
  "code": "RESOURCE_NOT_FOUND",
  "message": "Not found deal by id=..."
}
```

Luôn đọc trường `code` để xác định nguyên nhân và tra cứu hướng xử lý.

## Error Codes

### OpenAPI Errors

| HTTP status | Code   | Meaning               | Description                                                                                 | Recommended Action                                                      |
|-------------|--------|-----------------------|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| 400         | OA-400 | Bad request           | Request không hợp lệ — thiếu thông tin bắt buộc, sai tham số, sai format, thiếu trường body | Kiểm tra lại tham số và định dạng dữ liệu theo tài liệu API             |
| 401         | OA-401 | Unauthorized          | API Key không hợp lệ hoặc đã bị xóa                                                         | Kiểm tra API Key còn hiệu lực và được truyền đúng trong header          |
| 403         | OA-403 | Forbidden             | Không có quyền thực hiện yêu cầu — bị từ chối hoặc tài khoản thiếu quyền                    | Đảm bảo API Key được cấp đúng permission cho chức năng này              |
| 404         | OA-404 | Not found             | Endpoint không tồn tại hoặc tài nguyên không tìm thấy                                       | Kiểm tra lại đường dẫn endpoint và thông tin tài nguyên                 |
| 405         | OA-405 | Method not allowed    | HTTP Method không được hỗ trợ cho endpoint này                                              | Kiểm tra lại HTTP Method theo tài liệu API                              |
| 422         | OA-422 | Unprocessable entity  | Request đúng định dạng nhưng không thỏa mãn điều kiện nghiệp vụ                             | Kiểm tra dữ liệu nghiệp vụ trước khi gửi lại request                    |
| 429         | OA-429 | Too Many Requests     | Vượt quá giới hạn số lượng request                                                          | Kiểm tra lại tần suất nằm trong giới hạn giờ và ngày theo từng Endpoint |
| 500         | OA-500 | Internal server error | Lỗi hệ thống tạm thời                                                                       | Thử lại sau. Nếu lỗi tiếp tục, liên hệ bộ phận hỗ trợ                   |
| 503         | OA-503 | Service unavailable   | Dịch vụ tạm thời không khả dụng                                                             | Thử lại sau                                                             |

### Validation Errors

| Code                                            | Description                       | Recommended Action                                          |
|-------------------------------------------------|-----------------------------------|-------------------------------------------------------------|
| ACCOUNT_MISSING                                 | Thiếu thông tin tài khoản         | Truyền thông tin tiểu khoản giao dịch hợp lệ                |
| SYMBOL_MISSING                                  | Thiếu mã chứng khoán              | Truyền mã chứng khoán trong request                         |
| INPUT_MISSING                                   | Thiếu dữ liệu đầu vào             | Kiểm tra và truyền đầy đủ các trường bắt buộc               |
| INPUT_INVALID                                   | Dữ liệu đầu vào không hợp lệ      | Kiểm tra lại giá trị các trường theo tài liệu API           |
| INPUT_FORMAT_INVALID                            | Sai định dạng dữ liệu             | Kiểm tra kiểu dữ liệu và định dạng các trường               |
| INVALID_ORDER_TYPE                              | Loại lệnh không hợp lệ            | Kiểm tra giá trị `orderType` theo danh sách được hỗ trợ     |
| INVALID_ORDER_SIDE                              | Chiều lệnh không hợp lệ           | Kiểm tra giá trị `side` (NB/NS)                             |
| INVALID_SYMBOL                                  | Mã chứng khoán không hợp lệ       | Kiểm tra mã chứng khoán trước khi gửi request               |
| INVALID_PRICE                                   | Giá đặt không hợp lệ              | Kiểm tra giá nằm trong biên độ trần/sàn của mã              |
| INVALID_PRICE_LOT                               | Giá không đúng bước giá           | Điều chỉnh giá theo bước giá của sàn giao dịch              |
| INVALID_QUANTITY                                | Khối lượng không hợp lệ           | Kiểm tra khối lượng lớn hơn 0 và thỏa mãn điều kiện mua/bán |
| INVALID_QUANTITY_LOT                            | Khối lượng không đúng quy định lô | Điều chỉnh khối lượng theo quy định lô của sàn              |
| PRICE_MUST_LESS_THAN_OR_EQUAL_TO_CEILING_PRICE  | Giá đặt vượt giá trần             | Giảm giá xuống không vượt quá giá trần                      |
| PRICE_MUST_GREATER_THAN_OR_EQUAL_TO_FLOOR_PRICE | Giá đặt thấp hơn giá sàn          | Tăng giá lên không thấp hơn giá sàn                         |

### Trading Session Errors

| Code                                                    | Description                                 | Recommended Action                                           |
|---------------------------------------------------------|---------------------------------------------|--------------------------------------------------------------|
| CAN_NOT_PLACE_ORDER_ON_THIS_SESSION                     | Không thể đặt lệnh trong phiên này          | Thực hiện trong phiên giao dịch phù hợp                      |
| CAN_NOT_PLACE_ORDER_WITH_THAT_ORDER_TYPE_ON_ATO_SESSION | Loại lệnh không hỗ trợ trong phiên ATO      | Đổi loại lệnh sang LO/ATO hoặc thực hiện trong phiên phù hợp |
| CAN_NOT_PLACE_ORDER_WITH_THAT_ORDER_TYPE_ON_ATC_SESSION | Loại lệnh không hỗ trợ trong phiên ATC      | Đổi loại lệnh sang LO/ATC hoặc thực hiện trong phiên phù hợp |
| INVALID_ORDER_TYPE_FOR_THIS_SESSION                     | Loại lệnh không hợp lệ trong phiên hiện tại | Đổi loại lệnh hoặc đặt lại trong phiên phù hợp               |
| INVALID_TRADING_SESSION                                 | Phiên giao dịch không hợp lệ                | Kiểm tra thời gian và phiên giao dịch hiện tại               |
| BATCH_IN_PROGRESS                                       | Hệ thống đang xử lý cuối ngày               | Thử lại sau khi hệ thống mở giao dịch trở lại                |

### Order Processing Errors

| Code                                            | Description                                      | Recommended Action                                     |
|-------------------------------------------------|--------------------------------------------------|--------------------------------------------------------|
| INVALID_ORDER_ID                                | Không tìm thấy lệnh                              | Kiểm tra lại Order ID                                  |
| ORDER_STATUS_REJECTED                           | Không thể thao tác với trạng thái lệnh hiện tại  | Kiểm tra trạng thái lệnh trước khi thực hiện           |
| ORDER_IS_DONE                                   | Lệnh đã hoàn tất hoặc đã hủy                     | Không thể sửa hoặc hủy lệnh ở trạng thái này           |
| CAN_NOT_CANCEL_ATO_ORDER                        | Không thể hủy lệnh ATO                           | Lệnh ATO không hỗ trợ hủy trong phiên hiện tại         |
| CAN_NOT_CANCEL_MARKET_ORDER                     | Không thể hủy lệnh thị trường                    | Lệnh MTL/MOK/MAK không hỗ trợ hủy ngoài phiên liên tục |
| CAN_NOT_CANCEL_PENDINGNEW_ORDER_IN_OPEN_SESSION | Không thể hủy lệnh đang ở trạng thái Chờ gửi     | Thử lại sau khi lệnh được gửi lên sàn                  |
| CAN_NOT_CANCEL_THAT_ORDER_ON_THIS_SESSION       | Không thể hủy lệnh trong phiên hiện tại          | Thực hiện hủy trong phiên phù hợp                      |
| CAN_NOT_REPLACE_PLO_ORDER                       | Không thể sửa lệnh PLO                           | Lệnh PLO không hỗ trợ sửa trong phiên hiện tại         |
| CAN_NOT_REPLACE_THAT_ORDER_ON_THIS_SESSION      | Không thể sửa lệnh trong phiên hiện tại          | Thực hiện sửa lệnh trong phiên liên tục                |
| CAN_NOT_PLACE_PLO_ORDER_WITHOUT_MATCHED         | Không thể đặt lệnh PLO khi không có lệnh đối ứng | Đặt PLO khi đã có lệnh khớp trong phiên                |
| CANNOT_PLACE_OPPOSITE_ORDER                     | Không thể đặt lệnh ngược chiều                   | Kiểm tra và xử lý lệnh chờ khớp ở chiều đối diện trước |
| CANNOT_PLACE_OPPOSITE_ORDER_IN_THIS_SESSION     | Không thể đặt lệnh đối ứng trong phiên này       | Thực hiện trong phiên liên tục                         |
| RESOURCE_NOT_FOUND                              | Không tìm thấy tài nguyên                        | Kiểm tra lại thông tin định danh được yêu cầu          |

### Buying Power & Margin Errors

| Code                        | Description                                | Recommended Action                                                 |
|-----------------------------|--------------------------------------------|--------------------------------------------------------------------|
| PURCHASING_POWER_NOT_ENOUGH | Không đủ sức mua                           | Giảm giá trị lệnh hoặc nộp thêm tiền vào tài khoản để tăng sức mua |
| PP0_EXCEED                  | Vượt sức mua                               | Giảm giá trị hoặc khối lượng lệnh                                  |
| QMAX_EXCEED                 | Khối lượng vượt quá sức mua/sức bán tối đa | Giảm giá đặt hoặc khối lượng đặt lệnh                              |
| STOCK_NOT_ENOUGH            | Không đủ chứng khoán để bán                | Kiểm tra số dư khả dụng trước khi đặt lệnh bán                     |
| VIOLATE_POOL_RULE           | Vượt hạn mức Pool cho vay                  | Giảm khối lượng hoặc chọn gói vay khác                             |
| VIOLATE_ROOM_RULE           | Vượt hạn mức Room cho vay                  | Giảm khối lượng hoặc chờ hạn mức được cập nhật                     |
| OUT_OF_MARGIN_BASKET        | Mã chứng khoán không thuộc danh mục ký quỹ | Kiểm tra danh mục mã được phép mua vay                             |

### Symbol Status Errors

| Code                                          | Description                               | Recommended Action                             |
|-----------------------------------------------|-------------------------------------------|------------------------------------------------|
| SYMBOL_NOT_EXIST                              | Không tìm thấy mã chứng khoán             | Kiểm tra lại mã chứng khoán                    |
| CAN_NOT_PLACE_ORDER_ON_HALTED_SYMBOL          | Mã đang tạm ngừng giao dịch               | Đặt lại sau khi mã được giao dịch trở lại      |
| CAN_NOT_PLACE_ORDER_ON_AOM_HALTED_SYMBOL      | Mã bị chặn giao dịch trong phiên liên tục | Thử lại trong phiên phù hợp                    |
| CAN_NOT_PLACE_ORDER_ON_SUSPENDED_SYMBOL       | Mã bị đình chỉ giao dịch                  | Không thể giao dịch cho đến khi mã được mở lại |
| CAN_NOT_PLACE_ORDER_ON_UNLISTED_SYMBOL        | Mã đã hủy niêm yết                        | Không thể giao dịch mã này                     |
| CAN_NOT_PLACE_ODD_LOT_ORDER_ON_SPECIAL_SYMBOL | Không hỗ trợ lô lẻ với mã đặc biệt này    | Thực hiện giao dịch theo quy định của mã       |

### Authentication & Permission Errors

| Code                  | Description                             | Recommended Action                                  |
|-----------------------|-----------------------------------------|-----------------------------------------------------|
| FORBIDDEN             | Không có quyền thực hiện thao tác       | Đảm bảo tài khoản có quyền thực hiện chức năng này  |
| INVALID_OTP           | OTP không hợp lệ                        | Kiểm tra lại mã OTP hoặc lấy mã OTP mới             |
| INVALID_TRADING_TOKEN | Trading Token không hợp lệ hoặc hết hạn | Kiểm tra hoặc xác thực lại để lấy Trading Token mới |

### System Errors

| Code                | Description                | Recommended Action                                    |
|---------------------|----------------------------|-------------------------------------------------------|
| TIMEOUT             | Vượt quá thời gian xử lý   | Thử lại sau bằng cơ chế exponential backoff           |
| SYSTEM_ERROR        | Lỗi hệ thống               | Thử lại sau. Nếu lỗi tiếp tục, liên hệ bộ phận hỗ trợ |
| REMOTE_SERVER_ERROR | Lỗi từ hệ thống backend    | Thử lại sau hoặc liên hệ hỗ trợ nếu lỗi kéo dài       |
| THIRD_PARTY_ERROR   | Lỗi từ hệ thống bên thứ ba | Thử lại sau. Nếu lỗi tiếp tục, liên hệ bộ phận hỗ trợ |

---

## Khuyến nghị xử lý mã lỗi

- Kiểm tra HTTP Status trước — `2xx` xử lý bình thường, `4xx/5xx` đọc `code` để phân loại.
- Không retry đối với lỗi xác thực (401), phân quyền (403) và dữ liệu đầu vào (400, 422) — cần khắc phục root cause trước.
- Có thể retry với backoff cho lỗi 500, 503, `TIMEOUT` — khuyến nghị exponential backoff, tối đa 3 lần.
- Với lỗi 429, kiểm tra Header response để biết hạn mức còn lại, chờ đến thời điểm chỉ định trong header `X-RateLimit-Reset` trước khi retry.
- Khi cần hỗ trợ, cung cấp đầy đủ: HTTP Status, `code`, `message`, endpoint, thời điểm xảy ra, Request ID / Trace ID nếu có.