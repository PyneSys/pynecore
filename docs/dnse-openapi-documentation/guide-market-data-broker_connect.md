
# Broker WebSocket

Tài liệu này hướng dẫn cách thiết lập kết nối đến DNSE WebSocket để nhận dữ liệu giao dịch theo thời gian thực dành cho người dùng là Môi giới tại DNSE.

----

## Thông tin kết nối chung

- Base URL: `wss://ws-openapi.dnse.com.vn`
- DNSE cung cấp sẵn bộ SDK đã phân tách theo từng loại dữ liệu để khách hàng có thể sẵn sử dụng. Chi tiết xem [sample SDKs tại đây.](https://github.com/dnse-tech/openapi-sdk)
- Định dạng dữ liệu trong SDKs:
  - `msgpack`: Tốc độ xử lý nhanh, tiết kiệm băng thông
  - `json`: Phổ biến và dễ đọc trong quá trình phát triển
- Cơ chế kết nối:
  - Một kết nối WebSocket có hiệu lực tối đa 8 giờ, WebSocket Server sẽ chủ động ngắt kết nối sau thời gian này.
  - Cơ chế để các clients duy trì kết nối ổn định tới WebSocket server DNSE:
    - WebSocket Server sẽ định kỳ gửi 1 PING message sau mỗi 3 phút.
    - Mỗi PING message được gửi từ WebSocket đều yêu cầu nhận PONG message phản hồi từ các client trong thời gian tối đa là 1 phút kể từ lúc Server gửi PING. Nếu quá thời hạn 1 phút này, Server sẽ chủ động ngắt kết nối với Client không đáp ứng.
    - Client được phép gửi PONG message ngay cả khi không nhận được PING từ Server, để chủ động duy trì kết nối. Cách này giúp client giữ kết nối trong các trường hợp PING message bị miss do network issue hoặc các gián đoạn tạm thời khác.

<details>
  <summary>Ví dụ</summary>

- **Case 1: Good interaction**
  - T+0 min   Server → PING
  - T+1     Client → PONG
  - T+3 min   Server → PING
  - T+4     Client → PONG

  Client phản hồi PONG cho mỗi PING từ Server.

  ✅ Connection remains active

- **Case 2: Bad interaction**
  - T+0 min   Server → PING
  - No PONG back from client
  - T+1 min   Server disconnects

  Server đóng kết nối do không nhận được PONG trong khoảng thời gian kết nối định kỳ.

  ❌  Dead Connection

- **Case 3: Client-initiated keepalive**
  - Within every 3 minutes: Client → PING

  Client có thể chủ động gửi PONG message để duy trì kết nối, đặc biệt trong các tình huống:
  - Một số thư viện WS thực hiện auto-handle ping/pong hoặc ẩn các ping frames đối với các app/clients
  - Mobile networks / NATs chủ động ngắt kết nối đối với các idle TCP connections
  - Miss PING frame từ server

  Do đó, việc cho phép clients định kỳ gửi PONG lên để Server xác nhận client vẫn đang hoạt động.

  ✅ Connection keep alive

</details>

----

### Tổng quan các kênh dữ liệu

| Kênh dữ liệu (Function)                                 | Mô tả (Description)                                                                                       | Phân loại (Type) | Tần suất gửi dữ liệu (Frequency)                                  |
|---------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|------------------|-------------------------------------------------------------------|
| [Dữ liệu lệnh thời gian thực](#broker-order-event)      | Dữ liệu lệnh giao dịch theo thời gian thực trên tài khoản <br/>khách hàng thuộc quản lý của môi giới      | Real-time        | Cập nhật khi lệnh giao dịch trên tài khoản khách hàng có thay đổi |
| [Dữ liệu vị thế thời gian thực](#broker-position-event) | Dữ liệu vị thế đang nắm giữ theo thời gian thực trên<br/> tài khoản khách hàng thuộc quản lý của môi giới | Real-time        | Cập nhật khi vị thế trên tài khoản khách hàng có thay đổi         |

### Giới hạn kết nối

- Mỗi WebSocket Connection chỉ hỗ trợ subscribe Channel cho duy nhất 1 `investor_id`.
- Chỉ được phép sử dụng các `investor_id` đã được ủy quyền cho Môi giới. Nếu `investor_id` không thuộc danh sách được phân quyền, hệ thống sẽ trả về lỗi phân quyền `user does not have permissions for investorId`

---

## Các kênh dữ liệu giao dịch

### Dữ liệu lệnh thời gian thực (Broker Order Event) {#broker-order-event}

Cung cấp thông tin chi tiết về lệnh giao dịch trên tài khoản khách hàng thuộc quản lý của Môi giới. Hệ thống sẽ đẩy dữ liệu ngay khi có sự thay đổi liên quan đến: lệnh mới, thay đổi trạng thái, hoặc thay đổi giá khớp, khối lượng khớp.

#### Định dạng Channel

>  **order.broker.\{market_type\}.\{investor_id\}.\{encoding\}**

- **market_type**: Phân loại thị trường
  - `STOCK`: Lệnh giao dịch cơ sở
  - `DERIVATIVE`: Lệnh giao dịch phái sinh

- **investor_id**: Mã định danh tài khoản khách hàng tại DNSE, được trả về từ response Endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-brokers-accounts-care-by">**Get list careby (Danh sách tiểu khoản quản lý)**:</a>
- **encoding**: Định dạng dữ liệu `msgpack` hoặc `json`

#### Payload

```json lines
{
  "id": 596,                        // integer  // Id lệnh giao dịch
  "side": "NS",                     // string   // Chiều đặt lệnh (NB: Mua, NS: Bán)
  "accountNo": "0001179019",        // string   // Số tiểu khoản
  "symbol": "41I1G5000",            // string   // Mã chứng khoán
  "orderType": "LO",                // string   // Loại lệnh
  "price": 1920.0,                  // float    // Giá đặt
  "quantity": 5,                    // integer  // Khối lượng đặt
  "fillQuantity": 2,                // integer  // Khối lượng khớp
  "canceledQuantity": 0,            // integer  // Khối lượng đã hủy
  "leaveQuantity": 3,               // integer  // Khối lượng còn lại chưa khớp
  "orderStatus": "PartiallyFilled", // string   // Trạng thái lệnh
  "loanPackageId": 2278,            // integer  // Mã gói vay
  "marketType": "DERIVATIVE",       // string   // Loại thị trường
  "transDate": "2026-04-06T00:00:00Z", // string   // Ngày giao dịch
  "createdDate": "2026-04-13T04:24:05.274Z", // string   // Thời điểm tạo (UTC)
  "modifiedDate": "2026-04-13T04:28:27.749Z" // string   // Thời điểm cập nhật (UTC)
}
```

Để có thêm thông tin về vòng đời lệnh, các trạng thái của lệnh, người dùng tham khảo <a href="https://developers.dnse.com.vn/docs/guide/trading-api/trading_order">tại đây.</a>

### Dữ liệu vị thế thời gian thực (Broker Position Event) {#broker-position-event}

Cung cấp thông tin chi tiết về vị thế đang nắm giữ trên tài khoản khách hàng thuộc quản lý của Môi giới. Hệ thống sẽ đẩy dữ liệu ngay khi có sự thay đổi liên quan đến: mở vị thế, đóng vị thế, thay đổi khối lượng, giá vốn, giá đóng, giá thị trường hoặc trạng thái vị thế.

#### Định dạng Channel

>  **position.broker.\{market_type\}.\{investor_id\}.\{encoding\}**

- **market_type**: Phân loại thị trường
  - `STOCK`: Lệnh giao dịch cơ sở
  - `DERIVATIVE`: Lệnh giao dịch phái sinh

- **investor_id**: Mã định danh tài khoản khách hàng tại DNSE, được trả về từ response Endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-brokers-accounts-care-by">**Get list careby (Danh sách tiểu khoản quản lý)**:</a>
- **encoding**: Định dạng dữ liệu `msgpack` hoặc `json`

#### Payload

```json lines
{
  "id": 177796763592657,            // integer  // Id vị thế
  "accountNo": "0001179019",        // string   // Số tiểu khoản
  "symbol": "41I1G5000",            // string   // Mã chứng khoán
  "status": "OPEN",                 // string   // Trạng thái vị thế (OPEN: Đang mở, CLOSED: Đã đóng)
  "loanPackageId": 2278,            // integer  // Mã gói vay
  "side": "NB",                     // string   // Chiều vị thế (NB: Mua, NS: Bán)
  "accumulateQuantity": 247,        // integer  // Tổng khối lượng đã mở được cộng dồn trong vị thế
  "tradeQuantity": null,            // integer  // Dành cho thị trường cơ sở
  "closedQuantity": 236,            // integer  // Khối lượng đã đóng
  "costPrice": 2057.72425,          // float    // Giá vốn trung bình
  "marketPrice": 2070.0,            // float    // Giá thị trường hiện tại
  "breakEvenPrice": 2058.21911,     // float    // Giá hòa vốn
  "openQuantity": 11,               // integer  // Khối lượng vị thế đang mở
  "overNightQuantity": 0,           // integer  // Khối lượng mở qua đêm
  "averageClosePrice": 2094.28941,  // float    // Giá đóng trung bình tính trên khối lượng đã đóng
  "marketType": "DERIVATIVE",       // string   // Loại thị trường
  "createdDate": "2026-05-05T09:17:50.457893Z", // string // Thời điểm mở vị thế (UTC)
  "modifiedDate": "2026-05-07T04:19:20.901188117Z" // string // Thời điểm cập nhật gần nhất (UTC)
}
```
