
# Changelog

## [2026-08-06]

### 🚀 Added
- **[REST API]** Thêm Endpoint mới <a href="https://developers.dnse.com.vn/docs/dnse/get-expected-price">**Get Expected Price (Giá dự khớp)**</a>: Truy vấn thông tin giá dự khớp của mã chứng khoán trong phiên khớp lệnh định kỳ.

### 🧩 Update

**[REST API]**

- Cập nhật <a href="https://developers.dnse.com.vn/docs/dnse/get-account-balances">**GET account balances (Thông tin tiền)**</a>:
  - Response trả thêm `bond.totalValue`, `egg.totalValue`

- Cập nhật các Endpoint mở rộng giao dịch Trái phiếu (`marketType=BOND`):
  - <a href="https://developers.dnse.com.vn/docs/dnse/get-loan-packages">**GET Loan Package (Gói vay)**</a>: Hỗ trợ truy vấn gói vay đặt lệnh Trái phiếu
  - <a href="https://developers.dnse.com.vn/docs/dnse/get-ppse">**GET PPSE (Sức mua/sức bán)**</a>: Hỗ trợ truy vấn sức mua đặt lệnh Trái phiếu
  - <a href="https://developers.dnse.com.vn/docs/dnse/replace-order">**PUT Orders (Sửa lệnh)**</a>: Hỗ trợ sửa lệnh thường Trái phiếu

- Cập nhật <a href="https://developers.dnse.com.vn/docs/dnse/place-order">**POST Order (Đặt lệnh)**</a>
  - Bổ sung Endpoint mới `POST /accounts/{accountNo}/orders`
  - Endpoint mới mở rộng hỗ trợ các loại lệnh:

    | orderCategory          | NORMAL | STOP | OCO |
        |------------------------|:------:|:----:|:---:|
    | STOCK (cơ sở)          |   ✅    |  ✅   |  ❌  |
    | DERIVATIVE (phái sinh) |   ✅    |  ✅   |  ✅  |
    | BOND (trái phiếu)      |   ✅    |  ❌   |  ❌  |

  - Endpoint cũ `POST /accounts/orders` tiếp tục hoạt động đầy đủ và không bị ảnh hưởng, chỉ hỗ trợ lệnh thường cơ sở và lệnh thường phái sinh.

- Cập nhật <a href="https://developers.dnse.com.vn/docs/dnse/get-position-by-id">**GET Position by Id (Chi tiết vị thế theo ID)**</a>
  - Bổ sung Endpoint mới `GET /positions/{positionId}`
  - Endpoint cũ `GET /accounts/positions/{positionId}` tiếp tục được hỗ trợ để đảm bảo tương thích ngược.

- Cập nhật <a href="https://developers.dnse.com.vn/docs/dnse/get-pnl-configs-position">**GET PNL Configs (Cấu hình chốt lời, cắt lỗ của vị thế)**</a>
  - Bổ sung Endpoint mới `GET /positions/{positionId}/pnl-configs`
  - Endpoint cũ `GET /accounts/positions/{positionId}/pnl-configs` tiếp tục được hỗ trợ để đảm bảo tương thích ngược.

- Cập nhật <a href="https://developers.dnse.com.vn/docs/dnse/post-pnl-configs-position">**POST PNL Configs (Cài đặt chốt lời, cắt lỗ của vị thế)**</a>
  - Bổ sung Endpoint mới `POST /positions/{positionId}/pnl-configs`
  - Endpoint cũ `POST /accounts/positions/{positionId}/pnl-configs` tiếp tục được hỗ trợ để đảm bảo  tương thích ngược.

- Cập nhật <a href="https://developers.dnse.com.vn/docs/dnse/close-position">**POST Close Position (Đóng vị thế)**</a>
  - Bổ sung Endpoint mới `POST /positions/{positionId}/close`
  - Endpoint cũ `POST /accounts/positions/{positionId}/close` tiếp tục được hỗ trợ để đảm bảo tương thích ngược.

  :::warning[Khuyến nghị tích hợp]

  Người dùng nên tích hợp Endpoint mới để sử dụng các tính năng mới được cập nhật.

  :::

### 🚨 Breaking Changes

**[REST API] <a href="https://developers.dnse.com.vn/docs/guide/versioning/api#2026-07-23">API version: 2026-07-23</a>**
- Cập nhật endpoint <a href="https://developers.dnse.com.vn/docs/dnse/cancel-order">**DELETE Orders (Hủy lệnh)**</a>
  - Hỗ trợ hủy lệnh thường Trái phiếu  (truyền `marketType=BOND` và `orderCategory=NORMAL`)
  - Hỗ trợ hủy lệnh STOP Cơ sở và Phái sinh (`orderCategory=STOP`)
  - Hỗ trợ hủy lệnh OCO Phái sinh (`orderCategory=OCO`)
  - Trường `orderId` trong Request và Response thay đổi kiểu dữ liệu từ `integer` → `string`

- Cập nhật endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-orders">**GET Orders (Sổ lệnh)**</a>
  - Request yêu cầu truyền `pageIndex`, `pageSize`
  - Response trả thêm thông tin phân trang
  - Hỗ trợ truy vấn sổ lệnh thường Trái phiếu (truyền `marketType=BOND` và `orderCategory=NORMAL`)
  - Hỗ trợ truy vấn sổ lệnh STOP Cơ sở và Phái sinh (`orderCategory=STOP`)
  - Hỗ trợ truy vấn sổ lệnh OCO Phái sinh (`orderCategory=OCO`)
  - Trường `orderId` trong response thay đổi kiểu dữ liệu từ `integer` → `string`

- Cập nhật endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-order-detail">**GET Order Detail (Chi tiết lệnh theo ID)**</a>
  - Hỗ trợ truy vấn chi tiết lệnh thường Trái phiếu (truyền `marketType=BOND` và `orderCategory=NORMAL`)
  - Response bỏ trường `reports`

  :::warning[Khuyến nghị tích hợp]

  Để sử dụng các tính năng mới, client cần nâng cấp lên `API version: 2026-23-07` và chú ý cập nhật logic parse `orderId` sang `string`.<br/>Client giữ nguyên Header `version` cũ hoặc không truyền Header `version` sẽ vẫn tiếp tục được hỗ trợ tính năng cũ và không bị ảnh hưởng.<br/>Hoặc dử dụng [DNSE sample SDK](https://github.com/dnse-tech/openapi-sdk) phiên bản **2.0.0** trở lên.

  :::

----
## [2026-07-14]

### 🧩 Update
- [REST API] Cập nhật endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-executions">**Executions (Chi tiết trạng thái lệnh)**</a>
  - Hỗ trợ chi tiết trạng thái lệnh cổ phiếu với `marketType` = `STOCK`

----
## [2026-06-23]
### 🚀 Added
- [REST API] Thêm Endpoint mới <a href="https://developers.dnse.com.vn/docs/dnse/get-session">**Get Session (Thông tin phiên)**</a>: Truy vấn thông tin phiên giao dịch.

- [WEBSOCKET] Thêm function mới:

  - <a href="https://developers.dnse.com.vn/docs/guide/market-data/broker_connect">**Postion Event for Broker (Dữ liệu vị thế của tài khoản khách hàng theo thời gian thực)**</a>: Cung cấp thông tin chi tiết về vị thế trên tài khoản khách hàng thuộc quyền quản lý theo thời gian thực cho người dùng là Môi giới tại DNSE.

  - <a href="https://developers.dnse.com.vn/docs/guide/market-data/connect">**Session (Thông tin phiên)**</a>: Cung cấp thông tin chi tiết về phiên giao dịch hiện tại.

----
## [2026-06-23]
### 🚀 Added
- [REST API] Thêm Endpoint mới <a href="https://developers.dnse.com.vn/docs/dnse/get-foreign-trading">**Get Foreign Trading (Dữ liệu NĐT nước ngoài)**</a>: Truy vấn thông tin dữ liệu nhà đầu tư nước ngoài.

### 🛠️ Maintenance
- [Websocket] Sửa lỗi xử lý kết nối và cải thiện độ ổn định.

----
## [2026-06-11]
### 🧩 Update
- [WEBSOCKET] Thêm function mới:
  - <a href="https://developers.dnse.com.vn/docs/guide/market-data/broker_connect">**Order Event for Broker (Dữ liệu lệnh của tài khoản khách hàng theo thời gian thực)**</a>: Cung cấp thông tin chi tiết về lệnh giao dịch trên tài khoản khách hàng thuộc quyền quản lý theo thời gian thực cho người dùng là Môi giới tại DNSE.

  - <a href="https://developers.dnse.com.vn/docs/guide/market-data/connect">**Estimate VN30 Data (Dữ liệu chỉ số VN30 dự tính)**</a>: Cung cấp thông tin dữ liệu về chỉ số VN30 dự tính trong phiên giao dịch.

----
## [2026-06-04]
### 🚀 Added
- [REST API] Thêm Endpoint mới:
  - <a href="https://developers.dnse.com.vn/docs/dnse/get-pnl-configs-position">**Get PNL Configs (Cấu hình chốt lời, cắt lỗ của vị thế)**</a>: Trả ra cấu hình cài đặt chốt lời, cắt lỗ của vị thế đang nắm giữ.
  - <a href="https://developers.dnse.com.vn/docs/dnse/post-pnl-configs-position">**Post PNL Configs (Cài đặt cấu hình chốt lời, cắt lỗ cho vị thế)**</a>: Cài đặt cấu hình cài đặt chốt lời, cắt lỗ cho vị thế đang nắm giữ.

### 🧩 Update
- [REST API] Cập nhật endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-brokers-accounts-care-by">**Get list care by**</a>
  - Response trả thêm `investorId`

----
## [2026-05-28]
### 🚀 Added
- [REST API] Thêm Endpoint mới:
  - <a href="https://developers.dnse.com.vn/docs/dnse/get-corporate-action-history">**Get Corporate Action History (Lịch sử sự kiện quyền)**</a>: Trả ra danh sách sự kiện quyền chứng khoán trên tài khoản khách hàng.
  - <a href="https://developers.dnse.com.vn/docs/dnse/get-latest-quotes">**Get Latest Quotes (Dữ liệu bid/ask gần nhất)**</a>: Trả ra dữ liệu chào mua, chào bán gần nhất của mã chứng khoán
  - <a href="https://developers.dnse.com.vn/docs/dnse/get-quotes">**Get Quotes (Lịch sử bid/ask)**</a>: Trả ra lịch sử chào mua, chào bán của mã chứng khoán.

### 🧩 Update
- [REST API] Cập nhật endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-ppse">**Get PPSE (Sức mua, sức bán)**</a>
  - Response trả thêm `pp0Buy`, `pp0Short`

----
## [2026-05-26]
### 🧩 Update
- [REST API] Cập nhật endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-symbol-secdef">**Get Secdef (Thông tin giao dịch chứng khoán)**</a>
  - Response trả thêm `time` (định dạng `yyyy-MM-dd'T'HH:mm:ss.SSS | UTC+7`)

- [REST API] Cập nhật trường `time` trong response của các endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-price-symbol-close">**Get Close Price (Giá đóng cửa)**</a>, <a href="https://developers.dnse.com.vn/docs/dnse/get-history-trades">**Get Trades (Lịch sử khớp lệnh)**</a>, <a href="https://developers.dnse.com.vn/docs/dnse/get-latest-trades">**Get Latest Trades (Dữ liệu khớp gần nhất)**</a>
  - Thay đổi timezone từ `UTC` sang `UTC+7`
  - Bổ sung độ chính xác đến milliseconds (`yyyy-MM-dd'T'HH:mm:ss.SSS`)

### 🛠️ Maintenance
- [Websocket] Sửa lỗi xử lý kết nối và cải thiện độ ổn định.

----
## [2026-05-12]
### 🚀 Added
- [PLATFORM] API Versioning
  Chính thức áp dụng cơ chế quản lý phiên bản API cho OpenAPI theo định dạng date-based versioning. Tham khảo chi tiết tại <a href="https://developers.dnse.com.vn/docs/guide/versioning/api">**API Versioning**</a>
- [WEBSOCKET] Thêm function mới:
  <a href="https://developers.dnse.com.vn/docs/guide/trading-data-connect">**Position Event (Dữ liệu vị thế thời gian thực)**</a>: Cung cấp thông tin chi tiết về vị thế đang nắm giữ trên tài khoản người dùng theo thời gian thực.

### 🧩 Update
- [REST API] Cập nhật endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-positions">**Get Positions (Vị thế nắm giữ)**</a>
  - Response trả thêm `averageClosePrice`

### 🛠️ Maintenance
- [Websocket] Sửa lỗi xử lý dữ liệu, tối ưu hiệu năng và cải thiện độ ổn định

----
## [2026-04-21]

### 🚀 Added
- [REST API] Thêm Endpoint mới:
  - <a href="https://developers.dnse.com.vn/docs/dnse/get-brokers-accounts-care-by">**/Get list care by (Danh sách tiểu khoản quản lý)**</a>: Trả ra danh sách tiểu khoản giao dịch thuộc quyền quản lý của người dùng là Môi giới/SACO tại DNSE.
- [Websocket] Thêm function mới:
  - <a href="https://developers.dnse.com.vn/docs/guide/trading-data-connect">**Order Event (Dữ liệu lệnh thời gian thực)**</a>: Cung cấp thông tin chi tiết về lệnh giao dịch trên tài khoản người dùng theo thời gian thực.

----
## [2026-04-14]

### 🚀 Added
- [REST API] Thêm Endpoint mới:
  - <a href="https://developers.dnse.com.vn/docs/dnse/get-market-working-dates">**Working Dates (Ngày làm việc)**:</a> Trả ra danh sách ngày làm việc trong vòng 1 năm tính từ ngày hiện tại.
### 🧩 Update
- [Websocket] Function <a href="https://developers.dnse.com.vn/docs/guide/market-data/connect#security-definition">**Security Definition (Thông tin giao dịch chứng khoán)**</a>
  - Payload trả thêm `listingDate`, `finalTradeDate`
- [REST API] Cập nhật endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-symbol-secdef">**Thông tin giao dịch chứng khoán**</a>
  - Response trả thêm `listingDate`, `finalTradeDate`

----
## [2026-04-07]

### 🚀 Added

- [Websocket] Thêm function mới:
  - <a href="https://developers.dnse.com.vn/docs/guide/market-data/connect#security-definition">**OHLC Closed (OHLC đóng nến)**:</a>  Cung cấp dữ liệu nến đã đóng theo từng khung thời gian.
  - <a href="https://developers.dnse.com.vn/docs/guide/market-data/connect#security-definition">**Foreign Investor (Giao dịch nhà đầu tư nước ngoài)**:</a> Cung cấp dữ liệu giao dịch của nhà đầu tư nước ngoài theo từng mã chứng khoán.
- [REST API] Thêm Endpoint mới:
  - <a href="https://developers.dnse.com.vn/docs/dnse/get-executions">**Executions (Chi tiết trạng thái lệnh)**:</a>  Trả về lịch sử các lần cập nhật trạng thái hay khớp từng phần của một lệnh giao dịch, áp dụng cho phái sinh.
  - <a href="https://developers.dnse.com.vn/docs/dnse/get-price-symbol-close">**Close Price (Giá đóng cửa)**:</a> Trả về dữ liệu giá đóng cửa của mã chứng khoán

### 🧩 Update

- [REST API] Cập nhật endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-account-balances">**Thông tin tiền**</a>
  - Response trả thêm `secureAmount`, `orderSecured`

### 🚨 Breaking Changes

- <a href="https://developers.dnse.com.vn/docs/guide/market-data/connect">[Websocket]</a> Cấu trúc dữ liệu trả về được thay đổi từ **snake_case** (VD: `board_id`) sang **camelCase** (VD: `boardId`)
  - Áp dụng cho các trường trong dữ liệu WebSocket
  - Client cần cập nhật lại mapping các trường theo định dạng mới

- <a href="https://developers.dnse.com.vn/docs/guide/market-data/connect">[Websocket]</a> Một số trường enum được thay đổi kiểu dữ liệu từ **integer** (VD:`board_Id`: 2, 5, 9) sang **string** (VD: `boarId`: G1, G4, T1).
  - Áp dụng cho nhiều trường trong các payload WebSocket
  - Client cần cập nhật logic parse dữ liệu tương ứng

- Cập nhật định nghĩa enum
  - Tham khảo trang <a href="https://developers.dnse.com.vn/docs/guide/enum/market_data">**Enum Market Data**</a> để xem giá trị và format mới nhất.