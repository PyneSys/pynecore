---
sidebar_position: 1
---
# Market Data WebSocket

Tài liệu này hướng dẫn cách thiết lập kết nối đến DNSE WebSocket để nhận dữ liệu thị trường theo thời gian thực.

----

### Thông tin kết nối chung

- Base URL: `wss://ws-openapi.dnse.com.vn`
- DNSE cung cấp sẵn bộ SDK đã phân tách theo từng loại dữ liệu để khách hàng có thể sẵn sử dụng. Chi tiết xem sample
  SDKs [tại đây.](https://github.com/dnse-tech/openapi-sdk)
- Định dạng dữ liệu trong SDKs:
  - `msgpack`: Tốc độ xử lý nhanh, tiết kiệm băng thông
  - `json`: Phổ biến và dễ đọc trong quá trình phát triển
- Cơ chế kết nối:
  - Tất cả mã chứng khoán phải ở định dạng chữ in hoa. VD: ACB, HPG, 41I1G2000.
  - Một kết nối WebSocket có hiệu lực tối đa 8 giờ, WebSocket Server sẽ chủ động ngắt kết nối sau thời gian này.
  - Cơ chế để các clients duy trì kết nối ổn định tới WebSocket server DNSE:
    - WebSocket Server sẽ định kỳ gửi 1 PING message sau mỗi 3 phút.
    - Mỗi PING message được gửi từ WebSocket đều yêu cầu nhận PONG message phản hồi từ các client trong thời gian
      tối đa là 1 phút kể từ lúc Server gửi PING. Nếu quá thời hạn 1 phút này, Server sẽ chủ động ngắt kết nối với
      Client không đáp ứng.
    - Client được phép gửi PONG message ngay cả khi không nhận được PING từ Server, để chủ động duy trì kết nối.
      Cách này giúp client giữ kết nối trong các trường hợp PING message bị miss do network issue hoặc các gián đoạn
      tạm thời khác.

<details>
  <summary>Ví dụ</summary>

- **Case 1: Good interaction**
  - T+0 min Server → PING
  - T+1 Client → PONG
  - T+3 min Server → PING
  - T+4 Client → PONG

  Client phản hồi PONG cho mỗi PING từ Server.

  ✅ Connection remains active

- **Case 2: Bad interaction**
  - T+0 min Server → PING
  - No PONG back from client
  - T+1 min Server disconnects

  Server đóng kết nối do không nhận được PONG trong khoảng thời gian kết nối định kỳ.

  ❌ Dead Connection

- **Case 3: Client-initiated keepalive**
  - Within every 3 minutes: Client → PING

  Client có thể chủ động gửi PONG message để duy trì kết nối, đặc biệt trong các tình huống:
  - Một số thư viện WS thực hiện auto-handle ping/pong hoặc ẩn các ping frames đối với các app/clients
  - Mobile networks / NATs chủ động ngắt kết nối đối với các idle TCP connections
  - Miss PING frame từ server

  Do đó, việc cho phép clients định kỳ gửi PONG lên để Server xác nhận client vẫn đang hoạt động.

  ✅ Connection keep alive

</details>

### Tổng quan các kênh dữ liệu

| Kênh dữ liệu <br/>(Function)                | Mô tả <br/>(Description)                                                                                   | Phân loại <br/>(Type) | Tần suất gửi dữ liệu <br/>(Frequency)                          |
|---------------------------------------------|------------------------------------------------------------------------------------------------------------|-----------------------|----------------------------------------------------------------|
| [Security Definition](#security-definition) | Thông tin giao dịch chứng khoán (giá trần/sàn, trạng thái),<br/>dùng để lấy thông tin biên độ giá đầu ngày | Batch (BOD)           | Gửi 1 lần trước giờ giao dịch (≈ 08:00)                        |
| [Trade](#trade)                             | Dữ liệu khớp lệnh theo thời gian thực                                                                      | Real-time             | Cập nhật khi có thay đổi dữ liệu trong phiên liên tục          |
| [Trade Extra](#trade)                       | Dữ liệu khớp lệnh nâng cao (chiều mua/bán, giá trung bình)                                                 | Real-time             | Cập nhật khi có thay đổi dữ liệu trong phiên liên tục          |
| [Quotes](#quotes)                           | Độ sâu thị trường (giá chào mua/bán)                                                                       | Real-time             | Cập nhật khi có thay đổi dữ liệu trong phiên giao dịch         |
| [OHLC](#ohlc)                               | Dữ liệu nến đang hình thành (Open, High, Low, Close, Volume)                                               | Real-time             | Cập nhật khi có giá khớp trong phiên liên tục                  |
| [OHLC Closed](#ohlc-closed)                 | Dữ liệu nến đã đóng (Open, High, Low, Close, Volume)                                                       | Periodic              | Cập nhật khi có dữ liệu đóng nến                               |
| [Expected Price](#expected-price)           | Giá và khối lượng dự kiến khớp lệnh trong phiên ATO/ATC                                                    | Real-time             | Cập nhật khi có thay đổi dữ liệu trong phiên định kỳ (ATO/ATC) |
| [Market Index](#market-index)               | Thông tin chỉ số thị trường (VNINDEX, HNX…)                                                                | Periodic              | Cập nhật theo chu kỳ 5 giây + tổng hợp cuối ngày               |
| [Foreign Investor](#foreign-investor)       | Dữ liệu giao dịch của nhà đầu tư nước ngoài theo từng mã                                                   | Real-time             | Cập nhật khi có thay đổi dữ liệu                               |
| [Estimated VN30](#estimated-vn30)           | Dữ liệu dự tính của chỉ số VN30                                                                            | Real-time             | Cập nhật khi có thay đổi dữ liệu                               |
| [Thông tin phiên](#session)                 | Dữ liệu dự tính của chỉ số VN30                                                                            | Real-time             | Cập nhật khi có thay đổi dữ liệu                               |

---

### Phân loại dữ liệu (Data Classification)

Để tối ưu hóa việc phát triển ứng dụng, khách hàng cần nắm rõ đặc tính và cơ chế truyền tải của từng kênh dữ liệu:

* **Real-time (Dữ liệu thời gian thực):** Dữ liệu được hệ thống chủ động đẩy xuống (Push) ngay khi có sự kiện phát sinh.
  Cơ chế này đảm bảo độ trễ thấp nhất cho phía Client.
* **Periodic (Dữ liệu định kỳ):** Dữ liệu chỉ xuất hiện hoặc được cập nhật vào các khoảng thời gian cố định hoặc theo
  từng giai đoạn (Phase) của phiên giao dịch.
* **Batch Data (Dữ liệu lô):** Dữ liệu Snapshot được xử lý theo đợt lớn.
  * **BOD (Beginning of Day):** Dữ liệu khởi tạo trước giờ giao dịch (VD: Danh mục mã, Giá trần/sàn).
  * **EOD (End of Day):** Dữ liệu tổng hợp sau khi kết thúc ngày giao dịch (VD: Giá đóng cửa chính thức).

---

## Các loại dữ liệu thị trường

Một số input dùng chung cho các channel:
- **board_id**: Mã bảng giao dịch (VD: G1 – lô chẵn)

  | boardId | Mô tả                        |
  |---------|------------------------------|
  | G1      | Lô chẵn                      |
  | G3      | Board phiên sau giờ (PLO)    |
  | G4      | Lô lẻ                        |
  | T1      | Thỏa thuận lô chẵn 9h-14h45  |
  | T3      | Thỏa thuận lô chẵn 14h45-15h |
  | T4      | Thỏa thuận lô lẻ 9h-14h45    |
  | T6      | Thỏa thuận lô lẻ 14h45-15h   |

- **market_index**: Tên chỉ số thị trường

  | Giá trị     | Mô tả                                              |
    |-------------|----------------------------------------------------|
  | HNX30       | Chỉ số Top 30 cổ phiếu sàn HNX                     |
  | VN30        | Chỉ số Top 30 cổ phiếu sàn HOSE                    |
  | HNX         | Chỉ số sàn HNX                                     |
  | VNXALLSHARE | Chỉ số các cổ phiếu chọn lọc sàn HOSE              |
  | UPCOM       | Chỉ số sàn UPCOM                                   |
  | VNDIVIDEND  | Chỉ số nhóm cổ phiếu có tỷ suất cổ tức tăng trưởng |
  | VNINDEX     | Chỉ số sàn HOSE                                    |
  | VN50GROWTH  | Chỉ số nhóm 50 cổ phiếu tăng trưởng sàn HOSE       |
  | VN100       | Chỉ số Top 100 cổ phiếu sàn HOSE                   |
  | VNMITECH    | Chỉ số nhóm cổ phiếu công nghệ                     |  

- **tsc_prod_grp_id**: Nhóm sản phẩm theo thị trường

  | Enums (tscProGrpId) | Kiểu dữ liệu | Mô tả                         |
    |---------------------|--------------|-------------------------------|
  | FBX                 | string       | Hợp đồng tương lai Trái phiếu |
  | FIO                 | string       | Hợp đồng tương lai chỉ số     |
  | HCX                 | string       | Trái phiếu Doanh nghiệp HNX   |
  | STO                 | string       | Cổ phiếu sàn HOSE             |
  | STX                 | string       | Cổ phiếu sàn HNX              |
  | UPX                 | string       | Cổ phiếu sàn Upcom            |

### Thông tin giao dịch chứng khoán (Security Definition) {#security-definition}

Cung cấp thông tin về giá trần sàn tham chiếu và trạng thái của mã chứng khoán trong ngày giao dịch. Dữ liệu được hệ
thống gửi hai lần vào lúc hệ thống kết thúc chạy dữ liệu cuối ngày (khoảng 8 giờ tối) và vào đầu ngày giao dịch hôm
sau (khoảng 8 giờ sáng).

#### Channel

>  **security_definition.\{board_id\}.\{encoding\}**

#### Input
- `symbols`: Mã hoặc danh sách mã chứng khoán.
- `board_id`: Mã bảng giao dịch

#### Payload

```json lines
{
  "marketId": "DVX", //string  // Mã thị trường niêm yết mã chứng khoán
  "boardId": "G1", //string  // Mã bảng giao dịch
  "isin": "VN41I1G20009", //string  // Mã định danh quốc tế
  "symbol": "41I1G2000", //string  // Mã chứng khoán
  "productGrpId": "FIO", //string  // Nhóm sản phẩm theo thị trường
  "securityGroupId": "FU", //string  // Nhóm chứng khoán
  "basicPrice": 2066.6, //float   // Giá tham chiếu ngày giao dịch
  "ceilingPrice": 2211.2, //float   // Giá trần ngày giao dịch
  "floorPrice": 1922.0, //float   // Giá sàn ngày giao dịch
  "openInterestQuantity": 24473, //integer // Khối lượng hợp đồng phái sinh mở qua đêm
  "securityStatus": "NO_HALT", //string  // Trạng thái giao dịch của mã chứng khoán
  "symbolAdminStatusCode": "NRM", //string  // Trạng thái quản lý hành chính mã chứng khoán
  "symbolTradingMethodStatusCode": "NRM", //string  // Trạng thái cơ chế giao dịch mã chứng khoán
  "symbolTradingSanctionStatusCode": "NRM", //string  // Tình trạng giao dịch của mã chứng khoán
  "listingDate": 20260223, //string  // Ngày niêm yết
  "finalTradeDate": 20260416, //string  // Ngày giao dịch cuối cùng (đối với mã phái sinh, chứng quyền) 
  "time": {   // Thời gian trong các message giá của sàn trả về
    "Seconds": 1779757279,// integer // Unix timestamp seconds
    "Nanos": 101000000 // integer // Nanoseconds
  }  
}
```

### Dữ liệu khớp lệnh (Trade & Trade Extra) {#trade}

DNSE cung cấp dữ liệu khớp lệnh của một mã chứng khoán qua 2 Function khác nhau: Trade và Trade Extra. Trade Extra có
thêm một số thông tin mà DNSE tự tổng hợp thêm (mua bán chủ động, giá khớp trung bình), nếu người dùng không có nhu cầu
lấy các thông tin này thì có thể dùng function Trade đơn thuần để tối ưu hơn về tốc độ nhận dữ liệu.

#### Định dạng Channel

>  **tick.\{board_id\}.\{encoding\}**

>  **tick_extra.\{board_id\}.\{encoding\}**

#### Input
- `symbols`: Mã hoặc danh sách mã chứng khoán.
- `board_id`: Mã bảng giao dịch

#### Payload

```json lines
{
  "marketId": "STO", // string // Mã thị trường niêm yết mã chứng khoán
  "boardId": "G1", // string // Mã bảng giao dịch
  "isin": "VN000000HPG4", // string // Mã định danh quốc tế
  "symbol": "HPG", // string // Mã chứng khoán

  "matchPrice": 24.35, // float // Giá khớp gần nhất
  "matchQtty": 40, // integer // Khối lượng khớp gần nhất

  "totalVolumeTraded": 1184240, // integer // Tổng khối lượng khớp trong ngày
  "grossTradeAmount": 287.17458, // float // Tổng giá trị giao dịch trong ngày

  "highestPrice": 24.35, // float // Giá khớp cao nhất trong ngày
  "lowestPrice": 24.15, // float // Giá khớp thấp nhất trong ngày
  "openPrice": 24.25, // float // Giá mở cửa

  "tradingSessionId": "40", // string //  Mã phiên giao dịch hiện tại 

  "time": {   // Thời gian trong các message giá của sàn trả về
    "Seconds": 1779762571,// integer // Unix timestamp seconds
    "Nanos": 101000000 // integer // Nanoseconds
  }
}  
```

**VD Payload nhận được Function Trade Extra**

```json lines
{
  "marketId": "DVX", // string // Mã thị trường niêm yết mã chứng khoán
  "boardId": "G1", // string // Mã bảng giao dịch
  "isin": "VN41I1G60005", // string // Mã định danh quốc tế
  "symbol": "41I1G6000", // string // Mã chứng khoán

  "matchPrice": 2022.5, // float // Giá khớp gần nhất
  "matchQtty": 1.0, // float // Khối lượng khớp gần nhất

  "side": "SELL", // string // Chiều mua, bán chủ động
  "avgPrice": 2023.92, // float // Giá khớp trung bình

  "totalVolumeTraded": 55913, // integer // Tổng khối lượng khớp trong ngày
  "grossTradeAmount": 11316.34193, // float // Tổng giá trị giao dịch trong ngày

  "highestPrice": 2028.0, // float // Giá khớp cao nhất trong ngày
  "lowestPrice": 2018.3, // float // Giá khớp thấp nhất trong ngày
  "openPrice": 2018.6, // float // Giá mở cửa

  "tradingSessionId": "40", // string // Mã phiên giao dịch hiện tại

  "time": {   // Thời gian trong các message giá của sàn trả về
    "Seconds": 1779766822, // integer // Unix timestamp seconds
    "Nanos": 72000000 // integer // Nanoseconds
  }
}
```

### Độ sâu thị trường (Quotes) {#quotes}

Cung cấp thông tin giá chào mua và chào bán tốt nhất của mã chứng khoán tại bảng giao dịch cụ thể, cập nhật theo thời
gian thực trong phiên giao dịch.

- Sàn HOSE hỗ trợ 3 mức giá.
- Sàn HNX, UPCOM hỗ trợ 10 mức giá.

#### Channel

>  **top_price.\{board_id\}.\{encoding\}**

#### Input
- `symbols`: Mã hoặc danh sách mã chứng khoán.
- `board_id`: Mã bảng giao dịch

#### Payload

```json lines
{
  "marketId": "DVX", // string // Mã thị trường niêm yết mã chứng khoán
  "boardId": "G1", // string // Mã bảng giao dịch
  "isin": "VN41I1G60005", // string // Mã định danh quốc tế
  "symbol": "41I1G6000", // string // Mã chứng khoán

  "bid": [
    {
      "price": 2023.4, // float // Giá chào mua cao nhất
      "quantity": 9.0 // float // Tổng khối lượng chào mua tại mức giá này
    },
    {
      "price": 2023.3, // float // Mức giá chào mua tiếp theo
      "quantity": 22.0 // float // Tổng khối lượng chào mua tại mức giá này
    },
    {
      "price": 2023.2, // float // Mức giá chào mua thấp hơn
      "quantity": 15.0 // float // Tổng khối lượng chào mua tại mức giá này
    }
  ],

  "offer": [
    {
      "price": 2023.6, // float // Giá chào bán thấp nhất
      "quantity": 3.0 // float // Tổng khối lượng chào bán tại mức giá tương ứng
    },
    {
      "price": 2023.7, // float // Mức giá chào bán tiếp theo
      "quantity": 62.0 // float // Tổng khối lượng chào bán tại mức giá tương ứng
    },
    {
      "price": 2023.8, // float // Mức giá chào bán cao hơn
      "quantity": 16.0 // float // Tổng khối lượng chào bán tại mức giá tương ứng
    }
  ],

  "totalOfferQtty": 8353.0, // float // Tổng khối lượng chào bán
  "totalBidQtty": 6447.0, // float // Tổng khối lượng chào mua

  "time": {   //Thời gian trong các message giá của sàn trả về
    "Seconds": 1779767143, // integer // Unix timestamp seconds
    "Nanos": 736000000 // integer // Nanoseconds fraction
  }
}
```
### OHLC {#ohlc}

OHLC cung cấp thông tin nến (open, high, low, close, volume) theo khung thời gian thực dưới dạng dữ liệu nến đang hình thành và được cập nhật liên tục theo các giao dịch phát sinh. Áp dụng cho Cổ phiếu (stock), Phái sinh (derivative) và Chỉ số thị trường (index) với nhiều khung thời gian (resolution).

#### Channel

>  **ohlc.\{resolution\}.\{encoding\}**

#### Input

- `symbols`: Mã hoặc danh sách mã chứng khoán hay chỉ số thị trường
  - Lưu ý: Đối với phái sinh, truyền lên `symbolType` (VD: VN30F1M) thay vì `symbol` (VD: 41I1G4000).
- `resolution`: Khung thời gian của nến (VD: 1, 3, 5, 15, 30, 1H, 1D, 1W)

#### Payload

*Cổ phiếu*

```json lines
{
  "time": 1757992500, //integer   // Thời gian bắt đầu nến
  "open": 30.4, //float     // Giá mở cửa
  "high": 30.4, //float     // Giá cao nhất trong nến
  "low": 30.25, //float     // Giá thấp nhất trong nến
  "close": 30.3,  //float     // Giá đóng cửa
  "volume": 1398200,  //integer   // Khối lượng giao dịch
  "symbol": "HPG",  //string    // Mã chứng khoán
  "resolution": "15", //string    // Khung thời gian nến
  "lastUpdated": 1757993014,  //integer   // Thời gian cập nhật lần cuối
  "type": "STOCK" //string    // Loại nhóm thị trường
}
```

*Phái sinh*

```json lines
{
  "time": 1757991840, //integer   // Thời gian bắt đầu nến
  "open": 1881.2,   //float     // Giá mở cửa
  "high": 1881.2,   //float     // Giá cao nhất trong nến
  "low": 1881.0,    //float     // Giá thấp nhất trong nến
  "close": 1881.2,  //float     // Giá đóng cửa
  "volume": "12",   //integer   // Khối lượng giao dịch
  "symbol": "VN30F1M",  //string    // Mã chứng khoán
  "resolution": "1",    //string    // Khung thời gian nến
  "lastUpdated": 1757991844,  //integer   // Thời gian cập nhật lần cuối
  "type": "DERIVATIVE"  //string    // Loại nhóm thị trường
}
```

*Chỉ số index*

```json lines
{
  "time": 1757988000, //integer   // Thời gian bắt đầu nến
  "open": 1696.87,  //float     // Giá mở cửa
  "high": 1696.87,  //float     // Giá cao nhất trong nến
  "low": 1686.02,   //float     // Giá thấp nhất trong nến
  "close": 1686.31,  //float     // Giá đóng cửa
  "volume": 435873728,  //integer   // Khối lượng giao dịch
  "symbol": "VNINDEX",  //string    // Mã chứng khoán
  "resolution": "1D",   //string    // Khung thời gian nến
  "lastUpdated": 1757993070,  //integer   // Thời gian cập nhật lần cuối
  "type": "INDEX" //string    // Loại nhóm thị trường
}
```

### OHLC đóng nến (OHLC Closed) {#ohlc-closed}

Cung cấp dữ liệu nến đã đóng theo từng khung thời gian (resolution). Dữ liệu chỉ gửi khi kết thúc mỗi khung thời gian và
không thay đổi sau đó.

#### Channel

>  **ohlc_closed.\{resolution\}.\{encoding\}**

#### Input

- `symbols`: Mã hoặc danh sách mã chứng khoán hay chỉ số thị trường
  - Lưu ý: Đối với phái sinh, truyền lên `symbolType` (VD: VN30F1M) thay vì `symbol` (VD: 41I1G4000).
- `resolution`: Khung thời gian của nến (VD: 1, 3, 5, 15, 30, 1H, 1D, 1W)

#### Payload

```json lines
{
  "time": 1757992500, //integer   // Thời gian bắt đầu nến
  "open": 30.4,       //float     // Giá mở cửa
  "high": 30.4,       //float     // Giá cao nhất trong nến
  "low": 30.25,       //float     // Giá thấp nhất trong nến
  "close": 30.3,        //float     // Giá đóng cửa
  "volume": 1398200,    //integer   // Khối lượng giao dịch
  "symbol": "HPG",      //string    // Mã chứng khoán
  "resolution": "15",   //string    // Khung thời gian nến
  "lastUpdated": 1757993014,    //integer   // Thời gian cập nhật lần cuối
  "type": "STOCK"     //string    // Loại nhóm thị trường
}
```

### Giá khớp dự kiến (Expected Price) {#expected-price}

Cung cấp thông tin giá đóng cửa, giá khớp dự kiến và khối lượng khớp dự kiến của mã chứng khoán trong các phiên giao
dịch khớp lệnh định kỳ ATO và ATC.

#### Channel

> **expected_price.\{board_id\}.\{encoding\}**

#### Input
- `symbols`: Mã hoặc danh sách mã chứng khoán.
- `board_id`: Mã bảng giao dịch

#### Payload

```json lines
{
  "marketId": "DVX",        //string    // Mã thị trường niêm yết mã chứng khoán
  "boardId": "G1",          //string    // Mã bảng giao dịch
  "symbol": "41I1G1000",    //string    // Mã chứng khoán
  "isin": "VN41I1G10000",   //string    // Mã định danh quốc tế
  "closePrice": 28.45,      //float     // Giá đóng cửa
  "expectedTradePrice": 28.45,  //float     // Giá dự khớp tại thời điểm xác định
  "expectedTradeQuantity": 133780, //integer   // Khối lượng dự khớp tại thời điểm xác định
  "time": {   // Thời gian trong các message giá của sàn trả về
    "Seconds": 1779694639, // integer // Unix timestamp seconds
    "Nanos": 736000000 // integer // Nanoseconds fraction
  }
}
```

### Chỉ số thị trường (Market Index) {#market-index}

Cung cấp thông tin chỉ số thị trường bao gồm giá trị chỉ số, mức thay đổi, độ rộng thị trường (số mã tăng/giảm/đi ngang)
và thanh khoản. Dữ liệu được cập nhật liên tục trong phiên giao dịch.

#### Channel

>  **market_index.\{market_index\}.\{encoding\}**

#### Input

- `market_index`: Tên chỉ số thị trường

#### Payload

```json lines
{
  "indexName": "VNINDEX", // string // Tên chỉ số thị trường
  "changedRatio": 0.41, // float // Tỷ lệ thay đổi (%)
  "changedValue": 6.84, // float // Giá trị thay đổi so với tham chiếu
  "fluctuationSteadinessIssueCount": 67, // integer // Số lượng mã có giá không đổi
  "fluctuationDownIssueCount": 158, // integer // Số lượng mã có giá giảm
  "fluctuationUpIssueCount": 144, // integer // Số lượng mã có giá tăng
  "fluctuationLowerLimitIssueCount": null, // integer // Số lượng mã giảm sàn
  "fluctuationUpperLimitIssueCount": 7, // integer // Số lượng mã tăng trần
  "fluctuationDownIssueVolume": 220246500, // integer // Tổng khối lượng giao dịch các mã có giá giảm
  "fluctuationUpIssueVolume": 446927155, // integer // Tổng khối lượng giao dịch các mã có giá tăng
  "fluctuationSteadinessIssueVolume": 39390038, // integer // Tổng khối lượng giao dịch các mã có giá không đổi
  "currencyCode": "VND", // string // Đơn vị tiền tệ
  "indexTypeCode": "001", // string // Mã loại chỉ số
  "lowestValueIndexes": 1662.05, // float // Giá thấp nhất trong phiên
  "highestValueIndexes": 1677.83, // float // Giá cao nhất trong phiên
  "priorValueIndexes": 1662.54, // float // Giá trị tham chiếu
  "valueIndexes": 1669.38, // float // Giá trị hiện tại của chỉ số
  "contauctAccTrdVal": 15609.88011093, // float // Tổng giá trị giao dịch theo phương thức khớp lệnh
  "contauctAccTrdVol": 606182599, // integer // Tổng khối lượng giao dịch theo phương thức khớp lệnh
  "blkTrdAccTrdVal": 3040.58723198, // float // Tổng giá trị giao dịch theo phương thức thỏa thuận
  "blkTrdAccTrdVol": 100381155, // integer // Tổng khối lượng giao dịch theo phương thức thỏa thuận
  "grossTradeAmount": 18650.46734291, // float // Tổng giá trị giao dịch trong ngày
  "totalVolumeTraded": 706563754, // integer // Tổng khối lượng giao dịch trong ngày
  "marketIndexClass": 1, // integer // Phân loại chỉ số
  "marketId": "STO", // string // Mã thị trường
  "tradingSessionId": "40", // string // Mã phiên giao dịch hiện tại
  "transactTime": {    // Thời gian trong các message giá của sàn trả về
    "Seconds": 1774940705, // integer // Unix timestamp seconds
    "Nanos": 0 // integer // Nanoseconds fraction
  }
}  
```

### Giao dịch nhà đầu tư nước ngoài (Foreign Investor) {#foreign-investor}

Cung cấp dữ liệu giao dịch của nhà đầu tư nước ngoài theo từng mã chứng khoán, bao gồm khối lượng và giá trị mua/bán,
tổng lũy kế trong ngày và room còn lại. Dữ liệu được cập nhật trong phiên giao dịch khi có thay đổi.

#### Channel

>  **foreign.\{board_id\}.\{encoding\}**

#### Input

- `symbol`: Mã hoặc danh sách mã chứng khoán.
- `board_id`: Mã bảng giao dịch (VD: G1 – lô chẵn).

#### Payload

```json lines
{
  "marketId": "STO", //string  // Mã thị trường
  "boardId": "G1",  //string  // Mã bảng giao dịch
  "tradingSessionId": "40", //string  // Mã phiên giao dịch hiện tại
  "symbol": "FPT",  //string  // Mã chứng khoán
  "transactTime": "035200011",  //string  // Thời điểm cập nhật
  "foreignInvestorTypeCode": "10",  //string  // Loại nhà đầu tư nước ngoài
  "sellVolume": 1449400,  //integer // Khối lượng bán trong ngày theo bảng giao dịch
  "sellTradedAmount": 109774810000, //float   // Giá trị bán trong ngày theo bảng giao dịch
  "buyVolume": 608300,  //integer // Khối lượng mua trong ngày theo bảng giao dịch
  "buyTradedAmount": 46040960000, //float   // Giá trị mua trong ngày theo bảng giao dịch
  "totalSellVolume": 1449716, //integer // Tổng khối lượng bán trong ngày
  "totalSellTradedAmount": 109798718600,  //float   // Tổng giá trị bán trong ngày
  "totalBuyVolume": 608370, //integer // Tổng khối lượng mua trong ngày
  "totalBuyTradedAmount": 46046280000,  //float   // Tổng giá trị mua trong ngày
  "foreignerOrderLimitQuantity": 341884580, //integer // Tổng room sở hữu tối đa của NĐT nước ngoài
  "foreignerBuyPossibleQuantity": 351900000 //integer // Room còn lại có thể mua
}
```

### VN30 dự tính (Estimated VN30) {#estimated-vn30}

Cung cấp thông tin chỉ số VN30 dự tính trong phiên giao dịch, bao gồm giá trị chỉ số dự tính, mức tăng/giảm so với giá tham chiếu và số lượng mã tăng, giảm, đứng giá. Dữ liệu được cập nhật theo thời gian thực khi có thay đổi.

#### Channel

>  **estimated_market_index.\{market_index\}.\{encoding\}**

#### Input

- `market_index`: Tên chỉ số thị trường. Hiện tại chỉ hỗ trợ VN30.

#### Payload

```json lines
{
  "indexName": "VN30",                      //string  // Tên chỉ số
  "valueIndexes": 1948.57,                  //float   // Giá trị VN30 dự tính
  "changedValue": -37.71,                   //float   // Mức tăng/giảm của chỉ số so với giá tham chiếu
  "changedRatio": -1.9,                     //float   // Tỷ lệ (%) tăng/giảm của chỉ số so với giá tham chiếu
  "fluctuationUpIssueCount": 5,             //integer // Số lượng mã có giá tăng
  "fluctuationDownIssueCount": 25,          //integer // Số lượng mã có giá giảm
  "fluctuationSteadinessIssueCount": 0,     //integer // Số lượng mã có giá không đổi 
  "grossTradeAmount": 6391.86,              //float   // Tổng giá trị giao dịch của các mã trong rổ VN30
  "totalVolumeTraded": 184907600,           //integer // Tổng khối lượng giao dịch của các mã trong rổ VN30
  "time": "2026-06-08 13:56:29.371"         //string  // Thời điểm cập nhật dữ liệu
}
```

### Thông tin phiên (Session) {#session}

Cung cấp thông tin phiên giao dịch của các sàn theo bảng giao dịch và nhóm sản phẩm thị trường.

#### Channel

>  **session.\{product_group_id\}.\{board_id\}.\{encoding\}**

#### Input

- `tsc_prod_grp_id`: Mã nhóm sản phẩm thị trường.
- `board_id`: Mã bảng giao dịch (VD: G1 – lô chẵn).

#### Payload

```json lines
{
  "marketId":"DVX",      // string // Mã thị trường niêm yết mã chứng khoán
  "boardId":"G1",        // string // Mã bảng giao dịch
  "eventId":"AB2",       // string // Mã sự kiện chuyển trạng thái phiên giao dịch
  "tradingSessionId":"40",  // string // Mã phiên giao dịch hiện tại
  "tscProdGrpId":"STO",         // string // Mã nhóm sản phẩm thị trường
  "time":"2025-10-01T02:15:00.487Z" //string  // Thời điểm cập nhật dữ liệu
}
```