---
sidebar_position: 4
---


# Lệnh giao dịch
---

## Tổng quan

### Phân loại theo thị trường (`marketType`)

| marketType | Mô tả                           |
|------------|---------------------------------|
| STOCK      | Giao dịch chứng khoán cơ sở     |
| DERIVATIVE | Giao dịch chứng khoán phái sinh |
| BOND       | Giao dịch trái phiếu niêm yết   |

### Phân loại theo loại lệnh (`orderCategory`)

| orderCategory | Mô tả                            |
|---------------|----------------------------------|
| NORMAL        | Lệnh thông thường                |
| STOP          | Lệnh dừng có điều kiện           |
| OCO           | Lệnh OCO (One Cancels the Other) |

### Bảng hỗ trợ

| marketType | NORMAL | STOP | OCO |
|------------|:------:|:----:|:---:|
| STOCK      |   ✅    |  ✅   |  ❌  |
| DERIVATIVE |   ✅    |  ✅   |  ✅  |
| BOND       |   ✅    |  ❌   |  ❌  |

## Lệnh thường (NORMAL)

### Vòng đời lệnh thường

Vòng đời lệnh thường mô tả các trạng thái mà một lệnh có thể đi qua kể từ lúc bạn gửi yêu cầu đến khi kết thúc.

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/sd3.png)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/sd3.png)
</div>

### Trạng thái lệnh

| Trạng thái          | Giải nghĩa            | Chú thích                                                                                 |
|---------------------|-----------------------|-------------------------------------------------------------------------------------------|
| **pending**         | Lệnh mới được tạo     | Lệnh vừa gửi lên hệ thống, đang được kiểm tra và xử lý nội bộ                             |
| **pendingNew**      | Lệnh chờ gửi lên Sở   | Lệnh hợp lệ và đang chờ gửi lên hệ thống Sở giao dịch                                     |
| **new**             | Lệnh chờ khớp         | Lệnh được Sở ghi nhận và đang chờ khớp theo điều kiện thị trường                          |
| **partiallyFilled** | Lệnh đã khớp một phần | Một phần khối lượng đã khớp, phần còn lại tiếp tục chờ khớp                               |
| **filled**          | Lệnh khớp toàn bộ     | Toàn bộ khối lượng lệnh đã được khớp thành công                                           |
| **pendingReplace**  | Lệnh chờ sửa          | Yêu cầu sửa lệnh được ghi nhận, đang chờ hệ thống/Sở xử lý thay đổi                       |
| **pendingCancel**   | Lệnh chờ hủy          | Yêu cầu hủy lệnh đang chờ hệ thống/Sở xử lý                                               |
| **canceled**        | Lệnh hủy thành công   | Lệnh đã được hủy thành công và không còn hiệu lực giao dịch                               |
| **rejected**        | Lệnh bị từ chối       | Lệnh không được chấp nhận do không đáp ứng điều kiện (gói vay, sức mua, hạn mức cho vay…) |
| **expired**         | Lệnh hết hạn          | Lệnh hết hiệu lực do kết thúc phiên hoặc quá thời gian hiệu lực mà chưa được khớp         |
| **doneForDay**      | Lệnh đã được giải tỏa | Lệnh kết thúc vòng đời trong ngày giao dịch                                               |


### Đặt lệnh thường

Dưới đây là các thông tin bắt buộc cần gửi đối với một yêu cầu (Request) đặt lệnh.
- **`marketType`**: Phân loại giao dịch
    - `STOCK`: giao dịch chứng khoán cơ sở
    - `DERIVATIVE`: giao dịch chứng khoán phái sinh
    - `BOND`: giao dịch trái phiếu niêm yết
- **`orderCategory`**: Loại lệnh thường trong ngày (NORMAL)
- **`accountNo`:** Tiểu khoản giao dịch, được trả trong response Endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-accounts">Tài khoản giao dịch.</a>
- **`symbol`**: Mã chứng khoán giao dịch
- **`loanPackageId`**: Gói vay giao dịch, xem thêm thông tin về gói vay <a href="https://developers.dnse.com.vn/docs/guide/trading-api/dnse_margin#gói-vay-loan-packages">tại đây.</a>
- **`side`**: Chiều Mua (NB) hoặc Bán (NS)
- **`orderType`**: Loại lệnh tương ứng với sàn giao dịch
    - Sàn HOSE: ATO, ATC, LO, MTL
    - Sàn HNX: LO, MTL, MOK, MAK, ATC, PLO
    - Sàn Upcom: LO
- **`quantity`**: Khối lượng đặt
    - Khối lượng đặt không vượt quá khối lượng tối đa có thể mua (`qmaxBuy`)hoặc có thể bán (`qmaxSell`) trên tiểu khoản giao dịch, người dùng truy vấn thông tin đối với từng mã chứng khoán qua Endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-ppse">/Sức mua, sức bán.</a>
    - Với giao dịch cơ sở, khối lượng đặt là lô chẵn (100,200,...) hoặc lô lẻ (1,2,..99). Khối lượng lẻ lô (101,102,...) là không hợp lệ.
- **`price`**: Giá đặt
    - Nếu loại lệnh là LO, giá đặt phải > 0 và phải nằm trong khoảng giá trần sàn của mã chứng khoán tại phiên giao dịch đó.
    - Nếu loại lệnh khác LO, giá đặt truyền lên luôn = 0.


<details>
  <summary>VD Request đặt lệnh thường NORMAL</summary>

```json lines
{
  "method": "POST",
  "path": "/accounts/:accountNo/orders",
  "query": {
    "marketType": "STOCK",
    "orderCategory": "NORMAL"
  },
  "headers": {
    "x-api-key": "lB58g6iWzyrNx2EhwwQXeYeoAnkzlaXkJWi",    // APIkey được cấp khi đăng ký dịch vụ
    "x-signature": "fjsdhfryt6aaa6c91a8f88b472c9721fde161e0d89df8c",    // Chữ ký số theo thuật toán HMAC SHA256
    "trading-token": "7ceef658-9f01-414e-8b3e-faa77bb9061e",    // Token đặt lệnh         
    "date": "Fri, 16 Jan 2026 07:11:30 +0000",    // Thời gian tạo yêu cầu (UTC)
    "version": "2026-07-23" // API version (YYYY-MM-DD)
  },
  "body": {
    "symbol": "HPG",         // Mã chứng khoán đặt lệnh
    "side": "NB",            // Chiều lệnh giao dịch 
    "orderType": "LO",       // Loại lệnh giao dịch
    "price": 25950,          // Giá đặt
    "quantity": 100,         // Khối lượng đặt
    "loanPackageId": 5757   // Mã gói vay 
  }
}
```
</details>

Khi lệnh khớp mua, hệ thống hình thành các vị thế Positions (hay còn gọi là danh mục tài sản) theo cặp `symbol` - `loanPackage`. Nếu mua cùng mã nhưng khác gói vay → tạo Positions tách biệt (rủi ro được quản trị riêng).

### Sửa lệnh

**Điều kiện chung:**
- Chỉ được sửa lệnh LO trong phiên giao dịch liên tục và áp dụng cho lệnh ở trạng thái Chờ khớp (New) hoặc Đã khớp một phần (PartiallyFilled)
- Giá sửa phải nằm trong biên độ trần sàn của mã chứng khoán vào phiên giao dịch đó.
- Nếu giá hoặc khối lượng sau khi sửa vượt quá sức mua /sức bán cho phép, yêu cầu sửa lệnh sẽ bị từ chối.

**Sửa lệnh cơ sở:**
- Khi sửa lệnh thành công, hệ thống hủy lệnh hiện tại và đặt lại một lệnh mới với thông tin đã chỉnh sửa.
- Cho phép sửa đồng thời giá và khối lượng.
- Thứ tự ưu tiên của lệnh sau khi sửa sẽ được xác định lại theo thời điểm ghi nhận sửa lệnh thành công.

**Sửa lệnh phái sinh:**
- Người dùng chỉ được phép sửa hoặc giá hoặc khối lượng trong một yêu cầu.
- Khối lượng sửa phải lớn hơn khối lượng đã khớp (nếu lệnh đã khớp một phần).
- Nếu khối lượng sửa lớn hơn khối lượng ban đầu, thứ tự ưu tiên của lệnh sẽ được thay đổi.

<details>
  <summary>VD Request sửa lệnh</summary>

```json lines
{
  "method": "PUT",
  "path": "/accounts/:accountNo/orders/:orderId",
  "query": {
    "marketType": "STOCK",
    "orderCategory": "NORMAL"
  },
  "headers": {
    "x-api-key": "lB58g6iWzyrNx2EhwwQXeYeoAnkzlaXkJWi",    // APIkey được cấp khi đăng ký dịch vụ
    "x-signature": "fjsdhfryt6aaa6c91a8f88b472c9721fde161e0d89df8c",    // Chữ ký số theo thuật toán HMAC SHA256
    "trading-token": "7ceef658-9f01-414e-8b3e-faa77bb9061e",    // Token đặt lệnh         
    "date": "Fri, 16 Jan 2026 07:11:30 +0000",    // Thời gian tạo yêu cầu (UTC)
    "version": "2026-07-23" // API version (YYYY-MM-DD)
},
  "body": {
    "price": 25950,          // Giá sửa
    "quantity": 100         // Khối lượng sửa
  }
}
```
</details>

## Lệnh STOP

Lệnh STOP là lệnh điều kiện, khi giá thị trường đạt đến mức giá kích hoạt (stopPrice), hệ thống sẽ tự động tạo một lệnh giao dịch thông thường (NORMAL) với thông số mà người dùng đã thiết lập trước và gửi lên Sở giao dịch để thực hiện khớp lệnh.

### Trạng thái lệnh

| Trạng thái                 | Giải nghĩa           | Chú thích                                                                                  |
|----------------------------|----------------------|--------------------------------------------------------------------------------------------|
| **new**                    | Lệnh mới tạo         | Lệnh STOP được tạo thành công và đang chờ điều kiện kích hoạt                              |
| **activated**              | Đã kích hoạt         | Đạt điều kiện kích hoạt, hệ thống tạo và gửi lệnh thường (NORMAL) lên Sở giao dịch         |
| **cancelled**              | Đã hủy               | Lệnh đã hủy thành công                                                                     |
| **cancelledByRightsEvent** | Hủy do sự kiện quyền | Lệnh bị hủy tự động do phát sinh sự kiện quyền của mã chứng khoán                          |
| **rejected**               | Từ chối              | Lệnh không được hệ thống chấp nhận khi tạo                                                 |
| **failed**                 | Kích hoạt thất bại   | Điều kiện kích hoạt đã xảy ra nhưng hệ thống không thể tạo hoặc gửi lệnh NORMAL thành công |
| **expired**                | Hết hiệu lực         | Lệnh hết thời hạn hiệu lực và chưa được kích hoạt                                          | 

Sau khi lệnh STOP chuyển sang trạng thái `activated`, hệ thống sẽ sinh lệnh giao dịch thông thường (NORMAL). Từ thời điểm này, lệnh NORMAL sẽ tuân theo vòng đời và trạng thái của lệnh giao dịch thông thường.

### Đặt lệnh STOP

Dưới đây là các thông tin bắt buộc cần gửi đối với một yêu cầu (Request) đặt lệnh STOP.
- **`marketType`**: Phân loại giao dịch
    - `STOCK`: giao dịch cơ sở
    - `DERIVATIVE`: giao dịch phái sinh
- **`orderCategory`**: STOP (lệnh dừng có điều kiện)
- **`accountNo`:** Tiểu khoản giao dịch, được trả trong response Endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-accounts">Tài khoản giao dịch.</a>
- **`symbol`**: Mã chứng khoán giao dịch
- **`loanPackageId`**: Gói vay giao dịch, xem thêm thông tin về gói vay <a href="https://developers.dnse.com.vn/docs/guide/trading-api/dnse_margin#gói-vay-loan-packages">tại đây.</a>
- **`stopPrice`**: Giá kích hoạt lệnh STOP. Khi giá thị trường thỏa mãn điều kiện kích hoạt (`conditionOperator`), hệ thống sẽ tự động tạo và gửi lệnh giao dịch thông thường (NORMAL).
- **`conditionOperator`**: Điều kiện so sánh giữa giá thị trường và `stopPrice` để kích hoạt lệnh. Giá trị hỗ trợ:
    - `>=`: Kích hoạt khi giá thị trường lớn hơn hoặc bằng `stopPrice`.
    - `<=`: Kích hoạt khi giá thị trường nhỏ hơn hoặc bằng `stopPrice`.
- **`side`**: Chiều mua (NB) hoặc bán (NS) của lệnh thường
- **`orderType`**: LO hoặc MTL
- **`quantity`**: Khối lượng đặt thường
    - Với giao dịch cơ sở, khối lượng đặt là lô chẵn (100,200,...) hoặc lô lẻ (1,2,..99). Khối lượng lẻ lô (101,102,...) là không hợp lệ.
- **`price`**: Giá đặt lệnh thường
    - Nếu loại lệnh là LO, giá đặt phải > 0
    - Nếu loại lệnh là MTL, giá đặt truyền lên luôn = 0
- **`durationType`**: Loại thời hạn hiệu lực của lệnh STOP
    - `GTD`: Lệnh có hiệu lực đến thời điểm được chỉ định tại `durationDateTime`.
- **`durationDateTime`**: Thời điểm hết hiệu lực của lệnh STOP theo định dạng ISO 8601 (yyyy-MM-dd'T'HH:mm:ss.SSSXXX). Sau thời điểm này, nếu lệnh chưa được kích hoạt thì trạng thái sẽ chuyển sang `expired`.

<details>
  <summary>VD Yêu cầu đặt lệnh STOP</summary>

```json lines
{
  "method": "POST",
  "path": "/accounts/:accountNo/orders",
  "query": {
    "marketType": "STOCK",
    "orderCategory": "STOP"
  },
  "headers": {
    "x-api-key": "lB58g6iWzyrNx2EhwwQXeYeoAnkzlaXkJWi",    // APIkey được cấp khi đăng ký dịch vụ
    "x-signature": "fjsdhfryt6aaa6c91a8f88b472c9721fde161e0d89df8c",    // Chữ ký số theo thuật toán HMAC SHA256
    "trading-token": "7ceef658-9f01-414e-8b3e-faa77bb9061e",    // Token đặt lệnh         
    "date": "Fri, 16 Jan 2026 07:11:30 +0000",    // Thời gian tạo yêu cầu (UTC)
    "version": "2026-07-23" // API version (YYYY-MM-DD)
},
  "body": {
    "symbol": "HPG",         // Mã chứng khoán đặt lệnh
    "side": "NB",            // Chiều lệnh thường NORMAL
    "orderType": "LO",       // Loại lệnh thường NORMAL
    "price": 25950,          // Giá đặt lệnh thường NORMAL
    "quantity": 100,         // Khối lượng đặt thường NORMAL
    "loanPackageId": 5757,   // Mã gói vay thường NORMAL 
    "stopPrice": 28100,      // Giá kích hoạt lệnh STOP
    "conditionOperator": ">=",  // Điều kiện kích hoạt giữa giá kích hoạt và giá thị trường
    "durationType": "GTD",      // Loại thời hạn hiệu lực
    "durationDateTime": "2026-07-01T07:30:00.000+07:00"  // Thời điểm hết hiệu lực lệnh STOP 
  }    
}
```
</details>

## Lệnh OCO

Lệnh OCO (One Cancels the Other) là lệnh điều kiện kết hợp giữa chốt lời (Take Profit) và cắt lỗ (Stop Loss) trong cùng một yêu cầu đặt lệnh.

Cơ chế hoạt động:

- Người dùng gửi yêu cầu đặt lệnh OCO.
- Khi lệnh OCO được kích hoạt, hệ thống tạo một lệnh LO với giá chốt lời và gửi lên Sở giao dịch.
- Trong thời gian lệnh LO đang chờ khớp, hệ thống tiếp tục theo dõi giá thị trường.
- Nếu giá thị trường đạt điều kiện cắt lỗ trước khi lệnh LO được khớp toàn bộ, hệ thống tự động sửa giá của lệnh LO sang giá cắt lỗ.
- Tại một thời điểm chỉ có một lệnh LO được gửi lên Sở giao dịch.

### Đặt lệnh OCO

Dưới đây là các thông tin bắt buộc cần gửi đối với một yêu cầu (Request) đặt lệnh OCO.
- **`marketType`**: Phân loại giao dịch
    - `DERIVATIVE`: chỉ áp dụng giao dịch phái sinh
- **`orderCategory`**: `OCO`
- **`accountNo`:** Tiểu khoản giao dịch, được trả trong response Endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-accounts">Tài khoản giao dịch.</a>
- **`symbol`**: Mã chứng khoán giao dịch
- **`loanPackageId`**: Gói vay giao dịch, xem thêm thông tin về gói vay <a href="https://developers.dnse.com.vn/docs/guide/trading-api/dnse_margin#gói-vay-loan-packages">tại đây.</a>
- **`side`**: Chiều mua (NB) hoặc bán (NS) của lệnh thường
- **`quantity`**: Khối lượng đặt
- **`price`**: Giá chốt lời
    - Với lệnh Mua (NB): Giá chốt lời phải < Giá thị trường
    - Với lệnh Bán (NS): Giá chốt lời phải > Giá thị trường
- **`stopPrice`**: Giá kích hoạt lệnh cắt lỗ
- **`stopOrderPrice`**: Giá đặt cắt lỗ:
    - Với lệnh Mua (NB): Giá đặt cắt lỗ > Giá thị trường
    - Với lệnh Bán (NS): Giá đặt cắt lỗ < Giá thị trường
- **`durationType`**: Loại thời hạn hiệu lực của lệnh STOP
    - `DAY`: Lệnh có hiệu lực trong ngày

<details>
  <summary>VD Request đặt lệnh OCO</summary>

```json lines
{
  "method": "POST",
  "path": "/accounts/:accountNo/orders",
  "query": {
    "marketType": "DERIVATIVE",
    "orderCategory": "OCO"
  },
  "headers": {
    "x-api-key": "lB58g6iWzyrNx2EhwwQXeYeoAnkzlaXkJWi",    // APIkey được cấp khi đăng ký dịch vụ
    "x-signature": "fjsdhfryt6aaa6c91a8f88b472c9721fde161e0d89df8c",    // Chữ ký số theo thuật toán HMAC SHA256
    "trading-token": "7ceef658-9f01-414e-8b3e-faa77bb9061e",    // Token đặt lệnh         
    "date": "Fri, 16 Jan 2026 07:11:30 +0000",    // Thời gian tạo yêu cầu (UTC)
    "version": "2026-07-23" // API version (YYYY-MM-DD)
},
  "body": {
    "symbol": "41I1G900",          // Mã chứng khoán đặt lệnh
    "side": "NB",             // Chiều lệnh 
    "price": 1925,            // Giá đặt chốt lời
    "quantity": 3,          // Khối lượng đặt thường NORMAL
    "loanPackageId": 2278,    // Mã gói vay phái sinh
    "stopPrice": 1934,        // Giá kích hoạt lệnh cắt lỗ
    "stopOrderPrice": 1936,    // Giá đặt cắt lỗ
    "durationType": "DAY"      // Loại thời hạn hiệu lực
  }    
}
```
</details>
