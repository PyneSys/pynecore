## Sổ lệnh

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getOrders"></span>

### `GET /accounts/{accountNo}/orders`

Lấy sổ lệnh giao dịch theo từng loại lệnh tương ứng thị trường cơ sở, phái sinh hay trái phiếu, bao gồm trạng thái và thông tin xử lý của từng lệnh.

- Cơ sở (STOCK): lệnh thường (NORMAL), lệnh dừng có điều kiện (STOP)
    
- Phái sinh (DERIVATIVE): lệnh thường (NORMAL), lệnh dừng có điều kiện (STOP), lệnh OCO (OCO)
    
- Trái phiếu (BOND): lệnh thường (NORMAL)

<h3 id="getorders-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|string|true|Loại thị trường |
|orderCategory|query|string|true|Phân loại lệnh |
|pageIndex|query|integer|true|Kích thước trang dữ liệu |
|pageSize|query|integer|true|Số bản ghi trên mỗi trang|
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|version|header|string(date)|false|API version (YYYY-MM-DD)|
|accountNo|path|string|true|Số tiểu khoản|

#### Detailed descriptions

**marketType**: Loại thị trường 
- STOCK: Giao dịch cơ sở
- DERIVATIVE: Giao dịch phái sinh
- BOND: Giao dịch trái phiếu

**orderCategory**: Phân loại lệnh 
- NORMAL: lệnh thường  (cơ sở, phái sinh, trái phiếu)
- STOP: lệnh dừng có điều kiện (cơ sở, phái sinh)
- OCO: lệnh OCO (phái sinh)

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/accounts/{accountNo}/orders?marketType=STOCK&orderCategory=NORMAL&pageIndex=0&pageSize=10 \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/accounts/{accountNo}/orders?marketType=STOCK&orderCategory=NORMAL&pageIndex=0&pageSize=10 HTTP/1.1
Host: openapi.dnse.com.vn
Accept: application/json
X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==
X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000
X-Signature: your_signature
version: 2026-07-23

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Accept": []string{"application/json"},
        "X-API-Key": []string{"eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ=="},
        "X-Aux-Date": []string{"Mon, 19 Jan 2026 07:45:23 +0000"},
        "X-Signature": []string{"your_signature"},
        "version": []string{"2026-07-23"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/accounts/{accountNo}/orders", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

```javascript

const headers = {
  'Accept':'application/json',
  'X-API-Key':'eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==',
  'X-Aux-Date':'Mon, 19 Jan 2026 07:45:23 +0000',
  'X-Signature':'your_signature',
  'version':'2026-07-23'
};

fetch('https://openapi.dnse.com.vn/accounts/{accountNo}/orders?marketType=STOCK&orderCategory=NORMAL&pageIndex=0&pageSize=10',
{
  method: 'GET',

  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```python
import requests
headers = {
  'Accept': 'application/json',
  'X-API-Key': 'eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==',
  'X-Aux-Date': 'Mon, 19 Jan 2026 07:45:23 +0000',
  'X-Signature': 'your_signature',
  'version': '2026-07-23'
}

r = requests.get('https://openapi.dnse.com.vn/accounts/{accountNo}/orders', params={
  'marketType': 'STOCK',  'orderCategory': 'NORMAL',  'pageIndex': '0',  'pageSize': '10'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/accounts/{accountNo}/orders?marketType=STOCK&orderCategory=NORMAL&pageIndex=0&pageSize=10");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("GET");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

> Example responses

> OK

```json
{
  "orders": [
    {
      "id": 141,
      "side": "NB",
      "accountNo": "0001179019",
      "symbol": "BCM",
      "price": 51200,
      "priceSecure": 51200,
      "averagePrice": 0,
      "quantity": 300,
      "fillQuantity": 0,
      "canceledQuantity": 0,
      "leaveQuantity": 300,
      "orderType": "LO",
      "orderCategory": "NORMAL",
      "orderStatus": "New",
      "loanPackageId": 7937,
      "marketType": "STOCK",
      "transDate": "2026-03-16",
      "createdDate": "2026-03-24T03:15:22.226297778Z",
      "modifiedDate": "2026-03-24T03:15:22.358568266Z"
    }
  ],
  "pageIndex": 0,
  "pageSize": 10,
  "totalPages": 1,
  "totalRecords": 1
}
```

```json
{
  "orders": [
    {
      "id": 2286,
      "side": "NB",
      "accountNo": "0001179019",
      "symbol": "41I1G9000",
      "price": 1990,
      "priceSecure": 1990,
      "averagePrice": 0,
      "quantity": 3,
      "fillQuantity": 0,
      "canceledQuantity": 0,
      "leaveQuantity": 3,
      "orderType": "LO",
      "orderCategory": "NORMAL",
      "orderStatus": "New",
      "loanPackageId": 2278,
      "marketType": "DERIVATIVE",
      "transDate": "2026-07-31",
      "createdDate": "2026-07-31T03:15:22.226297778Z",
      "modifiedDate": "2026-07-31T03:15:22.358568266Z"
    }
  ],
  "pageIndex": 0,
  "pageSize": 10,
  "totalPages": 1,
  "totalRecords": 1
}
```

```json
{
  "orders": [
    {
      "accountNo": "0001179019",
      "conditionOperator": ">=",
      "createdDate": "2026-07-30T15:07:17.616863Z",
      "durationDateTime": "2026-08-20T07:30:00Z",
      "durationType": "GTD",
      "id": "d9lmh9fqs0csemaoe8tg",
      "loanPackageId": 1306,
      "marketType": "DERIVATIVE",
      "modifiedDate": "2026-07-30T15:07:17.616863Z",
      "orderCategory": "STOP",
      "orderStatus": "New",
      "orderType": "LO",
      "price": 1900,
      "quantity": 3,
      "side": "NS",
      "stopPrice": 1700,
      "symbol": "41I1G8000"
    },
    {
      "accountNo": "0001179019",
      "conditionOperator": ">=",
      "createdDate": "2026-07-30T10:43:31.246312Z",
      "durationDateTime": "2026-07-31T07:30:00Z",
      "durationType": "GTD",
      "id": "d9lilkvqs0csemaodq3g",
      "loanPackageId": 1306,
      "marketType": "DERIVATIVE",
      "modifiedDate": "2026-07-30T10:43:31.246312Z",
      "orderCategory": "STOP",
      "orderStatus": "New",
      "orderType": "LO",
      "price": 1980.5,
      "quantity": 1,
      "side": "NB",
      "stopPrice": 2000,
      "symbol": "41I1G8000"
    }
  ],
  "pageIndex": 0,
  "pageSize": 2,
  "totalPages": 4,
  "totalRecords": 8
}
```

```json
{
  "orders": [
    {
      "accountNo": "0001179019",
      "createdDate": "2026-07-22T07:41:30.357676Z",
      "durationType": "DAY",
      "id": "d9g78ahqn51s72sdntt0",
      "loanPackageId": 2278,
      "marketType": "DERIVATIVE",
      "modifiedDate": "2026-07-22T07:41:30.427436Z",
      "orderCategory": "OCO",
      "orderStatus": "Expired",
      "orderType": "LO",
      "price": 1990,
      "quantity": 3,
      "side": "NB",
      "stopOrderPrice": 2005,
      "stopPrice": 2000,
      "symbol": "41I1G8000"
    }
  ],
  "pageIndex": 0,
  "pageSize": 10,
  "totalPages": 1,
  "totalRecords": 1
}
```

> 400 Response

```json
{
  "code": "OA-003",
  "message": "Thông tin nhập không hợp lệ",
  "status": 400
}
```

<h3 id="getorders-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» orders|[object]|false|none|Danh sách lệnh giao dịch|
|»» id|integer(int32)|false|none|Id lệnh giao dịch|
|»» side|string|false|none|Chiều giao dịch<br>NB: Mua<br>NS: Bán|
|»» accountNo|string|false|none|Số tiểu khoản|
|»» symbol|string|false|none|Mã chứng khoán|
|»» price|number(double)|false|none|Giá đặt|
|»» priceSecure|number(double)|false|none|Giá dùng để kiểm tra sức mua/đặt lệnh|
|»» averagePrice|number(double)|false|none|Giá khớp trung bình|
|»» quantity|integer(int32)|false|none|Khối lượng đặt|
|»» fillQuantity|integer(int32)|false|none|Khối lượng đã khớp|
|»» canceledQuantity|integer(int32)|false|none|Khối lượng đã hủy|
|»» leaveQuantity|integer(int32)|false|none|Khối lượng còn lại|
|»» orderType|string|false|none|Loại lệnh<br>- LO: Lệnh giới hạn<br>- MOK/MAK/MTL: Lệnh thị trường<br>- ATO/ATC: Lệnh phiên định kỳ mở cửa/đóng cửa<br>- PLO: Lệnh khớp lệnh sau giờ|
|»» orderCategory|string|false|none|Phân loại lệnh <br>- NORMAL: lệnh thường  (cơ sở, phái sinh, trái phiếu)<br>- STOP: lệnh dừng có điều kiện (cơ sở, phái sinh)<br>- OCO: lệnh OCO (phái sinh)|
|»» orderStatus|string|false|none|Trạng thái lệnh NORMAL:<br><br>  - Pending/PendingNew: Chờ gửi<br><br>  - New: Chờ khớp<br><br>  - PendingReplace: Chờ sửa<br><br>  - PendingCancel: Chờ hủy<br><br>  - PartiallyFilled: Khớp một phần<br><br>  - Filled: Khớp toàn bộ<br><br>  - Canceled: Đã hủy<br><br>  - Rejected: Bị từ chối<br><br>  - Expired: Hết hạn trong phiên<br><br>  - DoneForDay: Lệnh được giải tỏa do không khớp trong phiên<br><br>Trạng thái lệnh STOP/OCO:<br><br>  - New: Chờ kích hoạt                              <br>  - Activated: Đã kích hoạt<br>  - Cancelled: Đã hủy<br>  - Expired: Hết hiệu lực<br>  - Rejected: Bị từ chối<br>  - Failed: Lệnh thất bại|
|»» loanPackageId|integer(int32)|false|none|Gói vay áp dụng cho mã chứng khoán|
|»» marketType|string|false|none|Loại thị trường<br>- STOCK: Lệnh cơ sở<br>- DERIVATIVE: Lệnh phái sinh|
|»» transDate|string|false|none|Ngày giao dịch|
|»» createdDate|string(date-time)|false|none|Thời điểm tạo|
|»» modifiedDate|string(date-time)|false|none|Thời điểm cập nhật|

Status Code **400**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» code|string|false|none|none|
|» message|string|false|none|none|
|» status|integer|false|none|none|

Status Code **500**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» code|string|false|none|none|
|» message|string|false|none|none|
|» status|integer|false|none|none|
