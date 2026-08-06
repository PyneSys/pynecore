## Hủy lệnh

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="cancelOrder"></span>

### `DELETE /accounts/{accountNo}/orders/{orderId}`

Gửi yêu cầu hủy lệnh đã đặt theo `orderId.`

Hỗ trợ hủy các loại lệnh:

- Lệnh thường NORMAL có trạng thái: chờ gửi, chờ khớp hoặc đã khớp 1 phần trong thời gian quy định
    
- Lệnh STOP /OCO có trạng thái: chờ kích hoạt

<h3 id="cancelorder-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|string|true|Loại thị trường |
|orderCategory|query|string|true|Phân loại lệnh |
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|trading-token|header|string|true|Token đặt lệnh|
|version|header|string(date)|true|API version (YYYY-MM-DD)|
|accountNo|path|string|true|Số tiểu khoản|
|orderId|path|integer|true|Mã lệnh giao dịch|

#### Detailed descriptions

**marketType**: Loại thị trường 
- STOCK: Đặt lệnh cơ sở
- DERIVATIVE: Đặt lệnh phái sinh

**orderCategory**: Phân loại lệnh 
- NORMAL: lệnh thường  (cơ sở, phái sinh, trái phiếu)
- STOP: lệnh dừng có điều kiện (cơ sở, phái sinh)
- OCO: lệnh OCO (phái sinh)

> Code samples

```shell
# You can also use wget
curl -X DELETE https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}?marketType=DERIVATIVE&orderCategory=STOP \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Signature: your_signature' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'trading-token: 7ceef658-9f01-414e-8b3e-faa77bb9061e' \
  -H 'version: 2026-07-23'

```

```http
DELETE https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}?marketType=DERIVATIVE&orderCategory=STOP HTTP/1.1
Host: openapi.dnse.com.vn
Accept: application/json
X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==
X-Signature: your_signature
X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000
trading-token: 7ceef658-9f01-414e-8b3e-faa77bb9061e
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
        "X-Signature": []string{"your_signature"},
        "X-Aux-Date": []string{"Mon, 19 Jan 2026 07:45:23 +0000"},
        "trading-token": []string{"7ceef658-9f01-414e-8b3e-faa77bb9061e"},
        "version": []string{"2026-07-23"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("DELETE", "https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}", data)
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
  'X-Signature':'your_signature',
  'X-Aux-Date':'Mon, 19 Jan 2026 07:45:23 +0000',
  'trading-token':'7ceef658-9f01-414e-8b3e-faa77bb9061e',
  'version':'2026-07-23'
};

fetch('https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}?marketType=DERIVATIVE&orderCategory=STOP',
{
  method: 'DELETE',

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
  'X-Signature': 'your_signature',
  'X-Aux-Date': 'Mon, 19 Jan 2026 07:45:23 +0000',
  'trading-token': '7ceef658-9f01-414e-8b3e-faa77bb9061e',
  'version': '2026-07-23'
}

r = requests.delete('https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}', params={
  'marketType': 'DERIVATIVE',  'orderCategory': 'STOP'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}?marketType=DERIVATIVE&orderCategory=STOP");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("DELETE");
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
  "id": "741",
  "accountNo": "0001179019",
  "orderCategory": "NORMAL",
  "marketType": "STOCK",
  "symbol": "SHS",
  "side": "NB",
  "orderType": "LO",
  "orderStatus": "PendingCancel",
  "price": 15600,
  "quantity": 800,
  "loanPackageId": 1372,
  "priceSecure": 15600,
  "averagePrice": 0,
  "fillQuantity": 0,
  "canceledQuantity": 0,
  "leaveQuantity": 800,
  "transDate": "2026-01-30",
  "createdDate": "2026-08-04T02:52:57.305213488Z",
  "modifiedDate": "2026-08-04T02:53:09.831996595Z"
}
```

```json
{
  "accountNo": "0001179019",
  "conditionOperator": ">=",
  "createdDate": "2026-08-04T02:56:19.530727Z",
  "durationDateTime": "2026-08-08T07:30:00Z",
  "durationType": "GTD",
  "id": "d9ol9kq0cvks72pfqiug",
  "loanPackageId": 2278,
  "marketType": "DERIVATIVE",
  "modifiedDate": "2026-08-04T02:56:43.821292Z",
  "orderCategory": "STOP",
  "orderStatus": "Canceled",
  "orderType": "LO",
  "price": 1991,
  "quantity": 4,
  "side": "NS",
  "stopPrice": 1990,
  "symbol": "41I1G9000"
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

<h3 id="cancelorder-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» id|string|false|none|Id lệnh giao dịch|
|» accountNo|string|false|none|Số tiểu khoản|
|» orderCategory|string|false|none|Phân loại lệnh <br>- NORMAL: lệnh thường  (cơ sở, phái sinh, trái phiếu)<br>- STOP: lệnh dừng có điều kiện (cơ sở, phái sinh)<br>- OCO: lệnh OCO (phái sinh)|
|» marketType|string|false|none|Loại thị trường<br>- STOCK: Lệnh cơ sở<br>- DERIVATIVE: Lệnh phái sinh<br>- BOND: Lệnh phái sinh|
|» symbol|string|false|none|Mã chứng khoán|
|» side|string|false|none|Chiều đặt lệnh<br>- NB: Mua<br>- NS: Bán|
|» orderType|string|false|none|Loại lệnh<br>- LO: Lệnh giới hạn<br>- MOK/MAK/MTL: Lệnh thị trường<br>- ATO/ATC: Lệnh phiên định kỳ mở cửa/đóng cửa<br>- PLO: Lệnh khớp lệnh sau giờ|
|» orderStatus|string|false|none|Trạng thái lệnh NORMAL:<br><br>  - Pending/PendingNew: Chờ gửi<br><br>  - New: Chờ khớp<br><br>  - PendingReplace: Chờ sửa<br><br>  - PendingCancel: Chờ hủy<br><br>  - PartiallyFilled: Khớp một phần<br><br>  - Filled: Khớp toàn bộ<br><br>  - Canceled: Đã hủy<br><br>  - Rejected: Bị từ chối<br><br>  - Expired: Hết hạn trong phiên<br><br>  - DoneForDay: Lệnh được giải tỏa do không khớp trong phiên<br><br>Trạng thái lệnh STOP/OCO:<br><br>  - New: Chờ kích hoạt                              <br>  - Activated: Đã kích hoạt<br>  - Cancelled: Đã hủy<br>  - Expired: Hết hiệu lực<br>  - Rejected: Bị từ chối<br>  - Failed: Lệnh thất bại|
|» price|integer(int32)|false|none|Giá đặt|
|» quantity|integer(int32)|false|none|Khối lượng đặt|
|» loanPackageId|integer(int32)|false|none|Mã gói vay|
|» priceSecure|integer(int32)|false|none|Giá bảo đảm|
|» averagePrice|integer(int32)|false|none|Giá khớp trung bình|
|» fillQuantity|integer(int32)|false|none|Khối lượng khớp|
|» canceledQuantity|integer(int32)|false|none|Khối lượng đã hủy|
|» leaveQuantity|integer(int32)|false|none|Khối lượng còn lại|
|» transDate|string|false|none|Ngày giao dịch|
|» createdDate|string(date-time)|false|none|Thời điểm tạo|
|» modifiedDate|string(date-time)|false|none|Thời điểm cập nhật|

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
