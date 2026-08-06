## Chi tiết lệnh theo ID

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getOrderDetail"></span>

### `GET /accounts/{accountNo}/orders/{orderId}`

Lấy thông tin chi tiết của một lệnh thường (NORMAL) theo `orderId`, bao gồm trạng thái, khối lượng, giá và các thông tin liên quan.

<h3 id="getorderdetail-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|string|true|Loại thị trường |
|orderCategory|query|string|true|Phân loại lệnh thường, lệnh điều kiện (mặc định NORMAL)|
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|version|header|string(date)|true|API version (YYYY-MM-DD)|
|accountNo|path|string|true|Số tiểu khoản|
|orderId|path|integer|true|Mã lệnh giao dịch|

#### Detailed descriptions

**marketType**: Loại thị trường 
- STOCK: Giao dịch cơ sở
- DERIVATIVE: Giao dịch phái sinh
- BOND: Giao dịch trái phiếu

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}?marketType=BOND&orderCategory=NORMAL \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}?marketType=BOND&orderCategory=NORMAL HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}", data)
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

fetch('https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}?marketType=BOND&orderCategory=NORMAL',
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

r = requests.get('https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}', params={
  'marketType': 'BOND',  'orderCategory': 'NORMAL'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}?marketType=BOND&orderCategory=NORMAL");
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

> 200 Response

```json
{
  "id": 966,
  "side": "NS",
  "accountNo": "0001179019",
  "symbol": "41I1G4000",
  "price": 1706.1,
  "quantity": 5,
  "orderType": "LO",
  "loanPackageId": 2278,
  "orderCategory": "NORMAL",
  "orderStatus": "Filled",
  "fillQuantity": 5,
  "lastQuantity": 2,
  "lastPrice": 1706.1,
  "averagePrice": 1750.96,
  "transDate": "2026-03-16",
  "taxRate": 0,
  "exchangeFeeRate": 0,
  "feeRate": 0,
  "leaveQuantity": 0,
  "canceledQuantity": 0,
  "error": "",
  "marketType": "DERIVATIVE",
  "priceSecure": 1706.1,
  "createdDate": "2026-03-23T03:10:06.556826794Z",
  "modifiedDate": "2026-03-23T04:07:45.683977124Z",
  "metadata": "{\"orderSession\":\"OPEN\",\"dealId\":\"1.77410795472387E14\",\"releaseSecureOnFilled\":\"false\",\"originMaker\":\"1000005917-close_deal_177410795472387\",\"dtaAccountNo\":\"D000113702\",\"isForeigner\":false,\"probType\":\"CUSTOMER\",\"eventNo\":5}"
}
```

<h3 id="getorderdetail-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» id|integer(int32)|false|none|ID lệnh|
|» side|string|false|none|Chiều giao dịch<br>- NB: Mua<br>- NS: Bán|
|» accountNo|string|false|none|Số tiểu khoản|
|» symbol|string|false|none|Mã hợp đồng phái sinh|
|» price|number(double)|false|none|Giá đặt lệnh|
|» quantity|integer(int32)|false|none|Khối lượng đặt lệnh|
|» orderType|string|false|none|Loại lệnh<br>- LO: Lệnh giới hạn<br>- MOK/MAK/MTL: Lệnh thị trường<br>- ATO/ATC: Lệnh phiên định kỳ mở cửa/đóng cửa<br>- PLO: Lệnh khớp lệnh sau giờ|
|» loanPackageId|integer(int32)|false|none|Mã gói vay|
|» orderCategory|string|false|none|Phân loại lệnh (mặc định NORMAL)|
|» orderStatus|string|false|none|Trạng thái lệnh<br>- Pending/PendingNew: Chờ gửi<br>- New: Chờ khớp<br>- PartiallyFilled: Khớp một phần<br>- Filled: Khớp toàn bộ<br>- Rejected: Bị từ chối<br>- Expired: Hết hạn trong phiên<br>- DoneForDay: Lệnh được giải tỏa do không khớp trong phiên|
|» fillQuantity|integer(int32)|false|none|Khối lượng đã khớp|
|» lastQuantity|integer(int32)|false|none|Khối lượng khớp gần nhất|
|» lastPrice|number(double)|false|none|Giá khớp gần nhất|
|» averagePrice|number(double)|false|none|Giá khớp trung bình|
|» transDate|string|false|none|Ngày giao dịch|
|» taxRate|integer(int32)|false|none|Tỷ lệ thuế|
|» exchangeFeeRate|integer(int32)|false|none|Tỷ lệ phí trả Sở giao dịch|
|» feeRate|integer(int32)|false|none|Tổng tỷ lệ phí của lệnh|
|» leaveQuantity|integer(int32)|false|none|Khối lượng còn lại chưa khớp|
|» canceledQuantity|integer(int32)|false|none|Khối lượng đã hủy|
|» error|string|false|none|Mã lỗi nếu lệnh bị từ chối|
|» marketType|string|false|none|Loại thị trường<br>- STOCK: Sổ lệnh cơ sở<br>- DERIVATIVE: Sổ lệnh phái sinh|
|» priceSecure|number(double)|false|none|Giá dùng để kiểm tra sức mua/đặt lệnh|
|» createdDate|string(date-time)|false|none|Thời điểm tạo lệnh|
|» modifiedDate|string(date-time)|false|none|Thời điểm cập nhật lệnh|
|» metadata|string|false|none|Thông tin bổ sung của lệnh|

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
