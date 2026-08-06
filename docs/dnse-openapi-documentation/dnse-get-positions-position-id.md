## Chi tiết vị thế theo ID

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getPositionsPositionId"></span>

### `GET /positions/{positionId}`

Lấy thông tin chi tiết của một vị thế cơ sở hoặc phái sinh đang mở theo `positionId.`

<h3 id="getpositionspositionid-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|string|true|Loại thị trường |
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|version|header|string(date)|false|API version (YYYY-MM-DD)|
|positionId|path|integer|true|Id vị thế|

#### Detailed descriptions

**marketType**: Loại thị trường 
- STOCK: Vị thế cơ sở
- DERIVATIVE: Vị thế phái sinh

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/positions/{positionId}?marketType=DERIVATIVE \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/positions/{positionId}?marketType=DERIVATIVE HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/positions/{positionId}", data)
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

fetch('https://openapi.dnse.com.vn/positions/{positionId}?marketType=DERIVATIVE',
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

r = requests.get('https://openapi.dnse.com.vn/positions/{positionId}', params={
  'marketType': 'DERIVATIVE'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/positions/{positionId}?marketType=DERIVATIVE");
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
  "id": 9589,
  "symbol": "BVS",
  "accountNo": "0001179019",
  "status": "OPEN",
  "loanPackageId": 5757,
  "side": "NB",
  "accumulateQuantity": 3400,
  "tradeQuantity": 3400,
  "closedQuantity": 0,
  "openQuantity": 3400,
  "costPrice": 28000,
  "averageCostPrice": 28000,
  "averageClosePrice": 0,
  "marketPrice": 25200,
  "breakEvenPrice": 28043.4752,
  "createdDate": "2026-07-20T04:22:38.904283Z",
  "modifiedDate": "2026-07-30T11:20:44.65933Z"
}
```

```json
{
  "id": 178488852178882,
  "symbol": "41I1G9000",
  "marketType": "DERIVATIVE",
  "accountNo": "0001179019",
  "status": "OPEN",
  "loanPackageId": 2279,
  "side": "NS",
  "accumulateQuantity": 12,
  "tradeQuantity": 10,
  "closedQuantity": 2,
  "openQuantity": 10,
  "overNightQuantity": 10,
  "costPrice": 1855.4,
  "averageCostPrice": 1855.4,
  "averageClosePrice": 1931,
  "marketPrice": 1931.5,
  "breakEvenPrice": 1854.93962,
  "createdDate": "2026-07-30T07:27:30.375891Z",
  "modifiedDate": "2026-07-30T15:41:24.838062Z"
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

<h3 id="getpositionspositionid-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» id|integer(int32)|false|none|ID vị thế|
|» symbol|string|false|none|Mã chứng khoán|
|» accountNo|string|false|none|Số tiểu khoản|
|» status|string|false|none|Trạng thái của vị thế<br>- OPEN: Đang mở<br>- PENDING_CLOSE: Chờ đóng<br>- CLOSED: Đã đóng<br>- ODD_LOT: Lô lẻ (cơ sở)|
|» loanPackageId|integer(int32)|false|none|Gói vay cơ sở hoặc phái sinh|
|» side|string|false|none|Loại vị thế<br>- NB: Mua<br>- NS: Bán|
|» accumulateQuantity|integer(int32)|false|none|Khối lượng cộng dồn|
|» tradeQuantity|integer(int32)|false|none|Khối lượng được giao dịch|
|» closedQuantity|integer(int32)|false|none|Khối lượng đã đóng|
|» openQuantity|integer(int32)|false|none|Khối lượng mở|
|» costPrice|integer(int32)|false|none|Giá vốn trung bình của khối lượng mở|
|» averageCostPrice|integer(int32)|false|none|Giá vốn trung bình|
|» averageClosePrice|integer(int32)|false|none|Giá đóng trung bình|
|» marketPrice|integer(int32)|false|none|Giá thị trường|
|» breakEvenPrice|number(double)|false|none|Giá hòa vốn|
|» createdDate|string(date-time)|false|none|Thời điểm mở|
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
