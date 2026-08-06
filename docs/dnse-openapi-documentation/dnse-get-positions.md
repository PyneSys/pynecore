## Vị thế nắm giữ

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getPositions"></span>

### `GET /accounts/{accountNo}/positions`

Lấy danh sách các vị thế đang nắm giữ trên tài khoản, bao gồm thông tin mã chứng khoán, khối lượng, giá vốn, và các thông tin khác.

<h3 id="getpositions-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|string|true|Loại thị trường |
|pageSize|query|string|false|Kích thước trang dữ liệu (page size)|
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|version|header|string|false|API version (YYYY-MM-DD)|
|accountNo|path|string|true|Số tiểu khoản|

#### Detailed descriptions

**marketType**: Loại thị trường 
- STOCK: Danh sách vị thế cơ sở
- DERIVATIVE: Danh sách vị thế phái sinh

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/accounts/{accountNo}/positions?marketType=STOCK \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/accounts/{accountNo}/positions?marketType=STOCK HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/accounts/{accountNo}/positions", data)
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

fetch('https://openapi.dnse.com.vn/accounts/{accountNo}/positions?marketType=STOCK',
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

r = requests.get('https://openapi.dnse.com.vn/accounts/{accountNo}/positions', params={
  'marketType': 'STOCK'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/accounts/{accountNo}/positions?marketType=STOCK");
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
  "positions": [
    {
      "id": 177410795472387,
      "symbol": "41I1G4000",
      "accountNo": "0001179019",
      "status": "OPEN",
      "loanPackageId": 2278,
      "side": "NB",
      "accumulateQuantity": 6,
      "tradeQuantity": 1,
      "closedQuantity": 5,
      "openQuantity": 1,
      "overNightQuantity": 0,
      "costPrice": 1834.5,
      "marketPrice": 1706.1,
      "breakEvenPrice": 1834.95691,
      "createdDate": "2026-03-23T03:08:32.773651Z",
      "modifiedDate": "2026-03-23T04:07:45.692156Z"
    }
  ],
  "pageIndex": 0,
  "pageSize": 20,
  "pageNumber": 1,
  "total": 1
}
```

<h3 id="getpositions-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» positions|[object]|false|none|none|
|»» id|integer(int64)|false|none|ID vị thế|
|»» symbol|string|false|none|Mã chứng khoán|
|»» marketType|string|false|none|Loại thị trường<br>- STOCK: Gói vay giao dịch cơ sở<br>- DERIVATIVE: Gói vay giao dịch phái sinh|
|»» accountNo|string|false|none|Số tiểu khoản|
|»» status|string|false|none|Trạng thái của vị thế<br>- OPEN: Đang mở<br>- PENDING_CLOSE: Chờ đóng<br>- CLOSED: Đã đóng<br>- ODD_LOT: Lô lẻ (cơ sở)|
|»» loanPackageId|integer(int32)|false|none|Gói vay cơ sở hoặc phái sinh|
|»» side|string|false|none|Loại vị thế<br>- NB: Mua<br>- NS: Bán|
|»» accumulateQuantity|integer(int32)|false|none|Khối lượng cộng dồn|
|»» tradeQuantity|integer(int32)|false|none|Khối lượng được giao dịch|
|»» closedQuantity|integer(int32)|false|none|Khối lượng đã đóng|
|»» openQuantity|integer(int32)|false|none|Khối lượng mở|
|»» overNightQuantity|integer(int32)|false|none|Khối lượng mở qua đêm (dành cho phái sinh)|
|»» costPrice|number(float)|false|none|Giá vốn trung bình của khối lượng mở|
|»» marketPrice|number(double)|false|none|Giá thị trường|
|»» breakEvenPrice|number(double)|false|none|Giá hòa vốn|
|»» createdDate|string(date-time)|false|none|Thời điểm mở|
|»» modifiedDate|string(date-time)|false|none|Thời điểm cập nhật|
|» pageIndex|integer(int32)|false|none|Kích thước trang dữ liệu (dành cho phái sinh)|
|» pageSize|integer(int32)|false|none|Số bản ghi trên 1 trang (dành cho phái sinh)|
|» pageNumber|integer(int32)|false|none|Số trang (chỉ dành cho phái sinh)|
|» total|integer(int32)|false|none|Tổng số vị thế (chỉ dành cho phái sinh)|

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
