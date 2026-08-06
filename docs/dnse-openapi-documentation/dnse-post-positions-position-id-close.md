## Đóng vị thế

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="postPositionsPositionIdClose"></span>

### `POST /positions/{positionId}/close`

<h3 id="postpositionspositionidclose-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|string|true|Loại thị trường |
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|trading-token|header|string|true|Token đặt lệnh|
|version|header|string(date)|false|API version (YYYY-MM-DD)|
|positionId|path|string|true|Id vị thế|

#### Detailed descriptions

**marketType**: Loại thị trường 
- DERIVATIVE: Deal phái sinh (chỉ hỗ trợ phái sinh)

> Code samples

```shell
# You can also use wget
curl -X POST https://openapi.dnse.com.vn/positions/{positionId}/close?marketType=DERIVATIVE \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Signature: your_signature' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'trading-token: 7ceef658-9f01-414e-8b3e-faa77bb9061e' \
  -H 'version: 2026-07-23'

```

```http
POST https://openapi.dnse.com.vn/positions/{positionId}/close?marketType=DERIVATIVE HTTP/1.1
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
    req, err := http.NewRequest("POST", "https://openapi.dnse.com.vn/positions/{positionId}/close", data)
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

fetch('https://openapi.dnse.com.vn/positions/{positionId}/close?marketType=DERIVATIVE',
{
  method: 'POST',

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

r = requests.post('https://openapi.dnse.com.vn/positions/{positionId}/close', params={
  'marketType': 'DERIVATIVE'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/positions/{positionId}/close?marketType=DERIVATIVE");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("POST");
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
  "id": 1636,
  "side": "NS",
  "accountNo": "0001179019",
  "symbol": "41I1G4000",
  "price": 1618.2,
  "quantity": 1,
  "orderType": "LO",
  "fillQuantity": 0,
  "leaveQuantity": 0,
  "canceledQuantity": 0,
  "loanPackageId": 2278,
  "createdDate": "2026-03-24T04:20:55.63",
  "modifiedDate": "2026-03-24T04:20:55.63"
}
```

<h3 id="postpositionspositionidclose-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» id|integer(int32)|false|none|Id lệnh giao dịch|
|» side|string|false|none|Chiều đặt lệnh<br>- NB: Mua<br>- NS: Bán|
|» accountNo|string|false|none|Số tiểu khoản|
|» symbol|string|false|none|Mã chứng khoán|
|» price|number(double)|false|none|Giá đặt|
|» quantity|integer(int32)|false|none|Khối lượng đặt|
|» orderType|string|false|none|Loại lệnh<br>- LO: Lệnh giới hạn<br>- MOK/MAK/MTL: Lệnh thị trường<br>- ATO/ATC: Lệnh phiên định kỳ mở cửa/đóng cửa<br>- PLO: Lệnh khớp lệnh sau giờ|
|» fillQuantity|integer(int32)|false|none|Khối lượng khớp|
|» leaveQuantity|integer(int32)|false|none|Khối lượng còn lại|
|» canceledQuantity|integer(int32)|false|none|Khối lượng đã hủy|
|» loanPackageId|integer(int32)|false|none|Mã gói vay|
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

<h1 id="openapi-v2-spec-260730-market-data">market-data</h1>
