## Cấu hình chốt lời, cắt lỗ của vị thế

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getPositionsPositionIdPnlConfigs"></span>

### `GET /positions/{positionId}/pnl-configs`

Lấy cấu hình chốt lời, cắt lỗ của vị thế đang nắm giữ theo `positionId.`

<h3 id="getpositionspositionidpnlconfigs-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|string|true|Loại thị trường |
|X-API-Key|header|string|false|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|false|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|false|Chữ ký xác thực yêu cầu|
|version|header|string|false|Phiên bản API |
|positionId|path|integer|true|Id vị thế |

#### Detailed descriptions

**marketType**: Loại thị trường 
- DERIVATIVE: Deal phái sinh
Hiện tại chỉ hỗ trợ phái sinh

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/positions/{positionId}/pnl-configs?marketType=DERIVATIVE \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-05-07'

```

```http
GET https://openapi.dnse.com.vn/positions/{positionId}/pnl-configs?marketType=DERIVATIVE HTTP/1.1
Host: openapi.dnse.com.vn
Accept: application/json
X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==
X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000
X-Signature: your_signature
version: 2026-05-07

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
        "version": []string{"2026-05-07"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/positions/{positionId}/pnl-configs", data)
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
  'version':'2026-05-07'
};

fetch('https://openapi.dnse.com.vn/positions/{positionId}/pnl-configs?marketType=DERIVATIVE',
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
  'version': '2026-05-07'
}

r = requests.get('https://openapi.dnse.com.vn/positions/{positionId}/pnl-configs', params={
  'marketType': 'DERIVATIVE'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/positions/{positionId}/pnl-configs?marketType=DERIVATIVE");
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
  "accountNo": "0001179019",
  "positionId": 177952229647243,
  "configs": {
    "takeProfit": {
      "enabled": true,
      "strategy": "DELTA_PRICE",
      "rate": 0.52,
      "deltaPrice": 162.8,
      "orderMethod": "FASTEST",
      "orderDeltaPrice": 2
    },
    "stopLoss": {
      "enabled": true,
      "strategy": "PNL_RATE",
      "rate": -0.34,
      "deltaPrice": 50.3,
      "orderMethod": "DELTA_PRICE",
      "orderDeltaPrice": 10.5,
      "trailingEnabled": true
    }
  },
  "createdDate": "2026-06-02T02:47:36.384761Z",
  "modifiedDate": "2026-06-03T07:52:08.00923505Z"
}
```

<h3 id="getpositionspositionidpnlconfigs-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» accountNo|string|false|none|Số tiểu khoản chứng khoán|
|» positionId|integer(int64)|false|none|ID vị thế|
|» configs|object|false|none|Cấu hình chốt lời và cắt lỗ của vị thế|
|»» takeProfit|object|false|none|Cấu hình chốt lời (Take Profit)|
|»»» enabled|boolean|false|none|Bật/tắt chức năng chốt lời|
|»»» strategy|string|false|none|Chiến lược kích hoạt chốt lời<br>- PNL_RATE: Kích hoạt theo tỷ lệ %<br>- DELTA_PRICE: Kích hoạt theo mức chênh lệch giá|
|»»» rate|number(double)|false|none|Tỷ lệ % so với giá hòa vốn, dùng để kích hoạt chốt lời khi strategy = PNL_RATE. Giá trị phải lớn hơn 0|
|»»» deltaPrice|number(double)|false|none|Mức chênh lệch giá so với giá hòa vốn, dùng để kích hoạt chốt lời khi strategy = DELTA_PRICE. Giá trị phải lớn hơn 0|
|»»» orderMethod|string|false|none|Phương thức đặt lệnh khi điều kiện chốt lời được kích hoạt.<br>- FASTEST: Lệnh khớp ngay với giá đặt là giá trần/sàn tùy theo chiều vị thế Bán/Mua<br>- DELTA_PRICE: Lệnh đặt theo biên độ giá so với giá kích hoạt|
|»»» orderDeltaPrice|integer(int32)|false|none|Biên độ giá đặt lệnh khi orderMethod = DELTA_PRICE. Bằng 0 khi orderMethod = FASTEST|
|»» stopLoss|object|false|none|Cấu hình cắt lỗ (Stop Loss)|
|»»» enabled|boolean|false|none|Bật/tắt chức năng cắt lỗ|
|»»» strategy|string|false|none|Chiến lược kích hoạt cắt lỗ<br>- PNL_RATE: Kích hoạt theo tỷ lệ %<br>- DELTA_PRICE: Kích hoạt theo mức chênh lệch giá.|
|»»» rate|number(double)|false|none|Tỷ lệ % so với giá hòa vốn, dùng để kích hoạt chốt lời khi strategy = PNL_RATE. Giá trị hợp lệ trong khoảng [-1.0, 0)|
|»»» deltaPrice|number(double)|false|none|Mức chênh lệch giá so với giá hòa vốn, dùng để kích hoạt chốt lời khi strategy = DELTA_PRICE. Giá trị phải lớn hơn 0|
|»»» orderMethod|string|false|none|Phương thức đặt lệnh khi điều kiện chốt lời được kích hoạt.<br>- FASTEST: Lệnh khớp ngay với giá đặt là giá trần/sàn tùy theo chiều vị thế Bán/Mua<br>- DELTA_PRICE: Lệnh đặt theo biên độ giá so với giá kích hoạt|
|»»» orderDeltaPrice|number(float)|false|none|Biên độ giá đặt lệnh khi orderMethod = DELTA_PRICE. Bằng 0 khi orderMethod = FASTEST|
|»»» trailingEnabled|boolean|false|none|Bật/tắt cơ chế Trailing Stop. Lưu ý cơ chế này chỉ hoạt động khi chức năng cắt lỗ được bật|
|» createdDate|string(date-time)|false|none|Thời điểm cấu hình được tạo|
|» modifiedDate|string(date-time)|false|none|Thời điểm cập nhật cấu hình gần nhất|

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

<h1 id="openapi-v2-spec-260730-trading">trading</h1>
