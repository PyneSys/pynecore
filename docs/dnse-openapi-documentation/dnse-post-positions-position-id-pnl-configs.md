## Cài đặt chốt lời, cắt lỗ cho vị thế

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="postPositionsPositionIdPnlConfigs"></span>

### `POST /positions/{positionId}/pnl-configs`

Cài đặt cấu hình chốt lời, cắt lỗ cho vị thế đang nắm giữ theo `positionId.`

Khuyến nghị: Body Request gửi lên đầy đủ các trường thông tin.

<h3 id="postpositionspositionidpnlconfigs-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|string|false|Loại thị trường |
|X-API-Key|header|string|false|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|false|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|false|Chữ ký xác thực yêu cầu|
|version|header|string|false|Phiên bản API |
|trading-token|header|string|false|Token đặt lệnh|
|body|body|object|false|none|
|» takeProfit|body|object|false|Cấu hình chốt lời (Take Profit)|
|»» enabled|body|boolean|false|Bật/tắt chức năng chốt lời|
|»» strategy|body|string|false|Chiến lược kích hoạt chốt lời |
|»» rate|body|number(double)|false|Tỷ lệ % so với giá hòa vốn, dùng để kích hoạt chốt lời khi strategy = PNL_RATE. Giá trị phải lớn hơn 0|
|»» deltaPrice|body|number(double)|false|Mức chênh lệch giá so với giá hòa vốn, dùng để kích hoạt chốt lời khi strategy = DELTA_PRICE. Giá trị phải lớn hơn 0|
|»» orderMethod|body|string|false|Phương thức đặt lệnh khi điều kiện chốt lời được kích hoạt.|
|»» orderDeltaPrice|body|integer(double)|false|Biên độ giá đặt lệnh so với giá kích hoạt khi orderMethod = DELTA_PRICE. Có thể là số âm, số dương hoặc bằng 0|
|» stopLoss|body|object|false|Cấu hình cắt lỗ (Stop Loss)|
|»» enabled|body|boolean|false|Bật/tắt chức năng cắt lỗ|
|»» strategy|body|string|false|Chiến lược kích hoạt cắt lỗ |
|»» rate|body|number(double)|false|Tỷ lệ % so với giá hòa vốn, dùng để kích hoạt cắt lỗ khi strategy = PNL_RATE. Giá trị hợp lệ trong khoảng [-1.0, 0)|
|»» deltaPrice|body|number(double)|false|Mức chênh lệch giá so với giá hòa vốn, dùng để kích hoạt cắt lỗ khi strategy = DELTA_PRICE. Giá trị phải lớn hơn 0|
|»» orderMethod|body|string|false|Phương thức đặt lệnh khi điều kiện chốt lời được kích hoạt.|
|»» orderDeltaPrice|body|number(double)|false|Biên độ giá đặt lệnh so với giá kích hoạt khi orderMethod = DELTA_PRICE. Có thể là số âm, số dương hoặc bằng 0|
|»» trailingEnabled|body|boolean|false|Bật/tắt cơ chế Trailing Stop. Lưu ý cơ chế này chỉ hoạt động khi chức năng cắt lỗ được bật|
|positionId|path|integer|true|Id vị thế |

#### Detailed descriptions

**marketType**: Loại thị trường 
- DERIVATIVE: Deal phái sinh (chỉ hỗ trợ phái sinh)

**»» strategy**: Chiến lược kích hoạt chốt lời 
  - PNL_RATE: Kích hoạt theo tỷ lệ %
  - DELTA_PRICE: Kích hoạt theo mức chênh lệch giá

**»» orderMethod**: Phương thức đặt lệnh khi điều kiện chốt lời được kích hoạt.
  - FASTEST: Lệnh khớp ngay với giá đặt là giá trần/sàn tùy theo chiều vị thế Bán/Mua
  - DELTA_PRICE: Lệnh đặt theo biên độ giá so với giá kích hoạt

**»» strategy**: Chiến lược kích hoạt cắt lỗ 
  - PNL_RATE: Kích hoạt theo tỷ lệ %
  - DELTA_PRICE: Kích hoạt theo mức chênh lệch giá

**»» orderMethod**: Phương thức đặt lệnh khi điều kiện chốt lời được kích hoạt.
  - FASTEST: Lệnh khớp ngay với giá đặt là giá trần/sàn tùy theo chiều vị thế Bán/Mua
  - DELTA_PRICE: Lệnh đặt theo biên độ giá so với giá kích hoạt

> Code samples

```shell
# You can also use wget
curl -X POST https://openapi.dnse.com.vn/positions/{positionId}/pnl-configs \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23' \
  -H 'trading-token: 7ceef658-9f01-414e-8b3e-faa77bb9061e'

```

```http
POST https://openapi.dnse.com.vn/positions/{positionId}/pnl-configs HTTP/1.1
Host: openapi.dnse.com.vn
Content-Type: application/json
Accept: application/json
X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==
X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000
X-Signature: your_signature
version: 2026-07-23
trading-token: 7ceef658-9f01-414e-8b3e-faa77bb9061e

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Content-Type": []string{"application/json"},
        "Accept": []string{"application/json"},
        "X-API-Key": []string{"eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ=="},
        "X-Aux-Date": []string{"Mon, 19 Jan 2026 07:45:23 +0000"},
        "X-Signature": []string{"your_signature"},
        "version": []string{"2026-07-23"},
        "trading-token": []string{"7ceef658-9f01-414e-8b3e-faa77bb9061e"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("POST", "https://openapi.dnse.com.vn/positions/{positionId}/pnl-configs", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

```javascript
const inputBody = '{
  "takeProfit": {
    "enabled": true,
    "strategy": "DELTA_PRICE",
    "rate": 0.52,
    "deltaPrice": 162.8,
    "orderMethod": "FASTEST",
    "orderDeltaPrice": 2
  },
  "stopLoss": {
    "enabled": null,
    "strategy": "DELTA_PRICE",
    "rate": -0.34,
    "deltaPrice": 50.3,
    "orderMethod": "FASTEST",
    "orderDeltaPrice": 10.5,
    "trailingEnabled": true
  }
}';
const headers = {
  'Content-Type':'application/json',
  'Accept':'application/json',
  'X-API-Key':'eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==',
  'X-Aux-Date':'Mon, 19 Jan 2026 07:45:23 +0000',
  'X-Signature':'your_signature',
  'version':'2026-07-23',
  'trading-token':'7ceef658-9f01-414e-8b3e-faa77bb9061e'
};

fetch('https://openapi.dnse.com.vn/positions/{positionId}/pnl-configs',
{
  method: 'POST',
  body: inputBody,
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
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'X-API-Key': 'eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==',
  'X-Aux-Date': 'Mon, 19 Jan 2026 07:45:23 +0000',
  'X-Signature': 'your_signature',
  'version': '2026-07-23',
  'trading-token': '7ceef658-9f01-414e-8b3e-faa77bb9061e'
}

r = requests.post('https://openapi.dnse.com.vn/positions/{positionId}/pnl-configs', headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/positions/{positionId}/pnl-configs");
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

> Body parameter

```json
{
  "takeProfit": {
    "enabled": true,
    "strategy": "DELTA_PRICE",
    "rate": 0.52,
    "deltaPrice": 162.8,
    "orderMethod": "FASTEST",
    "orderDeltaPrice": 2
  },
  "stopLoss": {
    "enabled": null,
    "strategy": "DELTA_PRICE",
    "rate": -0.34,
    "deltaPrice": 50.3,
    "orderMethod": "FASTEST",
    "orderDeltaPrice": 10.5,
    "trailingEnabled": true
  }
}
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

<h3 id="postpositionspositionidpnlconfigs-responseschema">Response Schema</h3>

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
