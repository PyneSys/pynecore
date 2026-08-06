## Lịch sử OHLC

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getOhlcHistory"></span>

### `GET /price/ohlc`

Truy vấn thông tin lịch sử nến (open, high, low, close, volume) cho Cổ phiếu, Phái sinh và Chỉ số thị trường theo khung thời gian và khoảng thời gian cụ thể.

<h3 id="getohlchistory-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|symbol|query|string|true|Mã chứng khoán |
|type|query|string|false|Loại thị trường|
|resolution|query|string|true|Khung thời gian nến|
|from|query|integer|true|Thời gian bắt đầu|
|to|query|integer|true|Thời gian kết thúc|
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|version|header|string|false|API version (YYYY-MM-DD)|

#### Detailed descriptions

**type**: Loại thị trường
- STOCK: Cổ phiếu
- DERIVATIVE: Phái sinh
- INDEX: Chỉ số thị trường

**resolution**: Khung thời gian nến
- 1,3,5,15,30,1h,1D,1W

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/price/ohlc?symbol=ACB&resolution=15&from=1773657310&to=1773830110 \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/price/ohlc?symbol=ACB&resolution=15&from=1773657310&to=1773830110 HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/price/ohlc", data)
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

fetch('https://openapi.dnse.com.vn/price/ohlc?symbol=ACB&resolution=15&from=1773657310&to=1773830110',
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

r = requests.get('https://openapi.dnse.com.vn/price/ohlc', params={
  'symbol': 'ACB',  'resolution': '15',  'from': '1773657310',  'to': '1773830110'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/price/ohlc?symbol=ACB&resolution=15&from=1773657310&to=1773830110");
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
  "t": [
    1773715500,
    1773716400,
    1773800100
  ],
  "o": [
    23.8,
    23.75,
    23.75
  ],
  "h": [
    23.8,
    23.8,
    23.9
  ],
  "l": [
    23.7,
    23.7,
    23.7
  ],
  "c": [
    23.75,
    23.75,
    23.8
  ],
  "v": [
    2530900,
    514100,
    494000
  ],
  "nextTime": 0
}
```

<h3 id="getohlchistory-responseschema">Response Schema</h3>

Status Code **200**

*Dữ liệu biểu đồ nến (OHLCV) theo từng khoảng thời gian.*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» t|[integer]|true|none|Danh sách thời gian của từng cây nến, định dạng Unix Timestamp (giây).|
|» o|[number]|true|none|Danh sách giá mở cửa (Open) tương ứng với từng cây nến.|
|» h|[number]|true|none|Danh sách giá cao nhất (High) tương ứng với từng cây nến.|
|» l|[number]|true|none|Danh sách giá thấp nhất (Low) tương ứng với từng cây nến.|
|» c|[number]|true|none|Danh sách giá đóng cửa (Close) tương ứng với từng cây nến.|
|» v|[integer]|true|none|Danh sách khối lượng giao dịch (Volume) tương ứng với từng cây nến.|
|» nextTime|integer(int64)|true|none|Mốc thời gian (Unix Timestamp, giây) của cây nến tiếp theo khi dữ liệu hiện tại chưa đủ để đáp ứng yêu cầu. Giá trị 0 nếu không còn dữ liệu.|

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
