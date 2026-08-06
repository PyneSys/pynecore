## Giá dự khớp

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="get-expected-price"></span>

### `GET /price/{symbol}/expected-price`

<h3 id="get-expected-price-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|boardId|query|string|false|Mã bảng giao dịch|
|from|query|string|true|Thời gian bắt đầu (timestamp)|
|to|query|string|true|Thời gian kết thúc (timestamp) (không vượt quá 1 ngày)|
|limit|query|integer|false|none|
|nextPageToken|query|string|false|none|
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|version|header|string(date)|false|API version (YYYY-MM-DD)|
|symbol|path|string|true|Mã chứng khoán|

#### Detailed descriptions

**boardId**: Mã bảng giao dịch
- G1: Lô chẵn
- G4: Lô lẻ
- T1: Thỏa thuận trong giờ (9h - 14h45)
- T3: Thỏa thuận sau giờ (14h45 - 15h)
- T4: Thỏa thuận lô lẻ trong giờ (9h - 14h45)
- T6: Thỏa thuận lô lẻ sau giờ  (14h45 - 15h)

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/price/{symbol}/expected-price?from=string&to=string \
  -H 'Accept: application/json' \
  -H 'X-API-Key: string' \
  -H 'X-Aux-Date: string' \
  -H 'X-Signature: string' \
  -H 'version: 2019-08-24'

```

```http
GET https://openapi.dnse.com.vn/price/{symbol}/expected-price?from=string&to=string HTTP/1.1
Host: openapi.dnse.com.vn
Accept: application/json
X-API-Key: string
X-Aux-Date: string
X-Signature: string
version: 2019-08-24

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
        "X-API-Key": []string{"string"},
        "X-Aux-Date": []string{"string"},
        "X-Signature": []string{"string"},
        "version": []string{"2019-08-24"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/price/{symbol}/expected-price", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

```javascript

const headers = {
  'Accept':'application/json',
  'X-API-Key':'string',
  'X-Aux-Date':'string',
  'X-Signature':'string',
  'version':'2019-08-24'
};

fetch('https://openapi.dnse.com.vn/price/{symbol}/expected-price?from=string&to=string',
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
  'X-API-Key': 'string',
  'X-Aux-Date': 'string',
  'X-Signature': 'string',
  'version': '2019-08-24'
}

r = requests.get('https://openapi.dnse.com.vn/price/{symbol}/expected-price', params={
  'from': 'string',  'to': 'string'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/price/{symbol}/expected-price?from=string&to=string");
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
  "expectedPrices": [
    {
      "marketId": "STO",
      "boardId": "G1",
      "isin": "VN000000HPG4",
      "symbol": "HPG",
      "closePrice": 0,
      "expectedTradePrice": 94.1,
      "expectedTradeQuantity": 2000,
      "time": "2026-08-03 14:40:05.148"
    }
  ],
  "nextPageToken": null
}
```

<h3 id="get-expected-price-responseschema">Response Schema</h3>

Status Code **200**

*Danh sách dữ liệu giá khớp dự kiến cùng thông tin phân trang.*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» expectedPrices|[object]|false|none|Danh sách dữ liệu giá khớp dự kiến của mã chứng khoán.|
|»» marketId|string|false|none|Mã thị trường niêm yết mã chứng khoán<br>- DVX: Phái sinh sàn HNX<br>- HCX: Trái phiếu doanh nghiệp HNX<br>- STO: Cổ phiếu sàn HOSE<br>- STX: Cổ phiếu sàn HNX<br>- UPX: Cổ phiếu sàn Upcom|
|»» boardId|string|false|none|Mã bảng giao dịch<br>- G1: Lô chẵn<br>- G4: Lô lẻ<br>- T1: Thỏa thuận trong giờ (9h - 14h45)<br>- T3: Thỏa thuận sau giờ (14h45 - 15h)<br>- T4: Thỏa thuận lô lẻ trong giờ (9h - 14h45)<br>- T6: Thỏa thuận lô lẻ sau giờ (14h45 - 15h)|
|»» isin|string|false|none|Mã định danh quốc tế (ISIN) của chứng khoán.|
|»» symbol|string|false|none|Mã chứng khoán|
|»» closePrice|number(double)|false|none|Giá đóng cửa|
|»» expectedTradePrice|number(double)|false|none|Giá khớp dự kiến|
|»» expectedTradeQuantity|integer(int32)|false|none|Khối lượng khớp dự kiến|
|»» time|string|false|none|Thời gian ghi nhận. Định dạng: YYYY-MM-DD HH:mm:ss.SSS (GMT+7).|
|» nextPageToken|string¦null|false|none|Token dùng để lấy trang dữ liệu tiếp theo. Giá trị null nếu không còn dữ liệu.|

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
