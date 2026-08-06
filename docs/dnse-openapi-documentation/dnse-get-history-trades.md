## Lịch sử khớp lệnh

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getHistoryTrades"></span>

### `GET /price/{symbol}/trades`

Truy vấn thông tin lịch sử khớp lệnh của mã chứng khoán theo bảng giao dịch và khoảng thời gian cụ thể.

<h3 id="gethistorytrades-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|boardId|query|string|false|Mã bảng giao dịch|
|from|query|integer|true|Thời gian bắt đầu (timestamp)|
|to|query|integer|true|Thời gian kết thúc (timestamp) (không vượt quá 1 ngày)|
|limit|query|integer|false|none|
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|version|header|string|false|API version (YYYY-MM-DD)|
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
curl -X GET https://openapi.dnse.com.vn/price/{symbol}/trades?from=1785727927&to=1785814327 \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/price/{symbol}/trades?from=1785727927&to=1785814327 HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/price/{symbol}/trades", data)
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

fetch('https://openapi.dnse.com.vn/price/{symbol}/trades?from=1785727927&to=1785814327',
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

r = requests.get('https://openapi.dnse.com.vn/price/{symbol}/trades', params={
  'from': '1785727927',  'to': '1785814327'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/price/{symbol}/trades?from=1785727927&to=1785814327");
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
  "trades": [
    {
      "marketId": "STO",
      "boardId": "G1",
      "isin": "VN000000HPG4",
      "symbol": "HPG",
      "matchPrice": 94.1,
      "matchQtty": 10,
      "side": "BUY",
      "avgPrice": 87.872,
      "totalVolumeTraded": 12070,
      "grossTradeAmount": 10.60617,
      "highestPrice": 94.1,
      "lowestPrice": 87.6,
      "openPrice": 87.6,
      "time": "2026-08-04 10:29:51.070"
    }
  ],
  "nextPageToken": "NTkzMDgyMThfMjAyNi0wOC0wNFQwMjoxNjozNy4zNlo="
}
```

<h3 id="gethistorytrades-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» trades|[object]|false|none|Danh sách giao dịch khớp lệnh|
|»» marketId|string|false|none|Mã thị trường niêm yết mã chứng khoán<br>- DVX: Phái sinh sàn HNX<br>- HCX: Trái phiếu doanh nghiệp HNX<br>- STO: Cổ phiếu sàn HOSE<br>- STX: Cổ phiếu sàn HNX<br>- UPX: Cổ phiếu sàn Upcom|
|»» boardId|string|false|none|Mã bảng giao dịch<br>- G1: Lô chẵn<br>- G4: Lô lẻ<br>- T1: Thỏa thuận trong giờ (9h - 14h45)<br>- T3: Thỏa thuận sau giờ (14h45 - 15h)<br>- T4: Thỏa thuận lô lẻ trong giờ (9h - 14h45)<br>- T6: Thỏa thuận lô lẻ sau giờ (14h45 - 15h)|
|»» isin|string|false|none|Mã định danh quốc tế (ISIN) của chứng khoán|
|»» symbol|string|false|none|Mã chứng khoán|
|»» matchPrice|number(double)|false|none|Giá khớp gần nhất|
|»» matchQtty|integer(int32)|false|none|Khối lượng khớp gần nhất|
|»» side|string|false|none|Chiều giao dịch. Giá trị: BUY (Mu chủ động), SELL (Bán chủ động), UNSPECIFIED (Không xác định)|
|»» avgPrice|number(double)|false|none|Giá khớp trung bình|
|»» totalVolumeTraded|integer(int32)|false|none|Tổng khối lượng giao dịch trong ngày|
|»» grossTradeAmount|number(double)|false|none|Tổng giá trị giao dịch trong ngày|
|»» highestPrice|number(double)|false|none|Giá cao nhất trong ngày|
|»» lowestPrice|number(double)|false|none|Giá thấp nhất trong ngày|
|»» openPrice|number(double)|false|none|Giá mở cửa|
|»» time|string|false|none|Thời gian ghi nhận. Định dạng: YYYY-MM-DD HH:mm:ss.SSS (GMT+7)|
|» nextPageToken|string|false|none|Token dùng để lấy trang dữ liệu tiếp theo. Không có hoặc rỗng nếu không còn dữ liệu|

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
