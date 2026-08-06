## Lịch sử bid/ask

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getQuotes"></span>

### `GET /price/{symbol}/quotes`

Truy vấn thông tin lịch sử bid/ask độ sâu thị trường của mã chứng khoán theo bảng giao dịch và khoảng thời gian cụ thể.

<h3 id="getquotes-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|boardId|query|string|true|Mã bảng giao dịch|
|from|query|string|true|Thời gian bắt đầu (timestamp)|
|to|query|string|true|Thời gian kết thúc (timestamp) (không vượt quá 1 ngày)|
|limit|query|integer|false|none|
|X-API-Key|header|string|false|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|false|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|false|Chữ ký xác thực yêu cầu|
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
curl -X GET https://openapi.dnse.com.vn/price/{symbol}/quotes?boardId=G1&from=1785727927&to=1785735127 \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-05-07'

```

```http
GET https://openapi.dnse.com.vn/price/{symbol}/quotes?boardId=G1&from=1785727927&to=1785735127 HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/price/{symbol}/quotes", data)
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

fetch('https://openapi.dnse.com.vn/price/{symbol}/quotes?boardId=G1&from=1785727927&to=1785735127',
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

r = requests.get('https://openapi.dnse.com.vn/price/{symbol}/quotes', params={
  'boardId': 'G1',  'from': '1785727927',  'to': '1785735127'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/price/{symbol}/quotes?boardId=G1&from=1785727927&to=1785735127");
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
  "quotes": [
    {
      "marketId": "STO",
      "boardId": "G1",
      "isin": "VN000000ACB8",
      "symbol": "ACB",
      "bid": [
        {
          "price": 101,
          "quantity": 10
        },
        {
          "price": 100,
          "quantity": 50
        },
        {
          "price": 99.5,
          "quantity": 10
        }
      ],
      "offer": [
        {
          "price": 106,
          "quantity": 4870
        },
        {
          "price": 106.1,
          "quantity": 500
        },
        {
          "price": 110,
          "quantity": 30
        }
      ],
      "totalOfferQtty": 0,
      "totalBidQtty": 0,
      "time": "2026-08-03 10:59:12.336"
    }
  ],
  "nextPageToken": "MTM2NDYxOV8yMDI2LTA1LTI2VDA0OjI3OjQyLjQyNFo="
}
```

<h3 id="getquotes-responseschema">Response Schema</h3>

Status Code **200**

*Danh sách dữ liệu sổ lệnh (Market Depth) cùng thông tin phân trang.*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» quotes|[object]|false|none|Danh sách dữ liệu sổ lệnh của mã chứng khoán|
|»» marketId|string|false|none|Mã thị trường niêm yết mã chứng khoán<br>- DVX: Phái sinh sàn HNX<br>- HCX: Trái phiếu doanh nghiệp HNX<br>- STO: Cổ phiếu sàn HOSE<br>- STX: Cổ phiếu sàn HNX<br>- UPX: Cổ phiếu sàn Upcom|
|»» boardId|string|false|none|Mã bảng giao dịch<br>- G1: Lô chẵn<br>- G4: Lô lẻ<br>- T1: Thỏa thuận trong giờ (9h - 14h45)<br>- T3: Thỏa thuận sau giờ (14h45 - 15h)<br>- T4: Thỏa thuận lô lẻ trong giờ (9h - 14h45)<br>- T6: Thỏa thuận lô lẻ sau giờ (14h45 - 15h)|
|»» isin|string|false|none|Mã định danh quốc tế (ISIN) của chứng khoán.|
|»» symbol|string|false|none|Mã chứng khoán|
|»» bid|[object]|false|none|Danh sách các mức giá chào mua|
|»»» price|number(double)|false|none|Giá đặt mua|
|»»» quantity|integer(int32)|false|none|Khối lượng đặt mua tại mức giá tương ứng|
|»» offer|[object]|false|none|Danh sách các mức giá chào bán|
|»»» price|number(double)|false|none|Giá đặt bán|
|»»» quantity|integer(int32)|false|none|Khối lượng đặt bán tại mức giá tương ứng|
|»» totalOfferQtty|integer(int32)|false|none|Tổng khối lượng dư bán|
|»» totalBidQtty|integer(int32)|false|none|Tổng khối lượng dư mua|
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
