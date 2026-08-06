## Dữ liệu NĐT nước ngoài

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="get-foreign-trading"></span>

### `GET /price/{symbol}/foreign-trading`

Truy vấn thông tin dữ liệu nhà đầu tư nước ngoài.

<h3 id="get-foreign-trading-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|boardId|query|string|false|Mã bảng giao dịch|
|from|query|integer|true|Thời gian bắt đầu (timestamp)|
|to|query|integer|true|Thời gian kết thúc (timestamp) (không vượt quá 1 ngày)|
|limit|query|integer|false|none|
|order|query|string|false|none|
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
curl -X GET https://openapi.dnse.com.vn/price/{symbol}/foreign-trading?from=1785729607&to=1785731407 \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/price/{symbol}/foreign-trading?from=1785729607&to=1785731407 HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/price/{symbol}/foreign-trading", data)
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

fetch('https://openapi.dnse.com.vn/price/{symbol}/foreign-trading?from=1785729607&to=1785731407',
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

r = requests.get('https://openapi.dnse.com.vn/price/{symbol}/foreign-trading', params={
  'from': '1785729607',  'to': '1785731407'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/price/{symbol}/foreign-trading?from=1785729607&to=1785731407");
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
  "foreigners": [
    {
      "marketId": "STO",
      "boardId": "G1",
      "symbol": "ACB",
      "tradingSessionId": "40",
      "sellVolume": 200,
      "sellTradedAmount": 19800000,
      "buyVolume": 500,
      "buyTradedAmount": 49350000,
      "totalSellVolume": 200,
      "totalSellTradedAmount": 19800000,
      "totalBuyVolume": 500,
      "totalBuyTradedAmount": 49350000,
      "foreignerOrderLimitQuantity": 62397864,
      "foreignerBuyPossibleQuantity": 265119351,
      "time": "2026-08-03 11:30:00.091"
    }
  ],
  "nextPageToken": "Nl8yXzQwX0FDQl8yMDI2LTA2LTExVDA2OjU0OjAwLjM2OFo="
}
```

<h3 id="get-foreign-trading-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» foreigners|[object]|false|none|Danh sách dữ liệu giao dịch của nhà đầu tư nước ngoài theo mã chứng khoán.|
|»» marketId|string|false|none|Mã thị trường niêm yết mã chứng khoán<br>- DVX: Phái sinh sàn HNX<br>- HCX: Trái phiếu doanh nghiệp HNX<br>- STO: Cổ phiếu sàn HOSE<br>- STX: Cổ phiếu sàn HNX<br>- UPX: Cổ phiếu sàn Upcom|
|»» boardId|string|false|none|Mã bảng giao dịch<br>- G1: Lô chẵn<br>- G4: Lô lẻ<br>- T1: Thỏa thuận trong giờ (9h - 14h45)<br>- T3: Thỏa thuận sau giờ (14h45 - 15h)<br>- T4: Thỏa thuận lô lẻ trong giờ (9h - 14h45)<br>- T6: Thỏa thuận lô lẻ sau giờ (14h45 - 15h)|
|»» symbol|string|false|none|Mã chứng khoán|
|»» tradingSessionId|string|false|none|Mã phiên giao dịch hiện tại<br>- 10: Phiên ATO<br>- 30: Phiên ATC<br>- 40: Phiên liên tục<br>- 80: PCA Mã halt<br>- 99: Đóng bảng|
|»» sellVolume|integer(int32)|false|none|Khối lượng bán khớp lệnh của nhà đầu tư nước ngoài trong phiên|
|»» sellTradedAmount|integer(int32)|false|none|Giá trị bán khớp lệnh của nhà đầu tư nước ngoài trong phiên|
|»» buyVolume|integer(int32)|false|none|Khối lượng mua khớp lệnh của nhà đầu tư nước ngoài trong phiên|
|»» buyTradedAmount|integer(int32)|false|none|Giá trị mua khớp lệnh của nhà đầu tư nước ngoài trong phiên|
|»» totalSellVolume|integer(int32)|false|none|Tổng khối lượng bán lũy kế của nhà đầu tư nước ngoài|
|»» totalSellTradedAmount|integer(int32)|false|none|Tổng giá trị bán lũy kế của nhà đầu tư nước ngoài|
|»» totalBuyVolume|integer(int32)|false|none|Tổng khối lượng mua lũy kế của nhà đầu tư nước ngoà|
|»» totalBuyTradedAmount|integer(int32)|false|none|Tổng giá trị mua lũy kế của nhà đầu tư nước ngoài|
|»» foreignerOrderLimitQuantity|integer(int32)|false|none|Giới hạn khối lượng sở hữu tối đa dành cho nhà đầu tư nước ngoài|
|»» foreignerBuyPossibleQuantity|integer(int32)|false|none|Khối lượng còn lại mà nhà đầu tư nước ngoài được phép mua|
|»» time|string|false|none|Thời gian ghi nhận. Định dạng: YYYY-MM-DD HH:mm:ss.SSS (GMT+7).|
|» nextPageToken|string|false|none|Token dùng để lấy trang dữ liệu tiếp theo. Giá trị null nếu không còn dữ liệu.|

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
