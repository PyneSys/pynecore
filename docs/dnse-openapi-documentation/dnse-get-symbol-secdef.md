## Thông tin giao dịch chứng khoán

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getSymbolSecdef"></span>

### `GET /price/{symbol}/secdef`

Truy vấn thông tin về giá trần/sàn/tham chiếu và trạng thái của mã chứng khoán vào ngày giao dịch.

<h3 id="getsymbolsecdef-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|boardId|query|string|false|Mã bảng giao dịch|
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
curl -X GET https://openapi.dnse.com.vn/price/{symbol}/secdef \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/price/{symbol}/secdef HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/price/{symbol}/secdef", data)
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

fetch('https://openapi.dnse.com.vn/price/{symbol}/secdef',
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

r = requests.get('https://openapi.dnse.com.vn/price/{symbol}/secdef', headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/price/{symbol}/secdef");
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
[
  {
    "marketId": "STO",
    "boardId": "G1",
    "isin": "VN000000HPG4",
    "symbol": "HPG",
    "productGrpId": "STO",
    "securityGroupId": "ST",
    "basicPrice": 94.1,
    "ceilingPrice": 100.6,
    "floorPrice": 87.6,
    "securityStatus": "NO_HALT",
    "symbolAdminStatusCode": "NRM",
    "symbolTradingMethodStatusCode": "NRM",
    "symbolTradingSanctionStatusCode": "NRM",
    "finalTradeDate": null,
    "listingDate": "2007-11-15T00:00:00Z",
    "time": "2026-08-04 08:00:24.008"
  }
]
```

<h3 id="getsymbolsecdef-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» marketId|string|false|none|Mã thị trường niêm yết mã chứng khoán<br>- DVX: Phái sinh sàn HNX<br>- HCX: Trái phiếu doanh nghiệp HNX<br>- STO: Cổ phiếu sàn HOSE<br>- STX: Cổ phiếu sàn HNX<br>- UPX: Cổ phiếu sàn Upcom|
|» boardId|string|false|none|Mã bảng giao dịch<br>- G1: Lô chẵn<br>- G4: Lô lẻ<br>- T1: Thỏa thuận trong giờ (9h - 14h45)<br>- T3: Thỏa thuận sau giờ (14h45 - 15h)<br>- T4: Thỏa thuận lô lẻ trong giờ (9h - 14h45)<br>- T6: Thỏa thuận lô lẻ sau giờ (14h45 - 15h)|
|» isin|string|false|none|Mã định danh quốc tế|
|» symbol|string|false|none|Mã chứng khoán|
|» productGrpId|string|false|none|Nhóm sản phẩm theo thị trường<br>- FBX: Hợp đồng tương lai Trái phiếu<br>- FIO: Hợp đồng tương lai Chỉ số<br>- HCX: Trái phiếu Doanh nghiệp HNX<br>- STO: Cổ phiếu sàn HOSE<br>- STX: Cổ phiếu sàn HNX<br>- UPX: Cổ phiếu sàn Upcom|
|» securityGroupId|string|false|none|Nhóm chứng khoán<br>- BS: Trái phiếu doanh nghiệp<br>- EF: Quỹ ETF<br>- EW: Chứng quyền<br>- FU: Hợp đồng tương lai<br>- ST: Cổ phiếu|
|» basicPrice|number(double)|false|none|Giá tham chiếu ngày giao dịch|
|» ceilingPrice|number(double)|false|none|Giá trần ngày giao dịch|
|» floorPrice|number(double)|false|none|Giá sàn ngày giao dịch|
|» securityStatus|string|false|none|Trạng thái giao dịch của mã chứng khoán<br>- HALT: Ngừng giao dịch<br>- NO_HALT: Không ngừng giao dịch|
|» symbolAdminStatusCode|string|false|none|Trạng thái quản lý hành chính mã chứng khoán<br>-  CR: Kiểm soát và hạn chế giao dịch<br>- CTR: Kiểm soát<br>- NRM: Bình thường<br>- RES: Hạn chế giao dịch<br>- WFR: Cảnh báo vi phạm BCTC<br>- WID: Cảnh báo vi phạm CBTT<br>- WOV: Cảnh báo vi phạm khác|
|» symbolTradingMethodStatusCode|string|false|none|Trạng thái cơ chế giao dịch mã chứng khoán<br>- NRM: Bình thường<br>- NWE: Niêm yết mới (biên độ đặc biệt)<br>-  NWN: Niêm yết mới (biên độ thường)<br>- SLS: Giao dịch đặc biệt sau tạm ngưng<br>- SNE: Giao dịch đặc biệt không có giao dịch dài hạn|
|» symbolTradingSanctionStatusCode|string|false|none|Tình trạng giao dịch của mã chứng khoán<br>- NRM: Bình thường<br>-  SUS: Tạm ngừng giao dịch<br>- DTL: Hủy niêm yết để chuyển sàn<br>- TFR: Ngưng giao dịch do hạn chế|
|» finalTradeDate|any|false|none|Ngày giao dịch cuối cùng (nếu có)|
|» listingDate|string(date-time)|false|none|Ngày niêm yế|
|» time|string|false|none|Thời gian ghi nhận (YYYY-MM-DD HH:mm:ss.SSS (GMT+7))|

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
