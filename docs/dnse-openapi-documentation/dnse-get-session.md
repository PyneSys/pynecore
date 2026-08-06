## Phiên giao dịch

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="get-session"></span>

### `GET /market/trading-session`

Truy vấn thông tin phiên giao dịch hiện tại.

<h3 id="get-session-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|boardId|query|string|false|Mã bảng giao dịch|
|tscProdGrpId|query|string|false|Nhóm sản phẩm theo thị trường|
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Chữ ký xác thực yêu cầu|
|X-Signature|header|string|true|Thời gian thực hiện yêu cầu|
|version|header|string|false|API version (YYYY-MM-DD)|

#### Detailed descriptions

**boardId**: Mã bảng giao dịch
- G1: Lô chẵn
- G4: Lô lẻ
- T1: Thỏa thuận trong giờ (9h - 14h45)
- T3: Thỏa thuận sau giờ (14h45 - 15h)
- T4: Thỏa thuận lô lẻ trong giờ (9h - 14h45)
- T6: Thỏa thuận lô lẻ sau giờ  (14h45 - 15h)

**tscProdGrpId**: Nhóm sản phẩm theo thị trường
- FBX: Hợp đồng tương lai Trái phiếu
- FIO: Hợp đồng tương lai Chỉ số
- HCX: Trái phiếu Doanh nghiệp HNX
- STO: Cổ phiếu sàn HOSE
- STX: Cổ phiếu sàn HNX
- UPX: Cổ phiếu sàn Upcom

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/market/trading-session \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/market/trading-session HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/market/trading-session", data)
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

fetch('https://openapi.dnse.com.vn/market/trading-session',
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

r = requests.get('https://openapi.dnse.com.vn/market/trading-session', headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/market/trading-session");
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
  "tradingSessions": [
    {
      "marketId": "STO",
      "boardId": "G1",
      "tscProdGrpId": "STO",
      "tradingSessionId": "99",
      "eventId": "AC2",
      "time": "2026-06-26 14:45:00.960"
    }
  ]
}
```

<h3 id="get-session-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» tradingSessions|[object]|false|none|none|
|»» marketId|string|false|none|Mã thị trường niêm yết mã chứng khoán<br>- DVX: Phái sinh sàn HNX<br>- HCX: Trái phiếu doanh nghiệp HNX<br>- STO: Cổ phiếu sàn HOSE<br>- STX: Cổ phiếu sàn HNX<br>- UPX: Cổ phiếu sàn Upcom|
|»» boardId|string|false|none|Mã bảng giao dịch<br>- G1: Lô chẵn<br>- G4: Lô lẻ<br>- T1: Thỏa thuận trong giờ (9h - 14h45)<br>- T3: Thỏa thuận sau giờ (14h45 - 15h)<br>- T4: Thỏa thuận lô lẻ trong giờ (9h - 14h45)<br>- T6: Thỏa thuận lô lẻ sau giờ (14h45 - 15h)|
|»» tscProdGrpId|string|false|none|Nhóm sản phẩm theo thị trường<br>- FBX: Hợp đồng tương lai Trái phiếu<br>- FIO: Hợp đồng tương lai Chỉ số<br>- HCX: Trái phiếu Doanh nghiệp HNX<br>- STO: Cổ phiếu sàn HOSE<br>- STX: Cổ phiếu sàn HNX<br>- UPX: Cổ phiếu sàn Upcom|
|»» tradingSessionId|string|false|none|Mã phiên giao dịch hiện tại<br>- 10: Phiên ATO<br>- 30: Phiên ATC<br>- 40: Phiên liên tục<br>- 80: PCA Mã halt<br>- v99: Đóng bảng|
|»» eventId|string|false|none|Mã sự kiện chuyển trạng thái phiên giao dịch:<br>- AA1: Mở phiên định kỳ mở cửa<br>- AB1: Mở phiên giao dịch liên tục<br>- AB2: Kết thúc giao dịch của bảng<br>- AC2: Thực hiện khớp lệnh định kỳ đóng cửa<br>- AD1: Bắt đầu nhận lệnh<br>- AD2: Thực hiện khớp lệnh định kỳ<br>- AW8: Nghỉ trưa<br>- AW9: Tiếp tục giao dịch sau nghỉ trưa<br>- AX1: Bảng bắt đầu giao dịch<br>- BB1: Bắt đầu giao dịch phiên liên tục với chứng khoán trạng thái thông thường<br>- BC1: Bắt đầu phiên định kỳ đóng cửa<br>- CC1: Bắt đầu phiên PCA (khớp lệnh định kỳ nhiều đợt) đóng cửa<br>- CD1: Bắt đầu phiên PCA (khớp lệnh định kỳ nhiều đợt)<br>- CD3: Thực hiện khớp lệnh định kỳ nhiều đợt (PCA)|
|»» time|string|false|none|Thời gian ghi nhận định dạng (YYYY-MM-DD HH:mm:ss.SSS (GMT+7))|

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
