## Chi tiết mã chứng khoán

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getInstruments"></span>

### `GET /instruments`

Truy vấn danh sách thông tin cơ bản của các mã chứng khoán theo điều kiện lọc.

<h3 id="getinstruments-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|symbol|query|string|false|Danh sách mã chứng khoán|
|marketId|query|string|false|Mã thị trường niêm yết|
|securityGroupId|query|string|false|Nhóm chứng khoán|
|indexName|query|string|false|Chỉ số thị trường |
|limit|query|integer|false|Số bản ghi trên mỗi trang|
|page|query|integer|false|Phân trang hiện tại|
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|version|header|string(date)|false|API version (YYYY-MM-DD)|

#### Detailed descriptions

**marketId**: Mã thị trường niêm yết
- STO: Cổ phiếu sàn HOSE
- STX: Cổ phiếu sàn HNX
- UPX: Cổ phiếu sàn UPCOM
- DVX: Phái sinh
- HCX: Trái phiếu doanh nghiệp

**securityGroupId**: Nhóm chứng khoán
- ST: Cổ phiếu
- EF: Quỹ ETF
- EW: Chứng quyền
- FU: Hợp đồng tương lai
- BS: Trái phiếu

**indexName**: Chỉ số thị trường 
- VN30: Top 30 cổ phiếu sàn HOSE
- VN100: Top 100 cổ phiếu sàn HOSE
- HNX30: Top 30 cổ phiếu sàn HNX

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/instruments \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/instruments HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/instruments", data)
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

fetch('https://openapi.dnse.com.vn/instruments',
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

r = requests.get('https://openapi.dnse.com.vn/instruments', headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/instruments");
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
  "data": [
    {
      "symbol": "ACB",
      "marketId": "STO",
      "securityGroupId": "ST",
      "symbolType": "",
      "listedDate": "2020-12-09",
      "shortName": "Ngân hàng Á Châu",
      "name": "Ngân hàng TMCP Á Châu",
      "indexName": [
        "VN100",
        "VN30"
      ]
    }
  ],
  "total": 2,
  "page": 1,
  "pageSize": 100
}
```

<h3 id="getinstruments-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» data|[object]|false|none|Danh sách thông tin mã chứng khoán|
|»» symbol|string|false|none|Mã chứng khoán|
|»» marketId|string|false|none|Mã thị trường niêm yết mã chứng khoán<br>- DVX: Phái sinh sàn HNX<br>- HCX: Trái phiếu doanh nghiệp HNX<br>- STO: Cổ phiếu sàn HOSE<br>- STX: Cổ phiếu sàn HNX<br>- UPX: Cổ phiếu sàn Upcom|
|»» securityGroupId|string|false|none|Nhóm chứng khoán<br>- BS: Trái phiếu doanh nghiệp<br>- EF: Quỹ ETF<br>- EW: Chứng quyền<br>- FU: Hợp đồng tương lai<br>- ST: Cổ phiếu|
|»» symbolType|string|false|none|Phân loại mã hợp đồng phái sinh theo thời gian đáo hạn (áp dụng cho DERIVATIVE)<br>- VN30F1M: HĐTL chỉ số VN30 1 tháng<br>- VN30F2M: HĐTL chỉ số VN30 2 tháng<br>- VN30F1Q: HĐTL chỉ số VN30 1 quý<br>- VN30F2Q: HĐTL chỉ số VN30 2 quý|
|»» listedDate|string|false|none|Ngày niêm yết|
|»» shortName|string|false|none|Tên viết tắt của tổ chức phát hành|
|»» name|string|false|none|Tên đầy đủ của tổ chức phát hành|
|»» indexName|[string]|false|none|Danh sách chỉ số mà mã chứng khoán thuộc về (nếu có)|
|» total|integer(int32)|false|none|Tổng số bản ghi|
|» page|integer(int32)|false|none|Trang hiện tại (bắt đầu từ 1)|
|» pageSize|integer(int32)|false|none|Số bản ghi trên mỗi trang|

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
