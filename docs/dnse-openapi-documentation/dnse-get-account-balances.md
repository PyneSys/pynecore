## Thông tin tiền

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getAccountBalances"></span>

### `GET /accounts/{accountNo}/balances`

Cung cấp thông tin số dư tài khoản

<h3 id="getaccountbalances-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|version|header|string(date)|false|API version (YYYY-MM-DD)|
|accountNo|path|string|true|Số tiểu khoản|

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/accounts/{accountNo}/balances \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/accounts/{accountNo}/balances HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/accounts/{accountNo}/balances", data)
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

fetch('https://openapi.dnse.com.vn/accounts/{accountNo}/balances',
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

r = requests.get('https://openapi.dnse.com.vn/accounts/{accountNo}/balances', headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/accounts/{accountNo}/balances");
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
  "stock": {
    "totalCash": 316338724,
    "availableCash": 316341327,
    "depositInterest": 71570,
    "totalDebt": 9491548,
    "depositFeeAmount": 4122,
    "secureAmount": 11174173,
    "orderSecured": 7903713,
    "withdrawableCash": 312369318,
    "cashDividendReceiving": 11100000
  },
  "derivative": {
    "pendingDepositWithdraw": 826996062,
    "remainSecure": 120234072489,
    "usedSecure": 307491492,
    "pendingSecure": 826996062,
    "holdTaxAndFee": 0,
    "totalLoanDebt": 4907500
  },
  "bond": {
    "totalValue": 40102762100
  },
  "egg": {
    "totalValue": 6849005
  }
}
```

<h3 id="getaccountbalances-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» stock|object|false|none|Thông tin tài sản chứng khoán cơ sở|
|»» totalCash|integer(int32)|false|none|Tổng tiền (Tiền mặt + Lãi tiền gửi + Tiền bán chờ về + Tiền cổ tức chờ về - Tiền mua trong ngày)|
|»» availableCash|integer(int32)|false|none|Tiền mặt hiện có|
|»» depositInterest|integer(int32)|false|none|Lãi tiền gửi không kỳ hạn|
|»» totalDebt|integer(int32)|false|none|Tổng nợ (Nợ margin tạm thu + Nợ margin còn lại)|
|»» depositFeeAmount|integer(int32)|false|none|Phí lưu ký.|
|»» secureAmount|integer(int32)|false|none|Tổng tiền ký quỹ.|
|»» orderSecured|integer(int32)|false|none|Số tiền đang được phong tỏa cho các lệnh giao dịch.|
|»» withdrawableCash|integer(int32)|false|none|Tiền có thể rút.|
|»» cashDividendReceiving|integer(int32)|false|none|Tiền cổ tức chờ về.|
|» derivative|object|false|none|Thông tin tài sản chứng khoán phái sinh.|
|»» pendingDepositWithdraw|integer(int32)|false|none|Tiền nộp/rút cọc chờ xử lý|
|»» remainSecure|integer(int64)|false|none|Cọc còn lại|
|»» usedSecure|integer(int32)|false|none|Cọc đã sử dụng|
|»» pendingSecure|integer(int32)|false|none|Cọc chờ duyệt|
|»» holdTaxAndFee|integer(int32)|false|none|Thuế và phí tạm giữ|
|»» totalLoanDebt|integer(int32)|false|none|Khoản ứng chưa hoàn|
|» bond|object|false|none|Thông tin tài sản trái phiếu|
|»» totalValue|integer(int64)|false|none|Tổng giá trị tài sản trái phiếu|
|» egg|object|false|none|Thông tin tài sản trứng vàng|
|»» totalValue|integer(int32)|false|none|Tổng giá trị tài sản trứng|

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
