## Danh sách gói vay

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getLoanPackages"></span>

### `GET /accounts/{accountNo}/loan-packages`

Truy vấn danh sách gói vay để đặt lệnh theo từng mã chứng khoán.

Với giao dịch cơ sở, trả về **tối đa 2 gói vay** bao gồm:

- Gói tiền mặt: Tỷ lệ ký quỹ tiền mặt 100%, không sử dụng tiền vay margin (`initialRate` = 1)
    
- Gói vay ký quỹ (margin): Tỷ lệ ký quỹ dưới 100%, có sử dụng đòn bẩy tiền vay margin (`initialRate` ≠ 1)
    

Với giao dịch phái sinh, tài khoản thường chỉ áp dụng **một gói vay duy nhất** cho tất cả các mã, với bộ tỷ lệ ký quỹ và phí cố định.

<h3 id="getloanpackages-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|string|true|Loại thị trường |
|symbol|query|string|true|Mã chứng khoán|
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|version|header|string(date)|false|API version (YYYY-MM-DD)|
|accountNo|path|string|true|Số tiểu khoản|

#### Detailed descriptions

**marketType**: Loại thị trường 
- STOCK: Gói vay giao dịch cơ sở
- DERIVATIVE: Gói vay giao dịch phái sinh
- BOND: Gói vay giao dịch trái phiếu

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/accounts/{accountNo}/loan-packages?marketType=STOCK&symbol=ACB \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/accounts/{accountNo}/loan-packages?marketType=STOCK&symbol=ACB HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/accounts/{accountNo}/loan-packages", data)
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

fetch('https://openapi.dnse.com.vn/accounts/{accountNo}/loan-packages?marketType=STOCK&symbol=ACB',
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

r = requests.get('https://openapi.dnse.com.vn/accounts/{accountNo}/loan-packages', params={
  'marketType': 'STOCK',  'symbol': 'ACB'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/accounts/{accountNo}/loan-packages?marketType=STOCK&symbol=ACB");
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

> STOCK

```json
{
  "symbol": "ACB",
  "marketType": "STOCK",
  "loanPackages": [
    {
      "id": 1775,
      "name": "GD Tiền mặt",
      "initialRate": 1,
      "interestRate": 0.125,
      "liquidRate": 0.3,
      "maintenanceRate": 0.4,
      "type": "M",
      "brokerFirmBuyingFeeRate": 0,
      "brokerFirmSellingFeeRate": 0
    },
    {
      "id": 1769,
      "name": "RocketX",
      "initialRate": 0.5,
      "interestRate": 0.125,
      "liquidRate": 0.3,
      "maintenanceRate": 0.4,
      "type": "M",
      "brokerFirmBuyingFeeRate": 0.00045,
      "brokerFirmSellingFeeRate": 0.00045
    }
  ]
}
```

```json
{
  "symbolType": "VN30F1M",
  "marketType": "DERIVATIVE",
  "loanPackages": [
    {
      "id": 2279,
      "name": "Gói giao dịch 02",
      "initialRate": 0.1848,
      "maintenanceRate": 0.1735,
      "liquidRate": 0.1731,
      "tradingFee": {
        "id": 2404,
        "name": "2000/HĐ",
        "scope": "PRODUCT",
        "channel": "ALL",
        "schemaType": "FIXED",
        "createdDate": "2022-12-13T08:22:12.530837Z",
        "modifiedDate": "2022-12-13T08:22:12.530837Z",
        "fixedTradingFee": 2000,
        "fixedDailyCloseTradingFee": 2000
      }
    },
    {
      "id": 5058,
      "name": "Test Phái sinh 1212",
      "initialRate": 0.1,
      "maintenanceRate": 0.03,
      "liquidRate": 0.02,
      "tradingFee": {
        "id": 5057,
        "name": "1500/HĐ",
        "scope": "PRODUCT",
        "channel": "ALL",
        "schemaType": "FIXED",
        "createdDate": "2023-12-12T03:55:37.22881Z",
        "modifiedDate": "2023-12-12T03:55:37.22881Z",
        "fixedTradingFee": 1500,
        "fixedDailyCloseTradingFee": 1500
      }
    },
    {
      "id": 2278,
      "name": "Gói giao dịch 01",
      "initialRate": 0.2065,
      "maintenanceRate": 0.1979,
      "liquidRate": 0.1938,
      "tradingFee": {
        "id": 2436,
        "name": "Miễn phí",
        "scope": "PRODUCT",
        "channel": "ALL",
        "schemaType": "FIXED",
        "createdDate": "2023-02-02T04:22:56.199278Z",
        "modifiedDate": "2023-02-02T04:22:56.199278Z",
        "fixedTradingFee": 2000,
        "fixedDailyCloseTradingFee": 2000
      }
    },
    {
      "id": 9990,
      "name": "Linhpham test",
      "initialRate": 0.2065,
      "maintenanceRate": 0.1979,
      "liquidRate": 0.1938,
      "tradingFee": {
        "id": 9989,
        "name": "fdfhv",
        "scope": "PRODUCT",
        "channel": "ALL",
        "schemaType": "PROGRESSIVE",
        "createdDate": "2026-01-16T09:09:15.556640382Z",
        "modifiedDate": "2026-01-16T09:09:15.556640382Z",
        "progressTradingFee": [
          {
            "fromQuantity": 1,
            "toQuantity": 2000,
            "fee": 1500
          }
        ],
        "progressDailyCloseTradingFee": [
          {
            "fromQuantity": 2001,
            "toQuantity": 3000,
            "fee": 2196
          }
        ]
      }
    }
  ]
}
```

> 400 Response

```json
{
  "code": "OA-003",
  "message": "Thông tin nhập không hợp lệ",
  "status": 400
}
```

<h3 id="getloanpackages-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» symbol|string|false|none|Mã chứng khoán|
|» marketType|string|false|none|Loại thị trường<br>- STOCK: Gói vay giao dịch cơ sở<br>- DERIVATIVE: Gói vay giao dịch phái sinh|
|» loanPackages|[object]|false|none|none|
|»» id|integer(int32)|false|none|Id gói vay|
|»» name|string|false|none|Tên gói vay|
|»» initialRate|integer(int32)|false|none|Tỷ lệ ban đầu|
|»» interestRate|number(float)|false|none|Tỷ lệ lãi vay|
|»» liquidRate|number(double)|false|none|Tỷ lệ xử lý (force sell)|
|»» maintenanceRate|number(double)|false|none|Tỷ lệ duy trì (call margin)|
|»» type|string|false|none|Loại gói vay|
|»» brokerFirmBuyingFeeRate|number(double)|false|none|Phí giao dịch chiều mua công ty chứng khoán thu|
|»» brokerFirmSellingFeeRate|number(double)|false|none|Phí giao dịch chiều bán công ty chứng khoán thu|

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
