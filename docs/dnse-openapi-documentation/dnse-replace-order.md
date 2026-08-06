## Sửa lệnh

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="replaceOrder"></span>

### `PUT /accounts/{accountNo}/orders/{orderId}`

Gửi yêu cầu sửa lệnh đã đặt theo `orderId`.

- Với cơ sở, lệnh sửa thành công đồng nghĩa với hủy lệnh cũ và đặt lại lệnh mới, người dùng có thể sửa đồng thời giá và khối lượng.
    
- Với phái sinh, người dùng chỉ có thể sửa hoặc giá hoặc khối lượng. Khối lượng sửa phải lớn hơn khối lượng đã khớp (nếu có).

<h3 id="replaceorder-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|string|true|Loại thị trường |
|orderCategory|query|string|true|Phân loại lệnh thường (mặc định NORMAL)|
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|trading-token|header|string|true|Token đặt lệnh|
|version|header|string(date)|false|API version (YYYY-MM-DD)|
|body|body|object|false|none|
|» price|body|number(double)|false|Giá mới cho lệnh LO|
|» quantity|body|integer(int32)|false|Khối lượng mới|
|accountNo|path|string|true|Số tiểu khoản|
|orderId|path|integer|true|Mã lệnh giao dịch|

#### Detailed descriptions

**marketType**: Loại thị trường 
- STOCK: Giao dịch cơ sở
- DERIVATIVE: Giao dịch phái sinh
- BOND: Giao dịch trái phiếu

**» price**: Giá mới cho lệnh LO
- Đối với lệnh cơ sở, có thể sửa cả giá và khối lượng
- Đối với lệnh phái sinh, chỉ được sửa hoặc giá hoặc khối lượng

**» quantity**: Khối lượng mới
- Đối với lệnh phái sinh, khối lượng mới phải lớn hơn khối lượng đã khớp (nếu có) của lệnh đã đặt

> Code samples

```shell
# You can also use wget
curl -X PUT https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}?marketType=DERIVATIVE&orderCategory=NORMAL \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Signature: your_signature' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'trading-token: 7ceef658-9f01-414e-8b3e-faa77bb9061e' \
  -H 'version: 2026-07-23'

```

```http
PUT https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}?marketType=DERIVATIVE&orderCategory=NORMAL HTTP/1.1
Host: openapi.dnse.com.vn
Content-Type: application/json
Accept: application/json
X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==
X-Signature: your_signature
X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000
trading-token: 7ceef658-9f01-414e-8b3e-faa77bb9061e
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
        "Content-Type": []string{"application/json"},
        "Accept": []string{"application/json"},
        "X-API-Key": []string{"eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ=="},
        "X-Signature": []string{"your_signature"},
        "X-Aux-Date": []string{"Mon, 19 Jan 2026 07:45:23 +0000"},
        "trading-token": []string{"7ceef658-9f01-414e-8b3e-faa77bb9061e"},
        "version": []string{"2026-07-23"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("PUT", "https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

```javascript
const inputBody = '{
  "price": 1851,
  "quantity": 3
}';
const headers = {
  'Content-Type':'application/json',
  'Accept':'application/json',
  'X-API-Key':'eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==',
  'X-Signature':'your_signature',
  'X-Aux-Date':'Mon, 19 Jan 2026 07:45:23 +0000',
  'trading-token':'7ceef658-9f01-414e-8b3e-faa77bb9061e',
  'version':'2026-07-23'
};

fetch('https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}?marketType=DERIVATIVE&orderCategory=NORMAL',
{
  method: 'PUT',
  body: inputBody,
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
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'X-API-Key': 'eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==',
  'X-Signature': 'your_signature',
  'X-Aux-Date': 'Mon, 19 Jan 2026 07:45:23 +0000',
  'trading-token': '7ceef658-9f01-414e-8b3e-faa77bb9061e',
  'version': '2026-07-23'
}

r = requests.put('https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}', params={
  'marketType': 'DERIVATIVE',  'orderCategory': 'NORMAL'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/accounts/{accountNo}/orders/{orderId}?marketType=DERIVATIVE&orderCategory=NORMAL");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("PUT");
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

> Body parameter

```json
{
  "price": 1851,
  "quantity": 3
}
```

> Example responses

> 200 Response

```json
{
  "id": 1626,
  "accountNo": "0001179019",
  "side": "NS",
  "loanPackageId": 2278,
  "symbol": "41I1G4000",
  "orderType": "LO",
  "orderCategory": "NORMAL",
  "price": 1851,
  "quantity": 3,
  "fillQuantity": 0,
  "canceledQuantity": 0,
  "marketType": "DERIVATIVE",
  "transDate": "2026-03-16",
  "createdDate": "2026-03-24T04:09:50.761146893Z",
  "modifiedDate": "2026-03-24T04:14:21.004856492Z"
}
```

<h3 id="replaceorder-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» id|integer(int32)|false|none|Id lệnh giao dịch|
|» accountNo|string|false|none|Số tiểu khoản|
|» side|string|false|none|Chiều đặt lệnh<br>- NB: Mua<br>- NS: Bán|
|» loanPackageId|integer(int32)|false|none|Mã gói vay|
|» symbol|string|false|none|Mã chứng khoán|
|» orderType|string|false|none|Loại lệnh<br>- LO: Lệnh giới hạn<br>- MOK/MAK/MTL: Lệnh thị trường<br>- ATO/ATC: Lệnh phiên định kỳ mở cửa/đóng cửa<br>- PLO: Lệnh khớp lệnh sau giờ|
|» orderCategory|string|false|none|Phân loại lệnh thường (mặc định NORMAL)|
|» price|number(double)|false|none|Giá đặt|
|» quantity|integer(int32)|false|none|Khối lượng đặt|
|» fillQuantity|integer(int32)|false|none|Khối lượng khớp|
|» canceledQuantity|integer(int32)|false|none|Khối lượng đã hủy|
|» marketType|string|false|none|Loại thị trường<br>- STOCK: Lệnh cơ sở<br>- DERIVATIVE: Lệnh phái sinh|
|» transDate|string|false|none|Ngày giao dịch|
|» createdDate|string(date-time)|false|none|Thời điểm tạo|
|» modifiedDate|string(date-time)|false|none|Thời điểm cập nhật|

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
