## Đặt lệnh

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="postAccountsAccountNoOrders"></span>

### `POST /accounts/{accountNo}/orders`

Gửi yêu cầu đặt lệnh giao dịch trên tài khoản. Hỗ trợ các loại lệnh: 
- Lệnh thường NORMAL: Cơ sở, Phái sinh, Trái phiếu            
- Lệnh STOP: Cơ sở, Phái sinh
    
- Lệnh OCO: Phái sinh
    

Body request sẽ gồm các trường bắt buộc và khác nhau theo từng loại lệnh. Người dùng tham khảo hướng dẫn đặt lệnh ([tại đây](https://developers.dnse.com.vn/docs/guide/trading-api/trading_order))

<h3 id="postaccountsaccountnoorders-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|string|true|Loại thị trường |
|orderCategory|query|string|true|Phân loại lệnh |
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|trading-token|header|string|true|Token đặt lệnh|
|version|header|string(date)|true|API version (YYYY-MM-DD)|
|body|body|object|false|none|
|» symbol|body|string|false|Mã chứng khoán cần đặt lệnh|
|» loanPackageId|body|integer(int32)|false|Mã gói vay theo mã chứng khoán|
|» orderType|body|string|false|- Loại lệnh với NORMAL: LO (lệnh giới hạn), MOK/MAK/MTL (lệnh thị trường), ATO/ATC (lệnh phiên định kỳ mở/đóng cửa), PLO (lệnh sau giờ)|
|» price|body|number(double)|false|- Lệnh NORMAL: Giá đặt|
|» quantity|body|integer(int32)|false|Khối lượng đặt|
|» side|body|string|false|Chiều đặt lệnh|
|» stopPrice|body|number(double)|false|Giá điều kiện dùng để kích hoạt lệnh dừng (lệnh STOP, OCO)|
|» stopOrderPrice|body|number(double)|false|Giá đặt của lệnh cắt lỗ (chỉ áp dụng với lệnh OCO)|
|» conditionOperator|body|string|false|Điều kiện kích hoạt lệnh dừng (chỉ áp dụng với lệnh STOP)|
|» durationType|body|string|false|Hiệu lực của lệnh|
|» durationDateTime|body|string(date-time)|false|Ngày, giờ hết hiệu lực của lệnh. Chỉ áp dụng khi durationType = GTD (lệnh STOP)|
|accountNo|path|string|true|Số tiểu khoản|

#### Detailed descriptions

**marketType**: Loại thị trường 
- STOCK: Giao dịch cơ sở
- DERIVATIVE: Giao dịch phái sinh
- BOND: Giao dịch trái phiếu

**orderCategory**: Phân loại lệnh 
- NORMAL: lệnh thường  (cơ sở, phái sinh, trái phiếu)
- STOP: lệnh dừng có điều kiện (cơ sở, phái sinh)
- OCO: lệnh OCO (phái sinh)

**» orderType**: - Loại lệnh với NORMAL: LO (lệnh giới hạn), MOK/MAK/MTL (lệnh thị trường), ATO/ATC (lệnh phiên định kỳ mở/đóng cửa), PLO (lệnh sau giờ)
- Loại lệnh với STOP: LO (lệnh giới hạn), MTL (lệnh thị trường)
- Loại lệnh với OCO: LO (lệnh giới hạn)

**» price**: - Lệnh NORMAL: Giá đặt
- Lệnh STOP: Giá đặt lệnh dừng
- Lệnh OCO: Giá đặt lệnh chốt lời

**» side**: Chiều đặt lệnh
- NB: Mua
- NS: Bán

**» conditionOperator**: Điều kiện kích hoạt lệnh dừng (chỉ áp dụng với lệnh STOP)
- `>=`: Kích hoạt khi giá thị trường lớn hơn hoặc bằng giá điều kiện
- `<=`: Kích hoạt khi giá thị trường nhỏ hơn hoặc bằng giá điều kiện

**» durationType**: Hiệu lực của lệnh
- GTD: Có hiệu lực đến ngày, giờ chỉ định (lệnh STOP)
- DAY: Lệnh có hiệu lực trong ngày (lệnh OCO)

> Code samples

```shell
# You can also use wget
curl -X POST https://openapi.dnse.com.vn/accounts/{accountNo}/orders?marketType=DERIVATIVE&orderCategory=STOP \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Signature: your_signature' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'trading-token: 7ceef658-9f01-414e-8b3e-faa77bb9061e' \
  -H 'version: 2026-07-23'

```

```http
POST https://openapi.dnse.com.vn/accounts/{accountNo}/orders?marketType=DERIVATIVE&orderCategory=STOP HTTP/1.1
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
    req, err := http.NewRequest("POST", "https://openapi.dnse.com.vn/accounts/{accountNo}/orders", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

```javascript
const inputBody = '{
  "symbol": "MBS",
  "loanPackageId": 5757,
  "orderType": "LO",
  "price": 18600,
  "quantity": 300,
  "side": "NB"
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

fetch('https://openapi.dnse.com.vn/accounts/{accountNo}/orders?marketType=DERIVATIVE&orderCategory=STOP',
{
  method: 'POST',
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

r = requests.post('https://openapi.dnse.com.vn/accounts/{accountNo}/orders', params={
  'marketType': 'DERIVATIVE',  'orderCategory': 'STOP'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/accounts/{accountNo}/orders?marketType=DERIVATIVE&orderCategory=STOP");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("POST");
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
  "symbol": "MBS",
  "loanPackageId": 5757,
  "orderType": "LO",
  "price": 18600,
  "quantity": 300,
  "side": "NB"
}
```

> Example responses

> OK

```json
{
  "id": "1631",
  "accountNo": "0001179019",
  "orderCategory": "NORMAL",
  "marketType": "STOCK",
  "symbol": "MBS",
  "side": "NB",
  "orderType": "LO",
  "orderStatus": "PendingNew",
  "price": 18600,
  "quantity": 300,
  "loanPackageId": 5757,
  "transDate": "2026-01-29",
  "createdDate": "2026-08-03T07:27:34.272131175Z",
  "modifiedDate": "2026-08-03T07:27:34.272132175Z"
}
```

```json
{
  "id": "2230",
  "accountNo": "0001179019",
  "orderCategory": "NORMAL",
  "marketType": "DERIVATIVE",
  "symbol": "41I1G9000",
  "side": "NB",
  "orderType": "LO",
  "orderStatus": "PendingNew",
  "price": 1990,
  "quantity": 3,
  "loanPackageId": 5757,
  "transDate": "2026-08-03",
  "createdDate": "2026-08-03T07:27:34.272131175Z",
  "modifiedDate": "2026-08-03T07:27:34.272132175Z"
}
```

```json
{
  "id": "d9guo2d1j9cc72osmg1g",
  "accountNo": "0001179019",
  "orderCategory": "STOP",
  "marketType": "DERIVATIVE",
  "symbol": "41I1G8000",
  "side": "NB",
  "orderType": "LO",
  "orderStatus": "New",
  "price": 1990,
  "quantity": 3,
  "loanPackageId": 2278,
  "stopPrice": 2000,
  "conditionOperator": ">=",
  "durationType": "GTD",
  "durationDateTime": "2026-08-01T07:30:00+07:00",
  "createdDate": "2026-07-23T10:25:13.251588Z",
  "modifiedDate": "2026-07-23T10:25:13.251588Z"
}
```

```json
{
  "accountNo": "0001179019",
  "createdDate": "2026-08-04T02:24:40.055903Z",
  "durationType": "DAY",
  "id": "d9okqq13qkqc72rbl930",
  "loanPackageId": 2278,
  "marketType": "DERIVATIVE",
  "modifiedDate": "2026-08-04T02:24:40.055903Z",
  "orderCategory": "OCO",
  "orderStatus": "New",
  "orderType": "LO",
  "price": 1916,
  "quantity": 3,
  "side": "NS",
  "stopOrderPrice": 1907,
  "stopPrice": 1910,
  "symbol": "41I1G9000"
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

<h3 id="postaccountsaccountnoorders-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» id|string|false|none|Id lệnh giao dịch|
|» accountNo|string|false|none|Số tiểu khoản|
|» orderCategory|string|false|none|Phân loại lệnh<br>- NORMAL: Lệnh thường<br>- STOP: Lệnh dừng có điều kiện<br>- OCO: Lệnh OCO phái sinh|
|» marketType|string|false|none|Loại thị trường<br>- STOCK: Giao dịch cơ sở<br>- DERIVATIVE: Giao dịch phái sinh<br>- BOND: Giao dịch trái phiếu|
|» symbol|string|false|none|Mã chứng khoán|
|» side|string|false|none|Chiều đặt lệnh<br>- NB: Mua<br>- NS: Bán|
|» orderType|string|false|none|Loại lệnh<br>- LO: Lệnh giới hạn<br>- MOK/MAK/MTL: Lệnh thị trường<br>- ATO/ATC: Lệnh phiên định kỳ mở cửa/đóng cửa<br>- PLO: Lệnh khớp lệnh sau giờ|
|» orderStatus|string|false|none|Trạng thái lệnh khi đặt thành công<br>- PendingNew: Chờ gửi (lệnh thường NORMAL)<br>- New: Chờ kích hoạt (lệnh STOP/OCO)|
|» price|number(double)|false|none|Giá đặt|
|» quantity|integer(int32)|false|none|Khối lượng đặt|
|» loanPackageId|integer(int32)|false|none|ID gói vay|
|» transDate|string(date)|false|none|Ngày giao dịch|
|» createdDate|string(date-time)|false|none|Thời điểm tạo lệnh|
|» modifiedDate|string(date-time)|false|none|Thời điểm cập nhật lệnh|

Status Code **201**

*Thông tin lệnh điều kiện.*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» id|string|false|none|Id lệnh giao dịch.|
|» accountNo|string|false|none|Số tiểu khoản.|
|» orderCategory|string|false|none|Phân loại lệnh<br>- NORMAL: Lệnh thường<br>- STOP: Lệnh dừng có điều kiện<br>- OCO: Lệnh OCO phái sinh|
|» marketType|string|false|none|Loại thị trường<br>- STOCK: Lệnh cơ sở<br>- DERIVATIVE: Lệnh phái sinh|
|» symbol|string|false|none|Mã chứng khoán|
|» side|string|false|none|Chiều đặt lệnh<br>- NB: Mua<br>- NS: Bán|
|» orderType|string|false|none|Loại lệnh đặt cho lệnh dừng (lệnh STOP/OCO)<br>- LO: Lệnh giới hạn<br>- MTL: Lệnh thị trường|
|» orderStatus|string|false|none|Trạng thái lệnh khi đặt thành công<br>- PendingNew: Chờ gửi (lệnh thường NORMAL)<br>- New: Chờ kích hoạt (lệnh STOP/OCO)|
|» price|integer|false|none|Giá đặt|
|» quantity|integer|false|none|Khối lượng đặt|
|» loanPackageId|integer|false|none|ID gói vay|
|» stopPrice|integer|false|none|Giá điều kiện kích hoạt lệnh dừng|
|» conditionOperator|string|false|none|Điều kiện kích hoạt lệnh dừng (chỉ áp dụng với lệnh STOP)<br><br>- `>=`: Kích hoạt khi giá thị trường lớn hơn hoặc bằng giá điều kiện<br><br>- `<=`: Kích hoạt khi giá thị trường nhỏ hơn hoặc bằng giá điều kiện|
|» durationType|string|false|none|Thời hạn hiệu lực của lệnh<br>- DAY: Lệnh có hiệu lực trong ngày (lệnh OCO)<br>- GTD: Có hiệu lực đến ngày, giờ chỉ định (lệnh STOP)|
|» durationDateTime|string(date-time)|false|none|Thời điểm hết hiệu lực của lệnh khi durationType là GTD (lệnh STOP).|
|» createdDate|string(date-time)|false|none|Thời điểm tạo lệnh.|
|» modifiedDate|string(date-time)|false|none|Thời điểm cập nhật lệnh.|

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

### Response Headers

|Status|Header|Type|Format|Description|
|---|---|---|---|---|
|201|Content-Length|integer||none|
|201|Date|string||none|
|201|Vary|string||none|
|201|X-Ratelimit-Limit|integer||none|
|201|X-Ratelimit-Remaining|integer||none|
|201|X-Ratelimit-Reset|integer||none|
|201|X-Request-Id|string||none|
|201|X-Tyk-Api-Expires|string||none|
