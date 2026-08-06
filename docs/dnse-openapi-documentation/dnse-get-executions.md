## Chi tiết trạng thái lệnh

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getExecutions"></span>

### `GET /accounts/{accountNo}/executions/{orderId}`

Lấy lịch sử các lần cập nhật trạng thái hay khớp từng phần của một lệnh giao dịch theo `orderId` , chỉ áp dụng cho lệnh thường thị trường cơ sở và phái sinh, trái phiếu.

<h3 id="getexecutions-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|string|true|Loại thị trường |
|orderCategory|query|string|true|Phân loại lệnh thường (mặc định NORMAL)|
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|version|header|string(date)|false|API version (YYYY-MM-DD)|
|accountNo|path|string|true|Số tiểu khoản|
|orderId|path|integer|true|Id lệnh giao dịch|

#### Detailed descriptions

**marketType**: Loại thị trường 
- STOCK: Lệnh cơ sở
- DERIVATIVE: Lệnh phái sinh
- BOND: Lệnh trái phiếu

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/accounts/{accountNo}/executions/{orderId}?marketType=DERIVATIVE&orderCategory=NORMAL \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/accounts/{accountNo}/executions/{orderId}?marketType=DERIVATIVE&orderCategory=NORMAL HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/accounts/{accountNo}/executions/{orderId}", data)
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

fetch('https://openapi.dnse.com.vn/accounts/{accountNo}/executions/{orderId}?marketType=DERIVATIVE&orderCategory=NORMAL',
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

r = requests.get('https://openapi.dnse.com.vn/accounts/{accountNo}/executions/{orderId}', params={
  'marketType': 'DERIVATIVE',  'orderCategory': 'NORMAL'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/accounts/{accountNo}/executions/{orderId}?marketType=DERIVATIVE&orderCategory=NORMAL");
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

> OK

```json
{
  "id": 1651,
  "side": "NB",
  "accountNo": "0001179019",
  "symbol": "HPG",
  "price": 24250,
  "quantity": 500,
  "orderType": "LO",
  "loanPackageId": 5757,
  "orderCategory": "NORMAL",
  "orderStatus": "PartiallyFilled",
  "fillQuantity": 200,
  "lastQuantity": 100,
  "lastPrice": 24250,
  "averagePrice": 24250,
  "transDate": "2026-07-13",
  "taxRate": 0,
  "exchangeFeeRate": 0.00027,
  "feeRate": 0.00027,
  "leaveQuantity": 300,
  "canceledQuantity": 0,
  "error": "",
  "marketType": "STOCK",
  "priceSecure": 24250,
  "createdDate": "2026-07-13T06:50:36.532741Z",
  "modifiedDate": "2026-07-13T06:50:59.052768Z",
  "metadata": "{\"orderSession\":\"OPEN\",\"ip\":\"202.60.110.58\",\"maker\":\"1000005917\",\"isForeigner\":false,\"reqId\":\"24534\",\"probType\":\"CUSTOMER_8000\",\"eventNo\":4.0}",
  "reports": [
    {
      "id": 1651,
      "side": "NB",
      "accountNo": "0001179019",
      "symbol": "HPG",
      "price": 24250,
      "quantity": 500,
      "orderType": "LO",
      "orderStatus": "PartiallyFilled",
      "fillQuantity": 100,
      "lastQuantity": 100,
      "lastPrice": 24250,
      "averagePrice": 24250,
      "transDate": "2026-07-13",
      "createdDate": "2026-07-13T06:50:36.532741Z",
      "modifiedDate": "2026-07-13T06:50:51.613222Z",
      "taxRate": 0,
      "exchangeFeeRate": 0,
      "feeRate": 0.00027,
      "leaveQuantity": 400,
      "canceledQuantity": 0,
      "error": "",
      "priceSecure": 24250,
      "metadata": "{\"orderSession\":\"OPEN\",\"ip\":\"202.60.110.58\",\"maker\":\"1000005917\",\"isForeigner\":false,\"reqId\":\"24534\",\"probType\":\"CUSTOMER_8000\",\"eventNo\":3}",
      "loanPackageId": 5757
    },
    {
      "id": 1651,
      "side": "NB",
      "accountNo": "0001179019",
      "symbol": "HPG",
      "price": 24250,
      "quantity": 500,
      "orderType": "LO",
      "orderStatus": "PartiallyFilled",
      "fillQuantity": 100,
      "lastQuantity": 100,
      "lastPrice": 24250,
      "averagePrice": 24250,
      "transDate": "2026-07-13",
      "createdDate": "2026-07-13T06:50:36.532741Z",
      "modifiedDate": "2026-07-13T06:50:51.613222Z",
      "taxRate": 0,
      "exchangeFeeRate": 0,
      "feeRate": 0.00027,
      "leaveQuantity": 400,
      "canceledQuantity": 0,
      "error": "",
      "priceSecure": 24250,
      "metadata": "{\"orderSession\":\"OPEN\",\"ip\":\"202.60.110.58\",\"maker\":\"1000005917\",\"isForeigner\":false,\"reqId\":\"24534\",\"probType\":\"CUSTOMER_8000\",\"eventNo\":3}",
      "loanPackageId": 5757
    },
    {
      "id": 1651,
      "side": "NB",
      "accountNo": "0001179019",
      "symbol": "HPG",
      "price": 24250,
      "quantity": 500,
      "orderType": "LO",
      "orderStatus": "PendingNew",
      "fillQuantity": 0,
      "lastQuantity": 0,
      "lastPrice": 0,
      "averagePrice": 0,
      "transDate": "2026-07-13",
      "createdDate": "2026-07-13T06:50:36.532741Z",
      "modifiedDate": "2026-07-13T06:50:36.532743Z",
      "taxRate": 0,
      "exchangeFeeRate": 0,
      "feeRate": 0.00027,
      "leaveQuantity": 500,
      "canceledQuantity": 0,
      "error": "<no value>",
      "priceSecure": 24250,
      "metadata": "{\"orderSession\":\"OPEN\",\"ip\":\"202.60.110.58\",\"maker\":\"1000005917\",\"isForeigner\":false,\"reqId\":\"24534\",\"probType\":\"CUSTOMER_8000\",\"eventNo\":1}",
      "loanPackageId": 5757
    },
    {
      "id": 1651,
      "side": "NB",
      "accountNo": "0001179019",
      "symbol": "HPG",
      "price": 24250,
      "quantity": 500,
      "orderType": "LO",
      "orderStatus": "New",
      "fillQuantity": 0,
      "lastQuantity": 0,
      "lastPrice": 0,
      "averagePrice": 0,
      "transDate": "2026-07-13",
      "createdDate": "2026-07-13T06:50:36.532741Z",
      "modifiedDate": "2026-07-13T06:50:36.545128Z",
      "taxRate": 0,
      "exchangeFeeRate": 0,
      "feeRate": 0.00027,
      "leaveQuantity": 500,
      "canceledQuantity": 0,
      "error": "",
      "priceSecure": 24250,
      "metadata": "{\"orderSession\":\"OPEN\",\"ip\":\"202.60.110.58\",\"maker\":\"1000005917\",\"isForeigner\":false,\"reqId\":\"24534\",\"probType\":\"CUSTOMER_8000\",\"eventNo\":2}",
      "loanPackageId": 5757
    },
    {
      "id": 1651,
      "side": "NB",
      "accountNo": "0001179019",
      "symbol": "HPG",
      "price": 24250,
      "quantity": 500,
      "orderType": "LO",
      "orderStatus": "PartiallyFilled",
      "fillQuantity": 200,
      "lastQuantity": 100,
      "lastPrice": 24250,
      "averagePrice": 24250,
      "transDate": "2026-07-13",
      "createdDate": "2026-07-13T06:50:36.532741Z",
      "modifiedDate": "2026-07-13T06:50:59.052768Z",
      "taxRate": 0,
      "exchangeFeeRate": 0,
      "feeRate": 0.00027,
      "leaveQuantity": 300,
      "canceledQuantity": 0,
      "error": "",
      "priceSecure": 24250,
      "metadata": "{\"orderSession\":\"OPEN\",\"ip\":\"202.60.110.58\",\"maker\":\"1000005917\",\"isForeigner\":false,\"reqId\":\"24534\",\"probType\":\"CUSTOMER_8000\",\"eventNo\":4}",
      "loanPackageId": 5757
    }
  ]
}
```

```json
{
  "id": 26,
  "side": "NB",
  "accountNo": "0001179019",
  "symbol": "41I1G9000",
  "price": 2003,
  "quantity": 2,
  "orderType": "LO",
  "loanPackageId": 2279,
  "orderCategory": "NORMAL",
  "orderStatus": "Filled",
  "fillQuantity": 2,
  "lastQuantity": 2,
  "lastPrice": 2003,
  "averagePrice": 2003,
  "transDate": "2026-07-31",
  "taxRate": 0,
  "exchangeFeeRate": 0,
  "feeRate": 0,
  "leaveQuantity": 0,
  "canceledQuantity": 0,
  "error": "",
  "marketType": "DERIVATIVE",
  "priceSecure": 2003,
  "createdDate": "2026-08-04T02:00:02.855824889Z",
  "modifiedDate": "2026-08-04T02:50:30.56129326Z",
  "metadata": "{\"orderSession\":\"ATO\",\"dealId\":178488852178882,\"releaseSecureOnFilled\":false,\"originMaker\":\"risk-management\",\"dtaAccountNo\":\"D000113702\",\"maker\":\"risk-management\",\"isForeigner\":false,\"probType\":\"CUSTOMER_8000\",\"eventNo\":3}",
  "reports": [
    {
      "id": 26,
      "side": "NB",
      "accountNo": "0001179019",
      "symbol": "41I1G9000",
      "price": 2003,
      "quantity": 2,
      "orderType": "LO",
      "orderStatus": "PendingNew",
      "fillQuantity": 0,
      "lastQuantity": 0,
      "lastPrice": 0,
      "averagePrice": 0,
      "transDate": "2026-07-31",
      "createdDate": "2026-08-04T02:00:02.855824889Z",
      "modifiedDate": "2026-08-04T02:00:02.855825105Z",
      "taxRate": 0,
      "exchangeFeeRate": 0,
      "feeRate": 0,
      "leaveQuantity": 2,
      "canceledQuantity": 0,
      "error": "<no value>",
      "priceSecure": 2003,
      "metadata": "{\"orderSession\":\"ATO\",\"dealId\":178488852178882,\"releaseSecureOnFilled\":false,\"originMaker\":\"risk-management\",\"dtaAccountNo\":\"D000113702\",\"maker\":\"risk-management\",\"isForeigner\":false,\"probType\":\"CUSTOMER_8000\",\"eventNo\":1}",
      "loanPackageId": 2279
    },
    {
      "id": 26,
      "side": "NB",
      "accountNo": "0001179019",
      "symbol": "41I1G9000",
      "price": 2003,
      "quantity": 2,
      "orderType": "LO",
      "orderStatus": "New",
      "fillQuantity": 0,
      "lastQuantity": 0,
      "lastPrice": 0,
      "averagePrice": 0,
      "transDate": "2026-07-31",
      "createdDate": "2026-08-04T02:00:02.855824889Z",
      "modifiedDate": "2026-08-04T02:00:02.938935544Z",
      "taxRate": 0,
      "exchangeFeeRate": 0,
      "feeRate": 0,
      "leaveQuantity": 2,
      "canceledQuantity": 0,
      "error": "",
      "priceSecure": 2003,
      "metadata": "{\"orderSession\":\"ATO\",\"dealId\":178488852178882,\"releaseSecureOnFilled\":false,\"originMaker\":\"risk-management\",\"dtaAccountNo\":\"D000113702\",\"maker\":\"risk-management\",\"isForeigner\":false,\"probType\":\"CUSTOMER_8000\",\"eventNo\":2}",
      "loanPackageId": 2279
    },
    {
      "id": 26,
      "side": "NB",
      "accountNo": "0001179019",
      "symbol": "41I1G9000",
      "price": 2003,
      "quantity": 2,
      "orderType": "LO",
      "orderStatus": "Filled",
      "fillQuantity": 2,
      "lastQuantity": 2,
      "lastPrice": 2003,
      "averagePrice": 2003,
      "transDate": "2026-07-31",
      "createdDate": "2026-08-04T02:00:02.855824889Z",
      "modifiedDate": "2026-08-04T02:50:30.56129326Z",
      "taxRate": 0,
      "exchangeFeeRate": 0,
      "feeRate": 0,
      "leaveQuantity": 0,
      "canceledQuantity": 0,
      "error": "",
      "priceSecure": 2003,
      "metadata": "{\"orderSession\":\"ATO\",\"dealId\":178488852178882,\"releaseSecureOnFilled\":false,\"originMaker\":\"risk-management\",\"dtaAccountNo\":\"D000113702\",\"maker\":\"risk-management\",\"isForeigner\":false,\"probType\":\"CUSTOMER_8000\",\"eventNo\":3}",
      "loanPackageId": 2279
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

<h3 id="getexecutions-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» id|integer(int32)|false|none|ID lệnh|
|» side|string|false|none|Chiều giao dịch<br>- NB: Mua<br>- NS: Bán|
|» accountNo|string|false|none|Số tiểu khoản|
|» symbol|string|false|none|Mã chứng khoán|
|» price|integer(int32)|false|none|Giá đặt lệnh|
|» quantity|integer(int32)|false|none|Khối lượng đặt lệnh|
|» orderType|string|false|none|Loại lệnh<br>- LO: Lệnh giới hạn<br>- MOK/MAK/MTL: Lệnh thị trường<br>- ATO/ATC: Lệnh phiên định kỳ mở cửa/đóng cửa<br>- PLO: Lệnh khớp lệnh sau giờ|
|» loanPackageId|integer(int32)|false|none|Mã gói vay|
|» orderCategory|string|false|none|Phân loại lệnh (mặc định NORMAL)|
|» orderStatus|string|false|none|Trạng thái lệnh<br>- Pending/PendingNew: Chờ gửi<br>- New: Chờ khớp<br>- PartiallyFilled: Khớp một phần<br>- Filled: Khớp toàn bộ<br>- Rejected: Bị từ chối<br>- Expired: Hết hạn trong phiên<br>- DoneForDay: Lệnh được giải tỏa do không khớp trong phiên|
|» fillQuantity|integer(int32)|false|none|Khối lượng đã khớp|
|» lastQuantity|integer(int32)|false|none|Khối lượng khớp gần nhất|
|» lastPrice|integer(int32)|false|none|Giá khớp gần nhất|
|» averagePrice|integer(int32)|false|none|Giá khớp trung bình|
|» transDate|string|false|none|Ngày giao dịch|
|» taxRate|integer(int32)|false|none|Tỷ lệ thuế|
|» exchangeFeeRate|number(double)|false|none|Tỷ lệ phí trả Sở giao dịch|
|» feeRate|number(double)|false|none|Tổng tỷ lệ phí của lệnh|
|» leaveQuantity|integer(int32)|false|none|Khối lượng còn lại chưa khớp|
|» canceledQuantity|integer(int32)|false|none|Khối lượng đã hủy|
|» error|string|false|none|Mã lỗi nếu lệnh bị từ chối|
|» marketType|string|false|none|Loại thị trường<br>- STOCK: Lệnh cơ sở<br>- DERIVATIVE: Lệnh phái sinh|
|» priceSecure|integer(int32)|false|none|Giá dùng để kiểm tra sức mua/đặt lệnh|
|» createdDate|string(date-time)|false|none|Thời điểm tạo lệnh|
|» modifiedDate|string(date-time)|false|none|Thời điểm cập nhật lệnh|
|» metadata|string|false|none|Thông tin bổ sung của lệnh|
|» reports|[object]|false|none|Danh sách trạng thái lệnh theo từng lần cập nhật|
|»» id|integer(int32)|false|none|ID lệnh|
|»» side|string|false|none|Chiều giao dịch<br>- NB: Mua<br>- NS: Bán|
|»» accountNo|string|false|none|Số tiểu khoản|
|»» symbol|string|false|none|Mã chứng khoán|
|»» price|integer(int32)|false|none|Giá đặt lệnh|
|»» quantity|integer(int32)|false|none|Khối lượng đặt lệnh|
|»» orderType|string|false|none|Loại lệnh<br>- LO: Lệnh giới hạn<br>- MOK/MAK/MTL: Lệnh thị trường<br>- ATO/ATC: Lệnh phiên định kỳ mở cửa/đóng cửa<br>- PLO: Lệnh khớp lệnh sau giờ|
|»» orderStatus|string|false|none|Trạng thái lệnh<br>- Pending/PendingNew: Chờ gửi<br>- New: Chờ khớp<br>- PartiallyFilled: Khớp một phần<br>- Filled: Khớp toàn bộ<br>- Rejected: Bị từ chối<br>- Expired: Hết hạn trong phiên<br>- DoneForDay: Lệnh được giải tỏa do không khớp trong phiên|
|»» fillQuantity|integer(int32)|false|none|Khối lượng đã khớp|
|»» lastQuantity|integer(int32)|false|none|Khối lượng khớp gần nhất|
|»» lastPrice|integer(int32)|false|none|Giá khớp gần nhất|
|»» averagePrice|integer(int32)|false|none|Giá khớp trung bình|
|»» transDate|string|false|none|Ngày giao dịch|
|»» createdDate|string(date-time)|false|none|Thời điểm tạo lệnh|
|»» modifiedDate|string(date-time)|false|none|Thời điểm cập nhật lệnh|
|»» taxRate|integer(int32)|false|none|Tỷ lệ thuế|
|»» exchangeFeeRate|integer(int32)|false|none|Tỷ lệ phí trả Sở giao dịch|
|»» feeRate|number(double)|false|none|Tổng tỷ lệ phí của lệnh|
|»» leaveQuantity|integer(int32)|false|none|Khối lượng còn lại chưa khớp|
|»» canceledQuantity|integer(int32)|false|none|Khối lượng đã hủy|
|»» error|string|false|none|Mã lỗi nếu lệnh bị từ chối|
|»» priceSecure|integer(int32)|false|none|Giá dùng để kiểm tra sức mua/đặt lệnh|
|»» metadata|string|false|none|Thông tin bổ sung của lệnh|
|»» loanPackageId|integer(int32)|false|none|ID gói vay|

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
