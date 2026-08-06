## Lịch sử sự kiện quyền

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getCorporateActionHistory"></span>

### `GET /accounts/{accountNo}/corporate-action-history`

Lấy danh sách lịch sử sự kiện quyền của tài khoản chứng khoán.

<h3 id="getcorporateactionhistory-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|symbol|query|string|false|Mã chứng khoán|
|caType|query|string|false|Phân loại sự kiện quyền |
|caStatus|query|string|false|Trạng thái xử lý sự kiện quyền |
|pageIndex|query|integer|false|Trang hiện tại|
|pageSize|query|integer|false|Số lượng bản ghi trên mỗi trang|
|X-API-Key|header|string|false|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|false|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|false|Chữ ký xác thực yêu cầu|
|version|header|string|false|API version (YYYY-MM-DD)|
|accountNo|path|string|true|Số tiểu khoản|

#### Detailed descriptions

**caType**: Phân loại sự kiện quyền 
- cashDividend: Sự kiện quyền trả cổ tức bằng tiền
- stockDividend: Sự kiện quyền trả cổ tức cổ phiếu
- stockBonus: Sự kiện quyền trả cổ phiếu thưởng 
- rightsOffering: Sự kiện quyền mua cổ phiếu phát hành thêm

**caStatus**: Trạng thái xử lý sự kiện quyền 
- pending: Quyền đang chờ xử lý hoặc đang được thực hiện
- completed: Quyền đã được xử lý hoàn tất
- canceled: Quyền đã hủy thành công

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/accounts/{accountNo}/corporate-action-history \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/accounts/{accountNo}/corporate-action-history HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/accounts/{accountNo}/corporate-action-history", data)
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

fetch('https://openapi.dnse.com.vn/accounts/{accountNo}/corporate-action-history',
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

r = requests.get('https://openapi.dnse.com.vn/accounts/{accountNo}/corporate-action-history', headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/accounts/{accountNo}/corporate-action-history");
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
  "accountNo": "0001179019",
  "data": {
    "cashDividend": [
      {
        "id": 617210,
        "symbol": "MSB",
        "caStatus": "pending",
        "recordDate": "2026-04-24",
        "processDate": "2026-05-13",
        "holdingQuantity": 600,
        "dividendValue": 500,
        "grossAmount": 300000,
        "taxAmount": 15000,
        "netAmount": 285000
      }
    ],
    "stockDividend": [
      {
        "id": 618949,
        "symbol": "OIL",
        "caStatus": "pending",
        "recordDate": "2026-04-29",
        "processDate": "2026-05-14",
        "holdingQuantity": 3211,
        "ratio": "5/3",
        "receivedQuantity": 1926
      }
    ],
    "stockBonus": [
      {
        "id": 617276,
        "symbol": "MSB",
        "caStatus": "completed",
        "recordDate": "2026-04-24",
        "processDate": "2026-05-13",
        "holdingQuantity": 600,
        "ratio": "15/6",
        "receivedQuantity": 240
      }
    ],
    "rightsOffering": [
      {
        "id": 623428,
        "symbol": "OIL",
        "caStatus": "pending",
        "recordDate": "2026-04-29",
        "processDate": "2026-05-14",
        "holdingQuantity": 5137,
        "ratio": "3/1",
        "rightPrice": 11000,
        "rightsQuantity": 1712,
        "registeredQuantity": 170,
        "startDateTransfer": "2026-05-04",
        "endDateTransfer": "2026-05-13",
        "startDateSubscription": "2026-05-04",
        "endDateSubscription": "2026-05-13"
      }
    ]
  },
  "pagination": {
    "pageIndex": 0,
    "pageSize": 100,
    "totalRecords": 100
  }
}
```

<h3 id="getcorporateactionhistory-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» accountNo|string|false|none|Số tiểu khoản|
|» data|object|false|none|Danh sách quyền chứng khoán của tiểu khoản, được phân loại theo từng loại quyền|
|»» cashDividend|[object]|false|none|Danh sách quyền trả cổ tức bằng tiền|
|»»» id|integer(int32)|false|none|ID sự kiện quyền|
|»»» symbol|string|false|none|Mã chứng khoán|
|»»» caStatus|string|false|none|Trạng thái xử lý sự kiện quyền<br>- pending: Quyền đang chờ xử lý hoặc đang được thực hiện<br>- completed: Quyền đã được xử lý hoàn tất<br>- canceled: Quyền đã hủy thành công|
|»»» recordDate|string|false|none|Ngày đăng ký cuối cùng sự kiện quyền|
|»»» processDate|string|false|none|Ngày thanh toán dự kiến|
|»»» holdingQuantity|integer(int32)|false|none|Số lượng chứng khoán sở hữu tại ngày chốt quyền|
|»»» dividendValue|integer(int32)|false|none|Giá trị cổ tức trên mỗi cổ phiếu|
|»»» grossAmount|integer(int32)|false|none|Tổng giá trị tiền cổ tức trước thuế|
|»»» taxAmount|integer(int32)|false|none|Số tiền thuế bị khấu trừ|
|»»» netAmount|integer(int32)|false|none|Số tiền thực nhận sau thuế|
|»» stockDividend|[object]|false|none|Danh sách quyền trả cổ tức bằng cổ phiếu|
|»»» id|integer(int32)|false|none|ID sự kiện quyền|
|»»» symbol|string|false|none|Mã chứng khoán|
|»»» caStatus|string|false|none|Trạng thái xử lý sự kiện quyền<br>- pending: Quyền đang chờ xử lý hoặc đang được thực hiện<br>- completed: Quyền đã được xử lý hoàn tất<br>- canceled: Quyền đã hủy thành công|
|»»» recordDate|string|false|none|Ngày đăng ký cuối cùng sự kiện quyền|
|»»» processDate|string|false|none|Ngày phân phối cổ phiếu cổ tức dự kiến|
|»»» holdingQuantity|integer(int32)|false|none|Số lượng chứng khoán sở hữu tại ngày chốt quyền|
|»»» ratio|string|false|none|Tỷ lệ thực hiện quyền nhận cổ tức bằng cổ phiếu|
|»»» receivedQuantity|integer(int32)|false|none|Số lượng cổ phiếu cổ tức được nhận|
|»» stockBonus|[object]|false|none|Danh sách quyền trả cổ phiếu thưởng|
|»»» id|integer(int32)|false|none|ID sự kiện quyền|
|»»» symbol|string|false|none|Mã chứng khoán|
|»»» caStatus|string|false|none|Trạng thái xử lý sự kiện quyền<br>- pending: Quyền đang chờ xử lý hoặc đang được thực hiện<br>- completed: Quyền đã được xử lý hoàn tất<br>- canceled: Quyền đã hủy thành công|
|»»» recordDate|string|false|none|Ngày đăng ký cuối cùng sự kiện quyền|
|»»» processDate|string|false|none|Ngày phân phối cổ phiếu thưởng dự kiến|
|»»» holdingQuantity|integer(int32)|false|none|Số lượng chứng khoán sở hữu tại ngày chốt quyền|
|»»» ratio|string|false|none|Tỷ lệ thực hiện quyền nhận cổ phiếu thưởng|
|»»» receivedQuantity|integer(int32)|false|none|Số lượng cổ phiếu thưởng khách hàng được nhận|
|»» rightsOffering|[object]|false|none|Danh sách quyền mua cổ phiếu phát hành thêm|
|»»» id|integer(int32)|false|none|ID sự kiện quyền|
|»»» symbol|string|false|none|Mã chứng khoán|
|»»» caStatus|string|false|none|Trạng thái xử lý sự kiện quyền<br>- pending: Quyền đang chờ xử lý hoặc đang được thực hiện<br>- completed: Quyền đã được xử lý hoàn tất<br>- canceled: Quyền đã hủy thành công|
|»»» recordDate|string|false|none|Ngày đăng ký cuối cùng sự kiện quyền để xác định quyền mua|
|»»» processDate|string|false|none|Ngày phân phối cổ phiếu quyền mua dự kiến|
|»»» holdingQuantity|integer(int32)|false|none|Số lượng chứng khoán sở hữu tại ngày chốt quyền|
|»»» ratio|string|false|none|Tỷ lệ thực hiện quyền mua cổ phiếu|
|»»» rightPrice|integer(int32)|false|none|Giá đăng ký mua cổ phiếu phát hành thêm|
|»»» rightsQuantity|integer(int32)|false|none|Số lượng quyền mua được phân bổ trên tài khoản|
|»»» registeredQuantity|integer(int32)|false|none|Số lượng cổ phiếu phát hành thêm đã đăng ký mua|
|»»» startDateTransfer|string|false|none|Ngày bắt đầu được phép chuyển nhượng quyền mua|
|»»» endDateTransfer|string|false|none|Ngày kết thúc chuyển nhượng quyền mua|
|»»» startDateSubscription|string|false|none|Ngày bắt đầu đăng ký thực hiện quyền mua|
|»»» endDateSubscription|string|false|none|Ngày kết thúc đăng ký thực hiện quyền mua|
|» pagination|object|false|none|Thông tin phân trang dữ liệu|
|»» pageIndex|integer(int32)|false|none|Trang hiện tại|
|»» pageSize|integer(int32)|false|none|Số lượng bản ghi trên mỗi trang|
|»» totalRecords|integer(int32)|false|none|Tổng số bản ghi|

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
