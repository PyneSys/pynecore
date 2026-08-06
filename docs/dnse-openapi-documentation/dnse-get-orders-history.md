## Lịch sử lệnh

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getOrdersHistory"></span>

### `GET /accounts/{accountNo}/orders/history`

Lấy danh sách lệnh thường đã đặt trong một khoảng thời gian nhất định. Thời gian tra cứu tối đa trong vòng 1 năm kể từ ngày hiện tại.

<h3 id="getordershistory-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|string|true|Loại thị trường |
|from|query|string|true|Ngày bắt đầu (yyyy-mm-dd) |
|to|query|string|true|Ngày kết thúc (yyyy-mm-dd) |
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|version|header|string|false|API version (YYYY-MM-DD)|
|accountNo|path|string|true|Số tiểu khoản|

#### Detailed descriptions

**marketType**: Loại thị trường 
- STOCK: Gói vay giao dịch cơ sở
- DERIVATIVE: Gói vay giao dịch phái sinh

**from**: Ngày bắt đầu (yyyy-mm-dd) 
- Thời gian tra cứu tối đa trong 1 năm tính từ ngày hiện tại

**to**: Ngày kết thúc (yyyy-mm-dd) 
- Lớn hơn hoặc bằng ngày bắt đầu và không vượt quá ngày hiện tại

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/accounts/{accountNo}/orders/history?marketType=STOCK&from=2026-02-15&to=2026-03-18 \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
GET https://openapi.dnse.com.vn/accounts/{accountNo}/orders/history?marketType=STOCK&from=2026-02-15&to=2026-03-18 HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/accounts/{accountNo}/orders/history", data)
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

fetch('https://openapi.dnse.com.vn/accounts/{accountNo}/orders/history?marketType=STOCK&from=2026-02-15&to=2026-03-18',
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

r = requests.get('https://openapi.dnse.com.vn/accounts/{accountNo}/orders/history', params={
  'marketType': 'STOCK',  'from': '2026-02-15',  'to': '2026-03-18'
}, headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/accounts/{accountNo}/orders/history?marketType=STOCK&from=2026-02-15&to=2026-03-18");
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
  "fillQuantity": 0,
  "total": 8,
  "start": 0,
  "end": 8,
  "marketType": "STOCK",
  "data": [
    {
      "id": "20260312_241",
      "symbol": "HPG",
      "side": "NB",
      "orderType": "LO",
      "orderStatus": "Expired",
      "price": 27200,
      "quantity": 200,
      "fillQuantity": 0,
      "leaveQuantity": 0,
      "canceledQuantity": 200,
      "averagePrice": 0,
      "loanPackageId": 5765,
      "transDate": "2026-03-12",
      "createdDate": "2026-03-17T06:44:18.813804Z",
      "modifiedDate": "2026-03-18T15:28:24.44026Z"
    }
  ]
}
```

<h3 id="getordershistory-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» accountNo|string|false|none|Số tiểu khoản|
|» fillQuantity|integer(int32)|false|none|Tổng khối lượng đã khớp|
|» total|integer(int32)|false|none|Tổng số bản ghi|
|» start|integer(int32)|false|none|Vị trí bắt đầu của tập bản ghi được trả về|
|» end|integer(int32)|false|none|Vị trí kết thúc của tập bản ghi được trả về|
|» marketType|string|false|none|Loại thị trường<br>- STOCK: Gói vay giao dịch cơ sở<br>- DERIVATIVE: Gói vay giao dịch phái sinh|
|» data|[object]|false|none|Danh sách lệnh giao dịch|
|»» id|string|false|none|ID lệnh trên hệ thống|
|»» symbol|string|false|none|Mã chứng khoán|
|»» side|string|false|none|Chiều giao dịch<br>NB: Mua<br>NS: Bán|
|»» orderType|string|false|none|Loại lệnh<br>- LO: Lệnh giới hạn<br>- MOK/MAK/MTL: Lệnh thị trường<br>- ATO/ATC: Lệnh phiên định kỳ mở cửa/đóng cửa<br>- PLO: Lệnh khớp lệnh sau giờ|
|»» orderStatus|string|false|none|Trạng thái lệnh<br>- Pending/PendingNew: Chờ gửi<br>- New: Chờ khớp<br>- PartiallyFilled: Khớp một phần<br>- Filled: Khớp toàn bộ<br>- Rejected: Bị từ chối<br>- Expired: Hết hạn trong phiên<br>- DoneForDay: Lệnh được giải tỏa do không khớp trong phiên|
|»» price|integer(int32)|false|none|Giá đặt|
|»» quantity|integer(int32)|false|none|Khối lượng đặt|
|»» fillQuantity|integer(int32)|false|none|Khối lượng đã khớp|
|»» leaveQuantity|integer(int32)|false|none|Khối lượng còn lại|
|»» canceledQuantity|integer(int32)|false|none|Khối lượng đã huỷ|
|»» averagePrice|integer(int32)|false|none|Giá khớp trung bình|
|»» loanPackageId|integer(int32)|false|none|ID gói vay|
|»» transDate|string|false|none|Ngày giao dịch|
|»» createdDate|string(date-time)|false|none|Thời điểm tạo|
|»» modifiedDate|string(date-time)|false|none|Thời điểm cập nhật|

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
