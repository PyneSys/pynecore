## Xác thực OTP

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="2FAVerification"></span>

### `POST /registration/trading-token`

Xác thực mã OTP theo phương thức xác thực (Email OTP hoặc Smart OTP) đã đăng ký để lấy Trading Token. Đây là thông tin bắt buộc để xác thực quyền giao dịch, có hiệu lực trong 8 tiếng.

<h3 id="2faverification-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|X-API-Key|header|string|true|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|true|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|true|Chữ ký xác thực yêu cầu|
|version|header|string(date)|false|API version (YYYY-MM-DD)|
|body|body|object|false|none|
|» otpType|body|string|false|Phương thức xác thực OTP|
|» passcode|body|string|false|Mã OTP tương ứng với phương thức xác thực|

#### Detailed descriptions

**» otpType**: Phương thức xác thực OTP
- email_otp: Áp dụng cho tài khoản đăng ký phương thức xác thực lớp thứ 2 là Email OTP
- smart_otp: Áp dụng cho tài khoản đăng ký phương thức xác thực lớp thứ 2 là Smart OTP

> Code samples

```shell
# You can also use wget
curl -X POST https://openapi.dnse.com.vn/registration/trading-token \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -H 'X-API-Key: eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==' \
  -H 'X-Aux-Date: Mon, 19 Jan 2026 07:45:23 +0000' \
  -H 'X-Signature: your_signature' \
  -H 'version: 2026-07-23'

```

```http
POST https://openapi.dnse.com.vn/registration/trading-token HTTP/1.1
Host: openapi.dnse.com.vn
Content-Type: application/json
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
        "Content-Type": []string{"application/json"},
        "Accept": []string{"application/json"},
        "X-API-Key": []string{"eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ=="},
        "X-Aux-Date": []string{"Mon, 19 Jan 2026 07:45:23 +0000"},
        "X-Signature": []string{"your_signature"},
        "version": []string{"2026-07-23"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("POST", "https://openapi.dnse.com.vn/registration/trading-token", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

```javascript
const inputBody = '{
  "otpType": "email_otp",
  "passcode": "519752"
}';
const headers = {
  'Content-Type':'application/json',
  'Accept':'application/json',
  'X-API-Key':'eyJvcmciOiJkbnNlIiwiaWQiOiI5YmMzYmViN2JjY2U0MmE0Yjk1NDE0MTA2YTMzODIxNyIsImgiOiJtdXJtdXIxMjgifQ==',
  'X-Aux-Date':'Mon, 19 Jan 2026 07:45:23 +0000',
  'X-Signature':'your_signature',
  'version':'2026-07-23'
};

fetch('https://openapi.dnse.com.vn/registration/trading-token',
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
  'X-Aux-Date': 'Mon, 19 Jan 2026 07:45:23 +0000',
  'X-Signature': 'your_signature',
  'version': '2026-07-23'
}

r = requests.post('https://openapi.dnse.com.vn/registration/trading-token', headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/registration/trading-token");
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
  "otpType": "email_otp",
  "passcode": "519752"
}
```

> Example responses

> 200 Response

```json
{
  "tradingToken": "9ab81e9c-8a81-45aa-8190-b69a1c179d52"
}
```

<h3 id="2faverification-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» tradingToken|string|false|none|Token đặt lệnh|

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
