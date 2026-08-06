---
sidebar_position: 3
---


# Xác thực
---
## Xác thực ứng dụng

Trong giao tiếp giữa ứng dụng của Người dùng và hệ thống DNSE, việc xác minh danh tính và bảo vệ tính toàn vẹn của dữ liệu là yêu cầu bắt buộc. Mọi RESTful API gửi đến DNSE đều bắt buộc kèm theo 3 thông tin cốt lõi trong Headers: X-Api-Key, X-Signature và Date (hoặc tên Header cấu hình riêng).

### API Key
- API Key là khóa định danh duy nhất được cấp cho từng tài khoản khi đăng ký sử dụng LightSpeed API.
- Trong mỗi request, API Key là thành phần bắt buộc để nhận diện ứng dụng và áp dụng cơ chế phân quyền, giới hạn truy cập.
- API Key cần được quản lý cẩn trọng và tránh chia sẻ công khai. Người dùng có thể chủ động tạo mới hoặc thu hồi API Key trong trường hợp nghi ngờ lộ thông tin hoặc không còn nhu cầu sử dụng.
- Trường hợp bị rò rỉ, các yêu cầu trái phép sẽ không được chấp nhận nếu không có Signature hợp lệ đi kèm.

### Signature (Chữ ký số)

- Hệ thống OpenAPI sử dụng chuẩn HTTP Signature để chứng thực tính toàn vẹn của dữ liệu trên đường truyền. Mỗi Request hợp lệ gửi lên hệ thống bắt buộc phải đính kèm Header X-Signature.

  Cấu trúc Signature:  **HMAC‑SHA256 (API Secret, method + path + date header + nonce) → Base64 URL‑encode**

<details>
  <summary>Quy trình 3 bước tạo Signature</summary>

**Bước 1 — Xây dựng Signing String**
Ghép các thông tin của Header request thành một chuỗi ký tự theo đúng ký tự và định dạng, không phụ thuộc Body Request.

:::info[Cấu trúc Signing String]

```http
(request-target): get /accounts     // method + path
date: Fri, 15 May 2026 07:11:30 +0000     // giá trị header Date theo chuẩn RFC1123
nonce: c9a8f88b472c9721fde161e0d89df8cc   
```
:::

- Mỗi dòng kết thúc bằng ký tự xuống dòng `\n`, trừ dòng cuối cùng
- Phân biệt chữ hoa/chữ thường: method viết thường (get, post...), tên field (date, nonce) viết thường
- `nonce` là chuỗi UUID4 hex 32 ký tự không có dấu gạch ngang, sinh mới cho mỗi request chống tấn công replay (dùng lại request cũ)

**Bước 2 — Tạo chuỗi Signature (ENCODED_SIGNATURE)**

Áp dụng lần lượt 4 phép biến đổi sau lên Signing String vừa tạo:

- Ký số chuỗi Signing String bằng thuật toán HMAC-SHA256 với API Secret (Định dạng UTF-8) để thu được chuỗi raw bytes.
- Mã hóa raw bytes đó sang định dạng Base64 encode.
- URL-encode: Chỉ mã hóa 3 ký tự đặc biệt có thể xuất hiện trong Base64 là `+` → `%2B`, / → `%2F`, và `=` → `%3D`. Không mã hóa toàn bộ chuỗi, không double-encode.
- Kết quả thu được chuỗi ký tự cuối cùng ký hiệu `ENCODED_SIGNATURE`

**Bước 3 — Đóng gói Header X-Signature**

:::info[X-Signature]
``` text
Signature keyId="{API_KEY}",algorithm="hmac-sha256",headers="(request-target) date",signature="{ENCODED_SIGNATURE}",nonce="{NONCE}"
```
:::

Ví dụ Headers hoàn chỉnh (Định dạng Raw HTTP)

```http
x-api-key: lB58g6iWzyrNx2EhwwQXeYeoAnkzlaXkJWi  // APIKey được cấp khi đăng ký dịch vụ
x-Signature: Signature keyId="lB58g6iWzyrNx2EhwwQXeYeoAnkzlaXkJWi",algorithm="hmac-sha256",headers="(request-target) date",signature="U7NOnhIlAlsWJviOqtlRZajLmZmbq0Bb2T1EVsHm3%2Bg%3D",nonce="26c4b530cf12427d95bf691e39aa8d74"  // Chữ ký số theo thuật toán HMAC SHA256
Date: Fri, 15 May 2026 07:11:30 +0000     // Thời gian tạo yêu cầu (UTC)
version: 2026-05-07    // Phiên bản API
```
</details>

:::warning[Lưu ý quan trọng]

- Giới hạn thời gian: Để chống replay attack, giá trị Header Date phải chính xác và không được lệch quá ±1 phút so với giờ chuẩn của hệ thống DNSE.
- Mỗi request phải có Date và Nonce mới — Signature gắn liền với giá trị Date và nonce tại thời điểm tạo. Không được tái sử dụng Signature cũ.
- Sai thứ tự dòng trong Signing String → Signature sai dẫn đến Request bị từ chối ngay lập tức.
- Kiểm tra kỹ URL-encode, chỉ mã hóa `+`, `/`, `=` từ chuỗi Base64 gốc.
- Để không cần tự implement thuật toán và giảm thiểu lỗi, DNSE cung cấp SDK tự động sinh Signature cho mỗi request. Tham khảo tại [GitHub DNSE OpenAPI](https://github.com/dnse-tech/openapi-sdk)

:::

### Lỗi có thể gặp

Nếu thông tin xác thực không hợp lệ (ví dụ sai `Signature` hoặc `Date`), API sẽ trả về lỗi:

```json lines
{
  "status": "error",
  "code": "OA-400",
  "message": "Authorization field missing, malformed or invalid"
}
```

**Nguyên nhân thường gặp:**

- `Signature` được tạo không chính xác (sai thuật toán, sai chuỗi ký hoặc sử dụng sai API Secret).
- Giá trị `Date` không đúng định dạng hoặc không khớp với giá trị dùng để tạo Signature.
- Authorization header bị thiếu hoặc sai định dạng.

**Cách khắc phục:**

- Kiểm tra lại chuỗi dữ liệu dùng để tạo Signature.
- Đảm bảo Signature được ký bằng đúng API Secret.
- Đảm bảo giá trị `Date` trong header giống hoàn toàn với giá trị đã sử dụng khi tạo Signature.
- Kiểm tra định dạng và nội dung của header Authorization.

----

## Xác thực giao dịch

Nếu API Key đóng vai trò là lớp bảo mật thứ nhất, thì Trading Token là lớp bảo mật thứ 2 đối với giao dịch đặt lệnh theo cơ chế 2FA – Two Factor Authentication.

Trading Token là mã có thời hạn 8 tiếng, được cung cấp sau khi người dùng hoàn tất xác thực OTP, và là thông tin bắt buộc phải được truyền kèm trong các API đặt lệnh. Trong thời gian token còn hiệu lực, người dùng có thể liên tục đặt lệnh mà không cần cấu hình lại OTP cho từng Request.

<details>
  <summary>Quy trình lấy và sử dụng Trading Token</summary>

**Bước 1: Yêu cầu gửi mã OTP**
- Tùy theo phương thức xác thực người dùng đã đăng ký, thực hiện lấy mã OTP:

    - Smart OTP: Tại app EntradeX by DNSE trên thiết bị di động, người dùng chọn mục SmartOTP từ menu, chọn Lấy mã OTP cho thiết bị khác. (Hiệu lực 30 giây)

        <div className="guideImg">

      [![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/otp.png)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/otp.png)

        </div>
    - Email OTP: Gọi Endpoint <a href="https://developers.dnse.com.vn/docs/dnse/send-email-otp">Gửi Email OTP</a> để nhận mã. (Hiệu lực 2 phút)

- Tại một thời điểm, tài khoản chỉ được đăng ký sử dụng duy nhất 1 phương thức và cần truyền đúng `otpType` tương ứng. Trường hợp đăng ký sử dụng Smart OTP nhưng truyền lên `email-otp` (hoặc ngược lại) yêu cầu gửi đến hệ thống sẽ bị từ chối.

  ```json lines
  {
    "status":400,
    "code":"OA-100",
    "message":"Invalid input: email otp is not registered for this account"
  }
  ```

**Bước 2: Xác thực OTP lấy Trading Token**

- Để lấy Trading Token, người dùng thực hiện gửi yêu cầu đến Endpoint <a href="https://developers.dnse.com.vn/docs/dnse/2-fa-verification">xác thực OTP</a>. Hệ thống chỉ chấp nhận đúng loại OTP đã cấu hình trên tài khoản của người dùng.

  Ví dụ Request:
  ```json lines
  {
    "method": "POST",
    "path": "/openapi/registration/trading-token",
    "headers": {
      "x-api-key": "lB58g6iWzyrNx2EhwwQXeYeoAnkzlaXkJWi",  // APIKey được cấp khi đăng ký dịch vụ
      "x-Signature": 'Signature keyId="lB58g6iWzyrNx2EhwwQXeYeoAnkzlaXkJWi",algorithm="hmac-sha256",headers="(request-target) date",signature="U7NOnhIlAlsWJviOqtlRZajLmZmbq0Bb2T1EVsHm3%2Bg%3D",nonce="26c4b530cf12427d95bf691e39aa8d74"',  // Chữ ký số theo thuật toán HMAC SHA256
      "Date":"Fri, 15 May 2026 07:11:30 +0000",  // Thời gian tạo yêu cầu (UTC)
      "version":"2026-05-07", // Phiên bản API
      "Content-Type": "application/json"
    }, 
    "body": {
      "otpType": "email_otp",   // Phương thức OTP đang sử dụng (email_otp hoặc smart_otp)
      "passcode": "1234"     // Mã OTP tương ứng phương thức
    }
  }
  ```

- Nếu thông tin hợp lệ, hệ thống sẽ trả về Trading Token có hiệu lực trong 8 giờ, hãy lưu chuỗi này vào bộ nhớ ứng dụng

  ```json lines
  {"trading-token": "7ceef658-9f01-414e-8b3e-faa77bb9061e"}    // Token đặt lệnh
  ```  

**Bước 3: Đính kèm Token trong API đặt lệnh**
- Sau khi có Trading Token, người dùng cần truyền vào Header cho các Request thực hiện giao dịch lệnh. Tham khảo <a href="https://developers.dnse.com.vn/docs/dnse/place-order">Endpoint Đặt lệnh</a>.
- Khi Trading Token hết hiệu lực sau 8 tiếng, người dùng cần thực hiện lại bước xác thực OTP tạo token mới, tránh bị gián đoạn giao dịch.

</details>

### Phương thức OTP

OpenAPI hiện hỗ trợ hai phương thức Email OTP hoặc Smart OTP để xác thực và tạo Trading Token. Tại mỗi thời điểm, chỉ một phương thức OTP duy nhất được hoạt động.

#### Email OTP

- Mã OTP được gửi về địa chỉ email mà người dùng đã đăng ký, có hiệu lực trong 2 phút.
- Ưu điểm:
    - Quản lý linh hoạt, có thể tự động hóa trong quy trình xác thực, mang lại trải nghiệm liền mạch cho người dùng khi xây dựng hệ thống giao dịch qua OpenAPI.
- Hạn chế:
    - Thời gian nhận email phụ thuộc vào bên thứ 3.

#### Smart OTP

- Mã OTP được lấy trực tiếp trên ứng dụng DNSE đã đăng ký SmartOTP, có hiệu lực trong 30 giây.
- Ưu điểm:
    - Mã luôn có sẵn trên ứng dụng và chỉ sinh trên thiết bị đã đăng ký.
    - Độ bảo mật cao do, giảm thiểu nguy cơ giả mạo hoặc truy cập trái phép.
- Hạn chế:
    - Người dùng cần thao tác thủ công vào ứng dụng để lấy mã khi cần thực hiện xác thực.
