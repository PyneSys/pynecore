
# Câu hỏi thường gặp (FAQ)

## Mục lục

1. [Đăng ký & Quản lý dịch vụ](#đăng-ký--quản-lý-dịch-vụ)
2. [Xác thực](#xác-thực)
3. [Trading Token (2FA)](#trading-token-2fa)
4. [API Versioning](#api-versioning)
5. [Rate Limit](#rate-limit)
6. [Giao dịch](#giao-dịch)
7. [WebSocket](#websocket)
8. [Broker & SACO](#broker--saco)
9. [Tài liệu & Tài nguyên](#tài-liệu--tài-nguyên)

---

## Đăng ký & Quản lý dịch vụ

#### 1. Làm thế nào để đăng ký sử dụng LightSpeed API?

Thực hiện 3 bước tại [Trang giao dịch chính thức DNSE](https://entradex.dnse.com.vn):
1. Đăng nhập tài khoản chứng khoán DNSE → vào **Thông tin tài khoản** → chọn **LightSpeed API** → chọn **Đăng ký**.
2. Chọn phương thức xác thực lớp 2: Smart OTP hoặc Email OTP.
3. Xác nhận và hoàn tất OTP.

Sau khi đăng ký thành công, hệ thống hiển thị **API Key** và **API Secret**. API Secret chỉ hiển thị **một lần duy nhất** — người dùng cần lưu lại ngay.

#### 2. Điều kiện để đăng ký LightSpeed API là gì?

Bạn cần có tài khoản chứng khoán tại DNSE với trạng thái **đang hoạt động** (ACTIVE). Nếu chưa có, cần mở tài khoản tại [Trang giao dịch chính thức DNSE](https://entradex.dnse.com.vn) trước khi đăng ký dịch vụ Lightspeed API. 

#### 3. Tôi quên lưu API Secret, tôi có thể xem lại ở đâu?

API Secret không thể xem lại sau khi đóng màn hình đăng ký thành công. Nếu bỏ lỡ, người dùng cần vào **LightSpeed API** → chọn **Tạo lại API Key**. Bộ khóa cũ bị vô hiệu hóa ngay lập tức, bộ key mới được sinh ra và hãy lưu lại ngay bộ key này.

#### 4. Tạo lại API Key có ảnh hưởng đến cài đặt OTP không?

Không. Tạo lại API Key chỉ thay đổi bộ khóa, không ảnh hưởng đến phương thức OTP (Smart OTP / Email OTP) đang sử dụng.

#### 5. Hủy API Key thì có đăng ký lại được không?

Hủy API Key đồng nghĩa với hủy toàn bộ dịch vụ OpenAPI. Nếu muốn dùng lại, người dùng phải **đăng ký mới từ đầu**. Thao tác này khác với Tạo lại API Key (chỉ đổi bộ khóa, giữ nguyên dịch vụ).

#### 6. Smart OTP và Email OTP khác nhau như thế nào?**

|              | Smart OTP                 | Email OTP               |
|--------------|---------------------------|-------------------------|
| Nguồn lấy mã | App DNSE trên điện thoại  | Email đã đăng ký        |
| Hiệu lực     | 30 giây                   | 2 phút                  |
| Tự động hóa  | Không (thao tác thủ công) | Có thể tích hợp tự động |
| Bảo mật      | Cao hơn                   | Phụ thuộc bên thứ 3     |

Tại mỗi thời điểm chỉ sử dụng được **1 phương thức**. Có thể đổi bất kỳ lúc nào qua Entrade X mà không ảnh hưởng API Key. Lưu ý phương thức này chỉ sử dụng cho giao dịch qua OpenAPI. 

#### 7. Tôi có thể dùng tối đa mấy địa chỉ Email nhận OTP?

Người dùng được đăng ký tối đa **2 địa chỉ email**. Để thêm email thứ 2: Vào **LightSpeed API → Thêm Email thứ 2**. Email thứ 2 cần xác thực qua link gửi về email thứ 1 để kích hoạt.

Các thông tin khác về Đăng ký và Quản lý dịch vụ, người dùng tham khảo <a href="https://developers.dnse.com.vn/docs/guide/intro/register_guide">**tại đây**</a>

---

## Xác thực

#### 8. Mỗi REST API request cần truyền những Header nào?

Mọi REST API request đến DNSE bắt buộc có 3 header:

| Header        | Mô tả                                    |
|---------------|------------------------------------------|
| `x-api-key`   | API Key được cấp khi đăng ký             |
| `x-Signature` | Chữ ký số HMAC-SHA256                    |
| `Date`        | Thời gian tạo request theo RFC1123 (UTC) |

Header `version` là tùy chọn nhưng được khuyến nghị truyền tường minh.

#### 9. Signature được tạo như thế nào?

Quy trình 3 bước:

**Bước 1 — Tạo Signing String:**
```
(request-target): {method} {path}
date: {date_header}
nonce: {uuid4_hex_32_ký_tự}
```
**Bước 2 — Ký và encode:**
- Ký Signing String bằng HMAC-SHA256 với API Secret → Base64 encode.
- URL-encode chỉ 3 ký tự: `+` → `%2B`, `/` → `%2F`, `=` → `%3D`.

**Bước 3 — Đóng gói header X-Signature:**
```
Signature keyId="{API_KEY}",algorithm="hmac-sha256",headers="(request-target) date",signature="{ENCODED_SIGNATURE}",nonce="{NONCE}"
```

> **Khuyến nghị:** Dùng [SDK DNSE](https://github.com/dnse-tech/openapi-sdk) để tự động sinh Signature, tránh các lỗi thủ công.

#### 10. Header Date cần định dạng gì?

Header Date có định dạng RFC1123, timezone UTC. VD: `Fri, 15 May 2026 07:11:30 +0000`

Date không được lệch quá **±1 phút** so với giờ chuẩn hệ thống DNSE. Nếu lệch quá thời gian này, request bị từ chối (cơ chế chống replay attack).

#### 11. Nonce là gì? Có thể tái sử dụng Signature cũ không?

Nonce là chuỗi UUID4 hex 32 ký tự (không có dấu gạch ngang), sinh mới cho mỗi request. Không được tái sử dụng Signature hoặc Nonce từ request trước.

#### 12. Request bị từ chối do sai Signature, tôi debug như thế nào?**

Kiểm tra lần lượt:
- Thứ tự dòng trong Signing String: `(request-target)` → `date` → `nonce`. Sai thứ tự = Signature sai.
- Method viết **thường** (`get`, `post`), tên field (`date`, `nonce`) viết **thường**.
- URL-encode: Chỉ encode 3 ký tự `+`, `/`, `=` từ Base64. Không double-encode.
- Header Date không lệch quá ±1 phút.
- Nonce đúng 32 ký tự hex, không có dấu gạch ngang.

Các thông tin khác về Xác thực API Key và Signature, người dùng tham khảo <a href="https://developers.dnse.com.vn/docs/guide/intro/register_guide">**tại đây**</a>

---

### Trading Token (2FA)

#### 13. Trading Token là gì? Khi nào cần dùng?

Trading Token là lớp bảo mật thứ 2 (2FA) bắt buộc cho các API giao dịch: **Đặt lệnh, Sửa lệnh, Hủy lệnh, Đóng vị thế**. Token có hiệu lực **8 tiếng** sau khi xác thực OTP thành công. Trong thời gian đó, bạn có thể giao dịch liên tục mà không cần xác thực lại.

#### 14. Làm thế nào để tôi lấy Trading Token?

**Bước 1:** Lấy mã OTP:
- Smart OTP: Mở app DNSE → SmartOTP → Lấy mã (hiệu lực 30 giây).
- Email OTP: Gọi `POST /openapi/registration/send-email-otp` (hiệu lực 2 phút).

**Bước 2:** Gọi `POST /openapi/registration/trading-token`:
```json
{
  "otpType": "email_otp",
  "passcode": "123456"
}
```

**Bước 3:** Lưu giá trị `trading-token` trả về, truyền vào header cho các request giao dịch.

#### 15. Trading Token hết hạn thì tôi cần làm gì?

Thực hiện lại bước xác thực OTP để lấy token mới. Khuyến nghị xây dựng cơ chế **tự động refresh token** trong ứng dụng để tránh gián đoạn giao dịch.

#### 16. Tôi bị lỗi "email otp is not registered for this account"?

Bạn đang truyền `otpType` không khớp với phương thức đang active. Kiểm tra phương thức OTP hiện tại tại **[Trang giao dịch chính thức DNSE](https://entradex.dnse.com.vn) → LightSpeed API** và đảm bảo phương thức đang sử dụng khớp với giá trị truyền lên: `"smart_otp"` hoặc `"email_otp"`.

---

## API Versioning

#### 17. DNSE quản lý version API như thế nào?

DNSE dùng **Date-based Versioning**. Version truyền qua header: `version: YYYY-MM-DD` (VD: `version: 2026-05-07`). Version mặc định nếu không truyền: `2026-05-07`.

Một version mới chỉ được tạo khi có **breaking changes**. Các thay đổi không phá vỡ tương thích (thêm field, thêm endpoint mới) không tạo version mới.

#### 18. Breaking change và non-breaking change khác nhau như thế nào?

| Breaking Changes (tạo version mới) | Non-breaking (không tạo version mới)  |
|------------------------------------|---------------------------------------|
| Thay đổi / xóa endpoint hiện tại   | Thêm endpoint mới                     |
| Đổi tên / xóa field trong response | Thêm field mới trong response         |
| Thay đổi kiểu dữ liệu              | Thêm optional parameter               |
| Thay đổi validation logic          | Thêm enum value mới tương thích ngược |
| Thay đổi auth mechanism            |                                       |

> **Khuyến nghị:** Implement parser theo hướng forward-compatible — bỏ qua các field không nhận diện trong response.

#### 19. Nếu tôi không truyền header version có ảnh hưởng gì không?

Nếu request không truyền Header version, hệ thống tự fallback về version mặc định `2026-05-07`. Request vẫn xử lý bình thường, nhưng người dùng sẽ bỏ lỡ các logic/tính năng mới từ breaking change version sau. DNSE khuyến nghị luôn **ghim (pin) version cụ thể** trong code.

#### 20. Tôi truyền sai định dạng version thì gặp lỗi gì?

Lỗi điển hình khi truyền sai định dạng Header version

```json
{
  "status": "error",
  "code": "OA-401",
  "message": "This API version does not seem to exist"
}
```

#### 21. DNSE có ngừng hỗ trợ version cũ không?

Hiện tại DNSE **chưa áp dụng sunset version**. Các version cũ vẫn tiếp tục được hỗ trợ. Nếu có thay đổi chính sách, DNSE thông báo qua [Changelog](https://developers.dnse.com.vn/docs/changelog).

Các thông tin khác về Verioning, người dùng tham khảo <a href="https://developers.dnse.com.vn/docs/guide/versioning/api">**tại đây**</a>
---

## Rate Limit

#### 22. Rate limit của DNSE hoạt động như thế nào?

Rate limit áp dụng theo từng **API Key** và từng **Endpoint**, với 2 ngưỡng:
- **Rate**: Tổng request tối đa trong 1 giờ.
- **Quota**: Tổng request tối đa trong 24 giờ.

Một số giới hạn phổ biến:

| Endpoint               | Rate/giờ | Quota/ngày |
|------------------------|----------|------------|
| Đặt / Sửa / Hủy lệnh   | 50,000   | 100,000    |
| Sổ lệnh, Chi tiết lệnh | 100,000  | 1,000,000  |
| Sức mua, sức bán       | 10,000   | 100,000    |
| Gửi Email OTP          | 100      | 1,000      |
| Xác thực OTP           | 100      | 1,000      |

#### 23. Tôi gặp lỗi `429 Too Many Requests` thì cần làm gì?

- Kiểm tra header response: `X-RateLimit-Remaining` và `X-RateLimit-Reset`.
- Implement **exponential backoff** — chờ trước khi gửi lại request.
- **Cache** dữ liệu ít thay đổi để giảm số lần gọi API.
- Phân bổ tần suất gọi hợp lý trong từng khoảng thời gian.

#### 24. Các header rate limit trong response có ý nghĩa gì?

| Header                  | Ý nghĩa                                  |
|-------------------------|------------------------------------------|
| `X-RateLimit-Limit`     | Tổng request tối đa được phép            |
| `X-RateLimit-Remaining` | Số request còn lại trong chu kỳ hiện tại |
| `X-RateLimit-Reset`     | Thời điểm giới hạn được làm mới          |

Các thông tin khác về Rate limit, người dùng tham khảo <a href="https://developers.dnse.com.vn/docs/guide/ratelimits">**tại đây**</a>

---

## Giao dịch

#### 25. Làm thế nào để lấy danh sách tiểu khoản?

Gọi `GET /accounts`. Response trả về:
- `investorId`: Mã định danh khách hàng — dùng cho WebSocket Trading.
- `custodyCode`: Số tài khoản lưu ký VSD.
- `accounts[].id`: Số tiểu khoản (`accountNo`) — dùng làm input cho các REST API giao dịch.
- `accounts[].derivativeAccount`: `true/false` — tiểu khoản có được giao dịch phái sinh không.

#### 26. `investorId` và `accountNo` khác nhau như thế nào?

- **accountNo** (tiểu khoản): Dùng cho các REST API giao dịch (đặt lệnh, sổ lệnh, số dư...). Một khách hàng có thể có nhiều tiểu khoản.
- **investorId**: Dùng cho kết nối WebSocket Trading Data để nhận dữ liệu lệnh/vị thế realtime. Mỗi khách hàng có 1 `investorId` duy nhất.

#### 27. Đặt lệnh

**Các thông tin bắt buộc khi đặt lệnh là gì?**

| Trường          | Mô tả                                  |
|-----------------|----------------------------------------|
| `marketType`    | `STOCK` hoặc `DERIVATIVE`              |
| `orderCategory` | Loại lệnh (`NORMAL` với lệnh thường)   |
| `accountNo`     | Số tiểu khoản (lấy từ GET /accounts)   |
| `symbol`        | Mã chứng khoán (viết hoa)              |
| `loanPackageId` | Mã gói vay (lấy từ GET /loan-packages) |
| `side`          | `NB` (mua) hoặc `NS` (bán)             |
| `orderType`     | Loại lệnh tương ứng sàn                |
| `quantity`      | Khối lượng đặt                         |
| `price`         | Giá đặt (= 0 nếu không phải lệnh LO)   |

Header bổ sung bắt buộc: `trading-token`.

Chi tiết Đặc tả <a href="https://developers.dnse.com.vn/docs/dnse/place-order">**API đặt lệnh**</a>

#### 28. Sàn HOSE, HNX, Upcom hỗ trợ những loại lệnh nào?**

| Sàn   | Loại lệnh hỗ trợ            |
|-------|-----------------------------|
| HOSE  | ATO, ATC, LO, MTL           |
| HNX   | LO, MTL, MOK, MAK, ATC, PLO |
| Upcom | LO                          |

#### 29. Quy tắc khối lượng đặt lệnh cơ sở?

- **Lô chẵn**: Bội số của 100 (100, 200, 300...).
- **Lô lẻ**: Từ 1 đến 99.
- **Lẻ lô** (101, 102...): KHÔNG hợp lệ.

Khối lượng không được vượt quá `qmaxBuy` hoặc `qmaxSell` trả về từ `GET /ppse`.

#### 30. Giá đặt lệnh cần thỏa mãn điều kiện gì?

- Lệnh **LO**: Giá > 0 và nằm trong khoảng [giá sàn, giá trần] của mã tại phiên giao dịch đó.
- Các loại lệnh khác (ATO, ATC, MTL, MOK, MAK): `price` truyền lên = 0.

Lấy giá trần/sàn: `GET /secdef` hoặc nhận qua WebSocket Security Definition.

#### 31. `loanPackageId` là gì? Tôi có thể lấy ở đâu?

`loanPackageId` là mã gói vay — bắt buộc khi đặt lệnh. Gọi `GET /loan-packages` để lấy danh sách.
Với giao dịch cơ sở, response trả về tối đa 2 gói: gói tiền mặt (`type: N`) và gói margin (`type: M`).

Chi tiết Đặc tả <a href="https://developers.dnse.com.vn/docs/dnse/get-loan-packages">**API danh sách gói vay**</a>

#### 32. Điều kiện để sửa lệnh là gì?

- Chỉ áp dụng cho lệnh **LO** trong phiên giao dịch liên tục.
- Trạng thái lệnh phải là **New** (Chờ khớp) hoặc **PartiallyFilled** (Khớp một phần)
- Giá sửa phải nằm trong biên độ trần/sàn.
- Nếu vượt quá sức mua/bán, yêu cầu bị từ chối.

#### 33. *Sửa lệnh cơ sở và phái sinh khác nhau như thế nào?**

|                                | Cơ sở (STOCK)            | Phái sinh (DERIVATIVE)       |
|--------------------------------|--------------------------|------------------------------|
| Sửa đồng thời giá + Khối lượng | Được                     | Không được                   |
| Khối lượng sửa                 | Không giới hạn           | Lớn hơn khối lượng đã khớp   |
| Thứ tự ưu tiên                 | Xác định lại sau khi sửa | Thay đổi nếu khối lượng tăng |

#### 34. Trạng thái lệnh

**Vòng đời lệnh và ý nghĩa từng trạng thái**

| Trạng thái        | Ý nghĩa                                              |
|-------------------|------------------------------------------------------|
| `Pending`         | Lệnh vừa gửi lên hệ thống DNSE, đang kiểm tra nội bộ |
| `PendingNew`      | Lệnh hợp lệ, đang chờ gửi lên Sở                     |
| `New`             | Sở ghi nhận, đang chờ khớp                           |
| `PartiallyFilled` | Khớp một phần, phần còn lại chờ khớp                 |
| `Filled`          | Khớp toàn bộ                                         |
| `PendingReplace`  | Yêu cầu sửa đang được xử lý                          |
| `PendingCancel`   | Yêu cầu hủy đang được xử lý                          |
| `Canceled`        | Hủy thành công                                       |
| `Rejected`        | Bị từ chối (không đủ điều kiện)                      |
| `Expired`         | Hết hiệu lực do kết thúc phiên                       |
| `DoneForDay`      | Kết thúc vòng đời trong ngày giao dịch               |

#### 35. Lệnh bị Rejected — nguyên nhân thường gặp?

- Không đủ sức mua / sức bán.
- Gói vay không hợp lệ hoặc vượt hạn mức cho vay.
- Giá ngoài biên độ trần/sàn.
- Khối lượng không hợp lệ (lẻ lô sai quy cách).
- Trading Token không hợp lệ hoặc hết hạn.

#### 36. Margin & Gói vay

**Mô hình Isolated Margin (Position) của DNSE là gì?**

Mỗi **Position = 1 mã chứng khoán + 1 gói vay**, được quản trị rủi ro độc lập. Ví dụ: Mua HPG với gói tiền mặt tạo Position A; mua thêm HPG với gói margin 50% tạo Position B riêng biệt.

Ưu điểm: Chỉ Position nào xuống dưới ngưỡng cảnh báo mới bị call margin hoặc force sell — không ảnh hưởng các Position an toàn khác.

#### 37. Các tỷ lệ trong gói vay có ý nghĩa gì?

| Trường            | Ý nghĩa                                                    |
|-------------------|------------------------------------------------------------|
| `initialRate`     | Tỷ lệ ký quỹ ban đầu (VD: 0.5 = ký quỹ 50%, vay 50%)       |
| `maintenanceRate` | Tỷ lệ duy trì — xuống dưới mức này → cảnh báo call margin  |
| `liquidRate`      | Tỷ lệ xử lý — xuống dưới mức này → DNSE force sell Deal đó |

---

## WebSocket

#### 38. Base URL và cơ chế kết nối WebSocket Trading?

- **Base URL**: `wss://ws-openapi.dnse.com.vn`
- Kết nối tối đa **8 tiếng**, sau đó server tự ngắt.
- Cơ chế keepalive: Server gửi **PING** mỗi 3 phút, client phải trả **PONG** trong vòng 1 phút. Client có thể chủ động gửi PONG để duy trì kết nối.

#### 39. Channel nhận dữ liệu lệnh realtime (Order Event)?**

```
order.{market_type}.{encoding}
```
- `market_type`: `STOCK` hoặc `DERIVATIVE`
- `encoding`: `json` hoặc `msgpack`

Dữ liệu được đẩy realtime khi lệnh có thay đổi: tạo mới, đổi trạng thái, khớp giá/khối lượng.

#### 40. Channel nhận dữ liệu vị thế realtime (Position Event)

```
position.{market_type}.{encoding}
```

Dữ liệu được đẩy khi vị thế có thay đổi: mở/đóng vị thế, thay đổi khối lượng, giá vốn, giá thị trường, trạng thái.

#### 41. WebSocket — Market Data

**Market Data WebSocket cung cấp những loại dữ liệu nào?**

| Channel             | Mô tả                                  | Tần suất              |
|---------------------|----------------------------------------|-----------------------|
| Security Definition | Giá trần/sàn/tham chiếu, trạng thái mã | 1 lần/ngày (~8h sáng) |
| Trade / Trade Extra | Dữ liệu khớp lệnh                      | Realtime              |
| Quotes              | Độ sâu thị trường bid/ask              | Realtime              |
| OHLC                | Nến đang hình thành                    | Realtime              |
| OHLC Closed         | Nến đã đóng                            | Theo khung thời gian  |
| Expected Price      | Giá dự khớp ATO/ATC                    | Realtime              |
| Market Index        | Chỉ số VNINDEX, HNX...                 | Mỗi 5 giây            |
| Foreign Investor    | Giao dịch NĐT nước ngoài               | Realtime              |
| Estimated VN30      | Chỉ số VN30 dự tính                    | Realtime              |

#### 42. Trade và Trade Extra khác nhau như thế nào?**

- **Trade** (`tick.{board_id}.{encoding}`): Dữ liệu khớp lệnh cơ bản — giá, khối lượng, tổng KL ngày, giá cao/thấp/mở cửa.
- **Trade Extra** (`tick_extra.{board_id}.{encoding}`): Bổ sung thêm `side` (mua/bán chủ động) và `avgPrice` (giá khớp trung bình).

Nếu không cần thông tin mua/bán chủ động, dùng **Trade** để tối ưu tốc độ và băng thông.

#### 43. Security Definition bị bỏ lỡ thì lấy dữ liệu ở đâu?

Security Definition gửi 2 lần/ngày (~8h sáng và ~20h tối). Nếu kết nối sau khi dữ liệu đã gửi, gọi REST API `GET /secdef` để lấy thông tin giá trần/sàn cho mã cụ thể.

#### 44. Mã chứng khoán truyền vào WebSocket cần định dạng gì?

Tất cả mã phải ở **chữ IN HOA**. VD: `ACB`, `HPG`, `41I1G2000`.  
Đặc biệt với OHLC channel cho phái sinh: Truyền `symbolType` (VD: `VN30F1M`) thay vì `symbol` cụ thể.

#### 45. Nên dùng encoding `json` hay `msgpack`?

- **json**: Dễ đọc, phù hợp giai đoạn phát triển và debug.
- **msgpack**: Tốc độ nhanh hơn, tiết kiệm băng thông — phù hợp production.

---

## Broker & SACO

#### 46. Môi giới / SACO cần điều kiện gì để sử dụng OpenAPI
- Tài khoản chứng khoán DNSE trạng thái ACTIVE.
- Đã được xác nhận là Môi giới / SACO chính thức tại DNSE.
- Đăng ký thành công dịch vụ LightSpeed API.

Khách hàng của Môi giới cũng phải có tài khoản chứng khoán đang hoạt động tại DNSE và đã liên kết ủy quyền với Môi giới / SACO.

#### 47. Môi giới lấy danh sách tài khoản khách hàng bằng API nào?

Gọi `GET /brokers/accounts/care-by`. Response trả về: `accountNo` (dùng cho REST API), `investorId` (dùng cho Broker WebSocket), `permissions` (quyền ADVISOR/BROKER theo từng sản phẩm).

Đây là bước đầu tiên bắt buộc trước khi thực hiện bất kỳ nghiệp vụ nào trên tài khoản KH.

#### 48. Quyền ADVISOR và BROKER khác nhau như thế nào?**

| Quyền     | Xem lệnh, vị thế, tài sản | Đặt lệnh |
|-----------|---------------------------|----------|
| `ADVISOR` | ✅                         | ❌        |
| `BROKER`  | ✅                         | ✅        |

Nếu cố đặt lệnh khi chỉ có quyền ADVISOR, hệ thống từ chối và trả về lỗi phân quyền.

#### 49. Liên kết / hủy liên kết giữa Môi giới và KH thực hiện ở đâu?**

BẮT BUỘC thực hiện trên **App/Web Entrade X by DNSE** — không thực hiện qua OpenAPI. Phạm vi ủy quyền (ADVISOR/BROKER) cũng được xác định trong bước này.

#### 50. Broker WebSocket nhận dữ liệu lệnh khách hàng qua channel nào?**

```
order.broker.{market_type}.{investor_id}.{encoding}
```

- `investor_id`: Lấy từ `GET /brokers/accounts/care-by`.
- Mỗi kết nối WebSocket chỉ subscribe được **1 investor_id**. Muốn theo dõi nhiều KH cần mở nhiều kết nối riêng biệt.

#### 51. Tôi subscribe Broker WebSocket bị lỗi phân quyền, nguyên nhân là gì?

- `investor_id` không thuộc danh sách KH đang được ủy quyền — chỉ dùng `investorId` lấy từ `GET /brokers/accounts/care-by`.
- Liên kết giữa Môi giới và KH chưa thiết lập hoặc đã bị hủy trên Entrade X.
- API Key / Token không hợp lệ hoặc hết hiệu lực.

---

## Tài liệu & Tài nguyên

| Tài nguyên               | Link                                                                           |
|--------------------------|--------------------------------------------------------------------------------|
| Tài liệu kỹ thuật đầy đủ | https://developers.dnse.com.vn                                                 |
| Changelog                | https://developers.dnse.com.vn/docs/changelog                                  |
| SDK & Code samples       | https://github.com/dnse-tech/openapi-sdk                                       |
| Python WebSocket SDK     | https://github.com/dnse-tech/openapi-sdk/tree/main/python/websocket-marketdata |

---

*Nếu câu hỏi của bạn chưa có trong danh sách này, vui lòng liên hệ đội hỗ trợ kỹ thuật.*
