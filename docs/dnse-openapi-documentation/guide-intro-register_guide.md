---
sidebar_position: 2
---

# Đăng ký & Quản lý dịch vụ
---

### Đăng ký sử dụng

Để bắt đầu tích hợp và sử dụng Lightspeed API của DNSE, khách hàng có thể thực hiện đăng ký dễ dàng qua [trang web giao dịch trực tuyến](https://entradex.dnse.com.vn) chính thức của DNSE

#### Bước 1: Truy cập trang web OpenAPI DNSE
Khách hàng chọn **Đăng ký** tại trang chủ OpenAPI: <a href="https://developers.dnse.com.vn"></a>

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk.png?ts=123456)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk.png)
</div>

Hệ thống sẽ chuyển đến trang giao dịch trực tuyến của DNSE để khách hàng đăng nhập (hoặc tạo tài khoản mới nếu chưa có). Sau khi đăng nhập sẽ điều hướng đến trang thông tin Lightspeed API.

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk2.png?ts=123456)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk2.png)
</div>

Hoặc ngay tại giao diện đã đăng nhập của trang giao dịch trực tuyến DNSE, khách hàng chọn Họ tên để đến trang Thông tin tài khoản, chọn tiếp LightSpeed API.

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk3.png?ts=123456)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk3.png)

</div>

#### Bước 2: Thực hiện đăng ký
Khách hàng lựa chọn duy nhất 1 phương thức xác thực lớp thứ hai để nhận mã OTP khi thực hiện giao dịch đặt lệnh qua OpenAPI.

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk4.png?ts=123456)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk4.png)

</div>

**Smart OTP**

- Để chọn phương thức, tài khoản chứng khoán cần phải kích hoạt sử dụng SmartOTP trên ứng dụng DNSE.
- Mã OTP để đặt lệnh lấy trực tiếp tại ứng dụng DNSE trên thiết bị di động đã đăng ký SmartOTP.

**Email OTP**

- Để chọn phương thức, tài khoản chứng khoán cần có Email hợp lệ và đã được xác thực.
- Mã OTP để đặt lệnh được gửi về địa chỉ Email đã đăng ký.
- Khách hàng có thể sử dụng tối đa 2 địa chỉ Email để nhận mã OTP.

Sau khi đã lựa chọn phương thức xác thực lớp thứ hai, chọn **Đăng ký**.

Khách hàng kiểm tra lại thông tin, thực hiện Xác nhận và Xác thực OTP để hoàn tất đăng ký.

#### Bước 3: Đăng ký thành công
Xác thực thành công, hệ thống sẽ hiển thị trạng thái Đã đăng ký và các thông tin quan trọng để kết nối OpenAPI.

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk5.png?ts=123456)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk5.png)
</div>

API Secret chỉ hiển thị một lần duy nhất sau khi đăng ký thành công. Khách hàng cần chủ động lưu lại và quản lý thông tin này.

Tham khảo chi tiết về các khóa bảo mật <a href="https://developers.dnse.com.vn/docs/guide/trading-api/authentication">tại đây.</a>

---
### Quản lý API Key

Sau khi đăng ký thành công, khách hàng có thể chủ động quản lý vòng đời API Key của mình, bao gồm :
- **Tạo lại API Key:** Bộ khóa API Key và API Secret sẽ tự động bị vô hiệu hóa và không thể tiếp tục sử dụng. Một bộ khóa mới sẽ được sinh ra và API Secret chỉ hiển thị một lần duy nhất. Việc Tạo lại này không ảnh hưởng tới phương thức xác thực lớp thứ 2 đang sử dụng.
- **Hủy API Key:** Đồng nghĩa với Hủy sử dụng dịch vụ OpenAPI, nếu muốn sử dụng lại, khách hàng cần thực hiện đăng ký mới từ đầu.

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk6.png?ts=123456)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk6.png)
</div>

Quản lý API Key giúp đảm bảo an toàn bảo mật, đặc biệt trong trường hợp nghi ngờ bị lộ thông tin hoặc không còn nhu cầu sử dụng.

:::tip[Lưu ý]

Thông tin về API key, API secret là thông tin nhạy cảm cần được bảo mật nghiêm ngặt. Tuyệt đối không chia sẻ hoặc tiết lộ cho bất kỳ cá nhân hay tổ chức nào không thuộc phạm vi được ủy quyền sử dụng. Việc bảo mật tốt giúp ngăn chặn các rủi ro về truy cập trái phép và bảo vệ tài khoản.


:::

---
### Thay đổi phương thức xác thực lớp thứ 2

Tại một thời điểm, chỉ có một phương thức xác thực lớp thứ 2 được sử dụng cho OpenAPI. Sau khi đăng ký thành công, khách hàng có thể thay đổi cách nhận mã OTP bất kỳ lúc nào giữa Smart OTP và Email OTP.

Khách hàng chọn **Đổi phương thức xác thực** và thao tác xác thực OTP để hoàn tất.

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk7.png?ts=123456)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk7.png)
</div>

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk8.png?ts=123456)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk8.png)
</div>

Việc thay đổi phương thức xác thực không ảnh hưởng tới các khóa API Key và API Secret hiện tại.

#### Lưu ý sử dụng Email OTP

Hệ thống hỗ trợ tối đa 2 địa chỉ Email để nhận mã OTP

- Trường hợp đăng ký mới: Khách hàng cần chọn Email OTP → Thêm Email thứ 2 → gửi yêu cầu Đăng ký.
- Trường hợp đã đăng ký:

  - Nếu khách hàng đang sử dụng Email OTP → Thêm Email thứ 2 → xác nhận OTP cập nhật Email.
  - Nếu khách hàng đang sử dụng Smart OTP → Thêm Email thứ 2 → Đổi phương thức xác thực.

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk9.png?ts=123456)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/dk9.png)
</div>

Khách hàng cần xác thực Email thứ 2 để kích hoạt nhận mã OTP về địa chỉ mail này. Đường link xác thực sẽ được gửi về địa chỉ Email thứ 1 để đảm bảo an toàn và bảo mật.