---
sidebar_position: 1
---

#  DNSE API Platform
---

### Giới thiệu

Chào mừng khách hàng và đối tác đến với tài liệu ***LightSpeed API*** của DNSE cung cấp. Dịch vụ OpenAPI mang đến trải nghiệm đầu tư toàn diện, linh hoạt và hiện đại với những lợi thế vượt trội:

- **Chủ động trong việc xây dựng hành trình đầu tư:** Từ theo dõi biến động thị trường, đưa ra quyết định đến đặt lệnh – mọi thao tác đều có thể được lập trình và quản lý chủ động bởi người dùng.
- **Nguồn dữ liệu thị trường đa dạng:** Cung cấp đầy đủ từ độ sâu thị trường, biến động giá thị trường, thông tin OHLC, các chỉ số indices và nhiều loại dữ liệu khác.
- **Tốc độ xử lý vượt trội:** Cập nhật realtime thông tin biến động tài sản, sổ lệnh giao dịch.
- **Khả năng mở rộng và tích hợp đa nền tảng:** Xây dựng ứng dụng giao dịch chủ động đơn giản cho cá nhân đến hệ thống phân tích và nền tảng giao dịch phức tạp dành cho tổ chức.

### Đối tượng sử dụng

- Nhà đầu tư cá nhân muốn số hóa chiến lược và quản lý danh mục realtime.
- Doanh nghiệp Công nghệ Tài chính muốn tích hợp dữ liệu thị trường và giao dịch vào sản phẩm.
- Đối tác Tổ chức Tài chính cần giám sát danh mục và xử lý số lượng lệnh lớn hay mở rộng dịch vụ tích hợp.
- Nhà phát triển công nghệ xây dựng giao dịch chủ động, dashboard hoặc công cụ phân tích.

### Sơ đồ hệ thống

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/sd1.png)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/sd1.png)
</div>

### Mô hình đa tầng bảo mật

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/sd2.png)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/sd2.png)
</div>

Sau khi đăng ký thành công, khách hàng sẽ nhận được bộ chuỗi bảo mật bao gồm: API Key, API secret.

**API Key**

- API Key được cung cấp sau khi đăng ký sử dụng OpenAPI, đây là khóa định danh duy nhất, đóng vai trò nhận diện và xác minh danh tính khi kết nối với hệ thống.
- API Key phải được giữ bí mật và chỉ dùng cho mục đích gọi API. Khách hàng có thể chủ động tạo mới hoặc thu hồi bất kỳ lúc nào.
- Khi tạo mới hoặc hủy API Key, bộ khóa cũ sẽ lập tức vô hiệu lực, giảm thiểu rủi ro khi bị lộ hoặc không còn nhu cầu sử dụng.

**API Secret**

- API Secret là một chuỗi ký tự mật dùng để xác minh và bảo vệ API, được sử dụng để sinh chữ ký số Signature cần thiết cho hầu hết các REST API.
- API Key và API Secret là cặp khóa luôn đi cùng nhau, có thể hiểu tương tự như tên người dùng và mật khẩu.
- API Secret chỉ hiển thị duy nhất một lần khi đăng ký thành công để bảo mật cho tài khoản. Khách hàng cần chủ động lưu lại và quản lý thông tin này.

**Phương thức xác thực lớp thứ 2 (2FA)**

Bên cạnh API Key và API Secret, hệ thống áp dụng thêm lớp xác thực thứ hai (2FA) với giao dịch đặt lệnh.

- Tại mỗi thời điểm, chỉ một phương thức xác thực lớp thứ hai được kích hoạt và là phương thức duy nhất được hệ thống chấp nhận khi thực hiện xác thực OTP.
- Khách hàng lựa chọn và có thể thay đổi giữa Smart OTP hoặc Email OTP.

Việc kết hợp API Key, API Secret và xác thực lớp thứ hai đảm bảo rằng chỉ các yêu cầu được thực hiện bởi đúng chủ tài khoản, đáp ứng đầy đủ điều kiện xác thực mới được hệ thống chấp nhận và xử lý.

--- 

:::tip[Lưu ý]

Các chuỗi bảo mật DNSE đã cung cấp trên là thông tin nhạy cảm cần được bảo mật nghiêm ngặt. Tuyệt đối không chia sẻ hoặc tiết lộ cho bất kỳ cá nhân hay tổ chức nào không thuộc phạm vi được ủy quyền sử dụng. Việc bảo mật tốt giúp ngăn chặn các rủi ro về truy cập trái phép và bảo vệ tài khoản.

:::
