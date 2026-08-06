
# Dành cho Brokers DNSE

Cung cấp hướng dẫn cách sử dụng OpenAPI cho tài khoản Môi giới/SACO (Sales Collaboration) để quản lý và thực hiện giao dịch trên tài khoản khách hàng (khi được ủy quyền).

### Tổng quan

Môi giới/SACO là người đồng hành cùng khách hàng trong quá trình đầu tư tại DNSE, có thể hỗ trợ theo dõi và thực hiện giao dịch cho khách hàng thông qua cơ chế ủy quyền.

Khi sử dụng OpenAPI, Môi giới/SACO có thể:
- Xem danh sách các khách hàng đang quản lý
- Thực hiện các nghiệp vụ như đặt lệnh, theo dõi tài sản cho từng khách hàng tùy theo mức độ ủy quyền.

#### Điều kiện cần
Các điều kiện cần đảm bảo từ phía người dùng:
- Người dùng là Môi giới/SACO cần có tài khoản chứng khoán tại DNSE với trạng thái đang hoạt động (ACTIVE).
- Được xác nhận là Môi giới hoặc SACO chính thức tại DNSE. Xem hướng dẫn đăng ký [tại đây.](https://hdsd.dnse.com.vn/san-pham-dich-vu/huong-dan-ban-dong-hanh-saco/0.-dnse_huong-dan-dang-ky-tro-thanh-saco)
- Cần đăng ký thành công dịch vụ LightSpeed API của DNSE để có các thông tin quan trọng phục vụ kết nối OpenAPI. Hướng dẫn đăng ký <a href="https://developers.dnse.com.vn/docs/guide/intro/register_guide">tại đây.</a>

#### Kết nối và ủy quyền khách hàng
- Khách hàng của Môi giới/SACO phải có tài khoản chứng khoán tại DNSE với trạng thái đang hoạt động (ACTIVE).
- Việc liên kết giữa Môi giới/SACO và khách hàng:
  - Kết nối hoặc hủy kết nối BẮT BUỘC thực hiện trên App/Web Entrade X by DNSE
  - Phạm vi Môi giới/SACO được uỷ quyền (chỉ xem / đặt lệnh) phụ thuộc vào nội dung ủy quyền đã xác nhận giữa hai bên khi thực hiện liên kết.

  Người dùng xem hướng dẫn kết nối và cài đặt phạm vi ủy quyền [tại đây.](https://hdsd.dnse.com.vn/san-pham-dich-vu/huong-dan-ban-dong-hanh-saco/1.-dnse_huong-dan-chon-ban-dong-hanh-dau-tu)
- Môi giới/SACO chỉ có thể sử dụng các tính năng OpenAPI trên các tài khoản khách hàng thuộc phạm vi quản lý của mình.


### Luồng tích hợp

#### Bước 1: Lấy thông tin tài khoản khách hàng
Đây là điểm khởi đầu cho mọi thao tác. API <a href="https://developers.dnse.com.vn/docs/dnse/get-brokers-accounts-care-by">/Danh sách tiểu khoản quản lý</a> trả về tất cả các tài khoản thuộc ủy quyền của người dùng.
- Endpoint: GET `/brokers/accounts/care-by`
- Response:

```json lines
{
  "accountNo": "0335000633", // Số tiểu khoản
  "fullName": "Hoang Tu Mai", // Họ tên khách hàng
  "custodyCode": "064C1MAI20",  // Số lưu ký của khách hàng
  "investorId":  "1000009250", // Mã định danh khách hàng tại DNSE 
  "underlyingNav": 21345678901, // Tài sản ròng cơ sở
  "derivativeNav": 32789123456, // Tài sản ròng phái sinh
  "totalNav": 54134802357,  // Tổng tài sản ròng
  "dealAccount": true,   // Tài khoản theo Deal hoặc không
  "derivativeAccount": true,  // Tiểu khoản được phép giao dịch phái sinh hoặc không
  "derivative": {
    "status": "ACTIVE" // Trạng thái tài khoản phái sinh
  },
  "permissions": [ // Phạm vi quyền quản lý
    {
      "product": "UNDERLYING_STOCK", // Sản phẩm chứng khoán cơ sở
      "role": "ADVISOR" // Quyền theo dõi
    },
    {
      "product": "DERIVATIVES_STOCK", // Sản phẩm chứng khoán phái sinh
      "role": "BROKER" // Quyền giao dịch
    }
  ]
}
```
Các dữ liệu trả về quan trọng:
- `accountNo`: Số tài khoản của khách hàng (Dùng làm input cho các API sau)
- `investorId`: Mã định danh khách hàng tại DNSE
- `permissions`: Quyền thao tác tương ứng với từng sản phẩm
  - `ADVISOR`: Chỉ được phép xem và quản lý tài sản
  - `BROKER`: Giao dịch chứng khoán, xem và quản lý tài sản

#### Bước 2: Giao dịch trên tài khoản khách hàng
Sau khi có `accountNo`, người dùng sử dụng các Endpoint tiêu chuẩn của OpenAPI giống như tài khoản thông thường.

Hệ thống sẽ tự động kiểm tra ủy quyền giữa tài khoản Môi giới/SACO và tiểu khoản cần giao dịch trước khi thực thi các nghiệp vụ. Nếu không đáp ứng đủ các điều kiện, hệ thống sẽ trả về lỗi tương ứng.

Một số nghiệp vụ phổ biến:

- Truy vấn số dư: Sử dụng Endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-account-balances">/Thông tin tiền</a>
- Đặt lệnh: Sử dụng Endpoint <a href="https://developers.dnse.com.vn/docs/dnse/place-order">/Đặt lệnh</a>

  Luôn kiểm tra mảng `permissions` trước khi thực hiện lệnh. Nếu người dùng đặt lệnh cho một tiểu khoản chỉ có quyền ADVISOR (theo dõi), yêu cầu sẽ bị từ chối và hệ thống trả về lỗi.
- Tra cứu vị thế nắm giữ: Sử dụng Endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-positions">/Vị thế nắm giữ</a>

Để biết thêm chi tiết về các thông số kỹ thuật của từng Endpoint, vui lòng tham khảo trang <a href="https://developers.dnse.com.vn/docs/dnse/account">Đặc tả API.</a>

#### Bước 3: Nhận dữ liệu lệnh và vị thế trên tài khoản khách hàng theo thời gian thực
Sau khi có `investorId`, người dùng kết nối đến WebSocket DNSE để nhận dữ liệu lệnh trên tài khoản khách hàng.
Hệ thống sẽ tự động kiểm tra ủy quyền giữa tài khoản Môi giới/SACO và tài khoản khách hàng khi có yêu cầu kết nối tới các Channel.

Chi tiết vui lòng tham khảo tại <a href="https://developers.dnse.com.vn/docs/guide/market-data/broker_connect">Broker WebSocket.</a>
