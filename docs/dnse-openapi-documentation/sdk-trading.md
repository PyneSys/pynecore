---
sidebar_position: 2
---

# Trading SDKs

---
DNSE cung cấp và hỗ trợ [sample SDKs](https://github.com/dnse-tech/openapi-sdk) với đa dạng ngôn ngữ. Người dùng có thể tải xuống và ứng dụng ngay vào các scripts hoặc luồng giao dịch của mình.

- SDK Python: https://github.com/dnse-tech/openapi-sdk/tree/main/python
- SDK Javascript: https://github.com/dnse-tech/openapi-sdk/tree/main/javascript

Trang này cung cấp các ví dụ SDKs minh hoạ cách thiết lập và thực hiện các yêu cầu (Request) cơ bản với OpenAPI, bao gồm xác thực, truy vấn thông tin tài khoản và thực hiện giao dịch.

---
### Thông tin tài khoản

#### Tài khoản giao dịch
Lấy danh sách tất cả các tiểu khoản giao dịch (sub-account) thuộc quyền quản lý của tài khoản tương ứng với API Key.

<details>
  <summary>SDK get_accounts</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)
status, body = client.get_accounts(dry_run=False)
print(status, body)
```
</details>

#### Thông tin tiền

Truy vấn thông tin tài sản cơ sở và phái sinh trên tiểu khoản giao dịch.
<details>
  <summary>SDK get_balances</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)

status, body = client.get_balances(
    account_no="0003979888",        // Số tiểu khoản giao dịch
    dry_run=False
)
print(status, body)
```
</details>

#### Danh sách gói vay

Lấy mã gói vay để đặt lệnh giao dịch tùy theo từng mã chứng khoán. Dựa vào response trả về, người dùng có thể chọn gói tiền mặt hoặc gói vay margin theo nhu cầu.

<details>
  <summary>SDK get_loan_packages</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)

status, body = client.get_loan_packages(
    account_no="0003979888",        // Số tiểu khoản giao dịch
    market_type="STOCK",     // STOCK (gói vay cơ sở) hoặc DERIVATIVE (gói vay phái sinh)
    symbol="HPG",       // Mã chứng khoán
    dry_run=False,
)
print(status, body)
```
</details>

#### Sức mua, sức bán

Truy vấn thông tin sức mua và sức bán của tài khoản theo mã chứng khoán và gói vay để kiểm tra khả năng đặt lệnh trước khi giao dịch.

<details>
  <summary>SDK get_ppse</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)

status, body = client.get_ppse(
    account_no="0003979888",
    market_type="STOCK",     // STOCK (gói vay cơ sở) hoặc DERIVATIVE (gói vay phái sinh)
    symbol="HPG",
    price=26000,
    loan_package_id=1775,
    dry_run=False,
)

print(status, body)
```
</details>

#### Sổ lệnh

Truy vấn sổ lệnh giao dịch trong ngày theo từng thị trường cơ sở hoặc phái sinh, bao gồm trạng thái và thông tin xử lý của từng lệnh.

<details>
  <summary>SDK get_orders</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)
status, body = client.get_orders(
    account_no="0003979888",
    market_type="STOCK",     // STOCK (gói vay cơ sở) hoặc DERIVATIVE (gói vay phái sinh)
    orderCategory="NORMAL",
    dry_run=False,
)
print(status, body)
```
</details>

#### Chi tiết lệnh theo ID

Truy vấn thông tin chi tiết của một lệnh giao dịch cụ thể theo `orderId`, bao gồm trạng thái, khối lượng, giá và các thông tin liên quan trong quá trình khớp lệnh.

<details>
  <summary>SDK get_order_detail</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)
status, body = client.get_order_detail(
    account_no="0003979888",     // Số tiểu khoản giao dịch
    market_type="STOCK",     // STOCK (gói vay cơ sở) hoặc DERIVATIVE (gói vay phái sinh)
    order_id="123",         // ID lệnh
    dry_run=False,
)
print(status, body)
```
</details>

#### Vị thế nắm giữ

Truy vấn danh sách các vị thế đang nắm giữ trên tài khoản, bao gồm thông tin khối lượng, giá vốn, lãi/lỗ dự tính và các thông tin khác.

<details>
  <summary>SDK get_positions</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)
status, body = client.get_positions(
    account_no="0003979888",
    market_type="STOCK",     // STOCK (vị thế cơ sở) hoặc DERIVATIVE (vị thế phái sinh)
    dry_run=False,
)
print(status, body)
```
</details>

#### Chi tiết vị thế theo ID

Truy vấn thông tin chi tiết của một vị thế cơ sở hoặc phái sinh đang mở theo `positionId`

<details>
  <summary>SDK get_position_by_id</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)
status, body = client.get_position_by_id(
    position_id="189",
    market_type="STOCK",
    dry_run=False,    
)
print(status, body)
```
</details>

#### Lịch sử lệnh đã đặt

Truy vấn thông tin danh sách lệnh đã đặt trong một khoảng thời gian nhất định. Thời gian tra cứu tối đa trong vòng 1 năm kể từ ngày hiện tại.

<details>
  <summary>SDK get_order_history</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)
status, body = client.get_order_history(
    account_no="0003979888",
    market_type="STOCK",     // STOCK (cơ sở) hoặc DERIVATIVE (phái sinh)
    from_date="2026-03-03",
    to_date="2026-02-01",
    dry_run=False,
)
print(status, body)
```
</details>

### Giao dịch

#### Lấy Email OTP (optional)

Gửi yêu cầu nhận mã OTP qua email, chỉ áp dụng cho các tài khoản đang sử dụng phương thức xác thực lớp thứ hai là Email OTP.

<details>
  <summary>SDK send_email_OTP</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)
status, body = client.send_email_otp(dry_run=False)
print(status, body)
```
</details>

Trường hợp không sử dụng phương thức Email OTP, người dùng có thể sử dụng Smart OTP, hướng dẫn <a href="https://developers.dnse.com.vn/docs/guide/intro/authentication#phương-thức-otp">tại đây</a>

#### Xác thực OTP lấy Trading Token

Xác thực mã OTP theo phương thức đã đăng ký để lấy Trading Toke. Đây là thông tin bắt buộc để xác thực quyền giao dịch, có hiệu lực trong 8 tiếng.

<details>
  <summary>SDK create_trading_token</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)
status, body = client.create_trading_token(
    otp_type="email_otp",       // Phương thức xác thực đã đăng ký (email_otp hoặc smart_otp)
    passcode="976981",      // Mã OTP tương ứng phương thức
    dry_run=False,
)
print(status, body)
```
</details>

#### Đặt lệnh

Gửi yêu cầu đặt lệnh giao dịch cơ sở hoặc phái sinh trên tài khoản.

<details>
  <summary>SDK post_order</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)
payload = {
    "accountNo": "0003979888",      // Số tiểu khoản giao dịch
    "symbol": "HPG",               // Mã chứng khoán đặt lệnh
    "side": "NB",                  // NB (mua) hoặc NS (bán)
    "orderType": "LO",             // Loại lệnh theo từng sàn 
    "price": 25950,                // Giá đặt 
    "quantity": 100,               // Khối lượng đặt
    "loanPackageId": 5757
}
status, body = client.post_order(
    market_type="STOCK",           // STOCK (gói vay cơ sở) hoặc DERIVATIVE (gói vay phái sinh)
    payload=payload,
    trading_token="2bccbdf1-32f0-4ea9-9234-b8977baebabc",   // Trading Token lấy từ response POST /create_trading_token
    order_category="NORMAL",
    dry_run=False,
)
print(status, body)
```
</details>

#### Hủy lệnh

Gửi yêu cầu hủy lệnh đã đặt theo `order_id`.

<details>
  <summary>SDK cancel_order</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)
status, body = client.cancel_order(
    account_no="0003979888",     // Số tiểu khoản giao dịch
    market_type="STOCK",     // STOCK (gói vay cơ sở) hoặc DERIVATIVE (gói vay phái sinh)
    order_id="123",         // ID lệnh
    trading_token="2bccbdf1-32f0-4ea9-9234-b8977baebabc",
    order_category="NORMAL",
    dry_run=False,
)
print(status, body)
```
</details>

#### Sửa lệnh

Gửi yêu cầu sửa lệnh đã đặt theo `order_id`.

- Với lệnh cơ sở, khi sửa lệnh thành công đồng nghĩa hủy lệnh cũ và đặt lại lệnh mới, nên người dùng có thể sửa đồng thời giá và khối lượng.
- Với lệnh phái sinh, người dùng chỉ có thể sửa hoặc giá hoặc khối lượng. Khối lượng sửa phải lớn hơn khối lượng đã khớp (nếu có).

<details>
  <summary>SDK put_order</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)
payload = {
    "price": 12500,
    "quantity": 100,
}
status, body = client.replace_order(
    account_no="0003979888",     // Số tiểu khoản giao dịch
    market_type="STOCK",     // STOCK (gói vay cơ sở) hoặc DERIVATIVE (gói vay phái sinh)
    order_id="123",         // ID lệnh
    trading_token="2bccbdf1-32f0-4ea9-9234-b8977baebabc",
    order_category="NORMAL",
    dry_run=False,
)
print(status, body)
```
</details>

#### Đóng vị thế

Gửi yêu cầu đóng vị thế đang mở của phái sinh theo `position_id`.
Đóng vị thế là lệnh đặt ngược chiều với vị thế đang mở, loại lệnh LO với giá đặt là giá trần/sàn của mã tương ứng và khối lượng đặt bằng khối lượng mở của vị thế.

<details>
  <summary>SDK cancel_order</summary>

```python
from dnse import DNSEClient

client = DNSEClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    base_url="https://openapi.dnse.com.vn",
)
status, body = client.close_position(
    position_id="389",
    market_type="DERIVATIVE",
    payload=payload,
    trading_token="replace-with-trading-token",
    dry_run=False,
)
print(status, body)
```
</details>
