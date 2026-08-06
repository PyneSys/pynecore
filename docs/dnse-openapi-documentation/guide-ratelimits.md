---
sidebar_position: 2
---


# Rate Limits

DNSE OpenAPI áp dụng rate limit theo từng APIKey và từng Endpoint.

Rate limit được định nghĩa bởi:

- Rate: tổng số request trong 1 giờ
- Quota: tổng số request trong 24 giờ (1 ngày)

### Normal Rate Limits

| Tên API                              | Endpoint                                                                                                                                    | Rate / giờ | Quota / ngày |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|------------|--------------|
| Thông tin tiền                       | <a href="https://developers.dnse.com.vn/docs/dnse/get-account-balances">/Get Account Balance</a>                  | 10,000     | 100,000      |
| Tài khoản giao dịch                  | <a href="https://developers.dnse.com.vn/docs/dnse/get-accounts">/Get Accounts</a>                                 | 1,000      | 10,000       |
| Danh sách gói vay                    | <a href="https://developers.dnse.com.vn/docs/dnse/get-loan-packages">/Get Loan Packages</a>                       | 10,000     | 100,000      |
| Sức mua, sức bán                     | <a href="https://developers.dnse.com.vn/docs/dnse/get-ppse">/Get PPSE</a>                                         | 10,000     | 100,000      |
| Sổ lệnh                              | <a href="https://developers.dnse.com.vn/docs/dnse/get-orders">/Get Orders</a>                                     | 100,000    | 1,000,000    |
| Lịch sử lệnh                         | <a href="https://developers.dnse.com.vn/docs/dnse/get-orders-history">/Get Order History</a>                      | 10,000     | 100,000      |
| Chi tiết lệnh theo ID                | <a href="https://developers.dnse.com.vn/docs/dnse/get-order-detail">/Get Order Detail</a>                         | 100,000    | 1,000,000    |
| Chi tiết lệnh theo ID                | <a href="https://developers.dnse.com.vn/docs/dnse/get-order-detail">/Get Order Detail</a>                         | 10,000     | 100,000      |
| Vị thế nắm giữ                       | <a href="https://developers.dnse.com.vn/docs/dnse/get-accounts-account-no-positions">/Get Positions</a>           | 10,000     | 100,000      |
| Chi tiết trạng thái lệnh             | <a href="https://developers.dnse.com.vn/docs/dnse/get-executions">/Get Executions Order by ID</a>                 | 10,000     | 100,000      |
| Lịch sử sự kiện quyền                | <a href="https://developers.dnse.com.vn/docs/dnse/get-corporate-action-history">/Get Corporate Action History</a> | 1,000      | 10,000       |
| Gửi Email OTP                        | <a href="https://developers.dnse.com.vn/docs/dnse/send-email-otp">/Send Email OTP</a>                             | 100        | 1,000        |
| Xác thực OTP                         | <a href="https://developers.dnse.com.vn/docs/dnse/2-fa-verification">/Create Trading Token</a>                    | 100        | 1,000        |
| Đặt lệnh                             | <a href="https://developers.dnse.com.vn/docs/dnse/place-order">/Place Order</a>                                   | 50,000     | 100,000      |
| Sửa lệnh                             | <a href="https://developers.dnse.com.vn/docs/dnse/replace-order">/Replace Order</a>                               | 50,000     | 100,000      |
| Hủy lệnh                             | <a href="https://developers.dnse.com.vn/docs/dnse/cancel-order">/Cancel Order</a>                                 | 50,000     | 100,000      |
| Đóng vị thế                          | <a href="https://developers.dnse.com.vn/docs/dnse/post-accounts-positions-position-id-close">/Close Position</a>  | 50,000     | 100,000      |
| Cấu hình chốt lời, cắt lỗ của vị thế | <a href="https://developers.dnse.com.vn/docs/dnse/get-pnl-configs-position">/Get PNL Configs Position</a>         | 10,000     | 100,000      |
| Cài đặt chốt lời, cắt lỗ của vị thế  | <a href="https://developers.dnse.com.vn/docs/dnse//post-pnl-configs-position">/Post PNL Configs Position</a>      | 10,000     | 100,000      |
| Thông tin giao dịch chứng khoán      | <a href="https://developers.dnse.com.vn/docs/dnse/get-secdef">/Get Security Definition</a>                        | 1,000      | 10,000       |
| Chi tiết mã chứng khoán              | <a href="https://developers.dnse.com.vn/docs/dnse/get-instruments">/Get Instruments</a>                           | 10,000     | 100,000      |
| Lịch sử OHLC                         | <a href="https://developers.dnse.com.vn/docs/dnse/get-ohlc-history">/Get OHLC</a>                                 | 50,000     | 100,000      |
| Lịch sử khớp lệnh                    | <a href="https://developers.dnse.com.vn/docs/dnse/get-price-symbol-trades">/Get Trades</a>                        | 10,000     | 100,000      |
| Dữ liệu khớp gần nhất                | <a href="https://developers.dnse.com.vn/docs/dnse/get-price-symbol-trades-latest">/Get Latest Trades</a>          | 10,000     | 100,000      |
| Lịch sử bid/ask                      | <a href="https://developers.dnse.com.vn/docs/dnse/get-quotes">/Get Quotes</a>                                     | 10,000     | 100,000      |
| Dữ liệu bid/ask gần nhất             | <a href="https://developers.dnse.com.vn/docs/dnse/get-latest-quotes">/Get Latest Quotes</a>                       | 10,000     | 100,000      |
| Giá đóng cửa                         | <a href="https://developers.dnse.com.vn/docs/dnse/get-price-symbol-close">/Get Closed Price</a>                   | 1,000      | 10,000       |
| Ngày làm việc                        | <a href="https://developers.dnse.com.vn/docs/dnse/get-market-working-dates">/Get Working Dates</a>                | 1,000      | 10,000       |
| Danh sách tiểu khoản quản lý         | <a href="https://developers.dnse.com.vn/docs/dnse/get-brokers-accounts-care-by">/Get List Care By</a>             | 1000       | 10,000       |

### Lưu ý

DNSE có thể cung cấp thông tin về giới hạn sử dụng API thông qua các header trong response:

| Header                | Ý nghĩa                                |
|-----------------------|----------------------------------------|
| X-RateLimit-Limit     | Số lượng request tối đa được phép      |
| X-RateLimit-Remaining | Số request còn lại trong ngày hiện tại |
| X-RateLimit-Reset     | Thời điểm giới hạn được làm mới        |


Khi vượt quá giới hạn, hệ thống sẽ trả về **HTTP 429 (Too Many Requests)**

```json lines
429
Too Many Requests
{
    "error": "Rate Limit Exceeded"
}
```

### Khuyến nghị

- Phân bổ tần suất gọi API hợp lý trong từng khoảng thời gian
- Hạn chế gọi lặp lại các dữ liệu ít thay đổi bằng cách cache
- Ưu tiên sử dụng các API hỗ trợ xử lý nhiều dữ liệu trong một lần gọi (nếu có)
- Theo dõi số lượng request còn lại để chủ động điều chỉnh hành vi gọi API
