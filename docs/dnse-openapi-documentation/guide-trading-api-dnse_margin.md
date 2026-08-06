---
sidebar_position: 3
---

# Margin tại DNSE
---

Khác với cách quản trị rủi ro trên tài khoản tổng, với Margin tại DNSE - mỗi một Deal (bao gồm một mã chứng khoán và một gói vay ký quỹ) khác nhau của khách hàng sẽ được quản trị tách bạch:

- Các Deals khác nhau về tỷ lệ vay được quản lý tách biệt, danh mục của khách hàng có thể gồm nhiều Deals vay khác nhau.
- Cách tính giá trung bình, giá hòa vốn của mỗi Deal (đã bao gồm lãi vay và các chi phí khác) rõ ràng, chính xác hơn so với giá vốn truyền thống.
- Tỷ lệ ký quỹ của mỗi Deal được kiểm soát độc lập. DNSE sẽ chỉ yêu cầu ký quỹ bổ sung hoặc bán giải chấp Deal có tỷ lệ xuống dưới mức cảnh báo mà không ảnh hưởng tới các Deal an toàn khác.

Đây là sự khác biệt mà DNSE xây dựng để khách hàng của mình quản lý tài sản minh bạch hơn (mô hình Isolated Margin)

### Gói vay (Loan packages)

Gói vay là khái niệm đại diện cho chính sách sản phẩm tại DNSE. Mỗi gói vay quy định các điều kiện áp dụng cho giao dịch, bao gồm: phí giao dịch, tỷ lệ vay, tỷ lệ ký quỹ, lãi suất vay, kỳ hạn và một số thông tin khác.

Các gói vay được áp dụng cho từng mã chứng khoán được trả về trong response Endpoint <a href="https://developers.dnse.com.vn/docs/dnse/get-loan-packages">/Danh sách gói vay</a>

**Danh sách gói vay cơ sở**

VD: Danh sách gói vay cho mã HPG

```json lines
{
  "symbol": "HPG",         // Mã chứng khoán
  "marketType": "STOCK",   // Loại thị trường (STOCK: cơ sở / DERIVATIVE: phái sinh)
  "loanPackages": [       // Danh sách gói vay 
    {
      "id": 1775,         // Mã gói vay
      "name": "Mana RocketX LS 5.99% HPG - KQ 100%",    // Tên gói vay 
      "initialRate": 1,             // Tỷ lệ ký quỹ ban đầu 
      "interestRate": 0.0599,       // Tỷ lệ lãi vay (nếu phát sinh ứng sức mua, nợ margin)
      "liquidRate": 0.3,            // Tỷ lệ xử lý (force sell)
      "maintenanceRate": 0.4,       // Tỷ lệ duy trì (call margin)
      "type": "M",              // Loại gói vay (M: gói vay margin/ N: gói tiền mặt)
      "brokerFirmBuyingFeeRate": 0,     //  Phí mua chứng khoán cơ sở DNSE thu
      "brokerFirmSellingFeeRate": 0     //  Phí bán chứng khoán cơ sở DNSE thu
    },
    {
      "id": 1769,       // Mã gói vay
      "name": "Rocket X LS 5.99% HPG - KQ 50%",    // Tên gói vay 
      "initialRate": 0.5,           // Tỷ lệ ký quỹ ban đầu 
      "interestRate": 0.0599,       // Tỷ lệ lãi vay (nếu phát sinh ứng sức mua, nợ margin)
      "liquidRate": 0.3,            // Tỷ lệ xử lý (force sell)
      "maintenanceRate": 0.4,       // Tỷ lệ duy trì (call margin)
      "type": "M",                  // Loại gói vay (M: gói vay margin/ N: gói tiền mặt)
      "brokerFirmBuyingFeeRate": 0.00045,     //  Phí mua chứng khoán cơ sở DNSE thu
      "brokerFirmSellingFeeRate": 0.00045     //  Phí bán chứng khoán cơ sở DNSE thu
    }  
  ]
}
```

VD: Danh sách gói vay cho mã VGI

```json lines
{
  "symbol": "VGI",      // Mã chứng khoán
  "marketType": "STOCK",      // Loại thị trường (STOCK: cơ sở / DERIVATIVE: phái sinh)
  "loanPackages": [     // Danh sách gói vay 
    {
      "id": 1036,   // Mã gói vay
      "name": "GD tiền mặt",    // Tên gói vay 
      "type": "N",      // Loại gói vay (M: gói vay margin/ N: gói tiền mặt)
      "brokerFirmBuyingFeeRate": 0,     //  Phí mua chứng khoán cơ sở DNSE thu
      "brokerFirmSellingFeeRate": 0     //  Phí bán chứng khoán cơ sở DNSE thu
    }
  ]
}
```

Đối với giao dịch chứng khoán cơ sở, hệ thống sẽ trả về tối đa 2 gói vay mà người dùng có thể sử dụng để đặt lệnh cho mã chứng khoán truy vấn, bao gồm:

- Gói vay giao dịch tiền mặt:

    + Tỷ lệ ký quỹ tiền mặt 100% (`initialRate` = 1)
    + Dành cho giao dịch không sử dụng đòn bẩy tiền vay
- Gói vay ký quỹ (margin) cơ bản:

    + Dành cho giao dịch có sử dụng đòn bẩy tiền vay (`initialRate` ≠ 1)

**Danh sách gói vay phái sinh**

VD: Danh sách gói vay cho mã 41I1G1000

```json lines
{
  "symbolType": "VN30F1M",           // Mã giao dịch Hợp đồng tương lai
  "marketType": "DERIVATIVE",       // Loại thị trường 
  "loanPackages": [                 // Danh sách gói vay 
    {
      "id": 1306,                   // Mã gói vay
      "name": "Gói giao dịch 01",   // Tên gói vay 
      "initialRate": 0.1848,        // Tỷ lệ ký quỹ ban đầu
      "maintenanceRate": 0.1771,    // Tỷ lệ duy trì (call margin)
      "liquidRate": 0.1735,         // Tỷ lệ xử lý (force sell)
      "tradingFee": {               // Chính sách phí giao dịch (dành cho phái sinh)
        "id": 1304,                 // ID chính sách phí    
        "name": "Miễn phí",         // Tên phí
        "scope": "PRODUCT",         // Phạm vi áp dụng chính sách
        "channel": "ALL",           // Kênh giao dịch áp dụng
        "schemaType": "FIXED",      // Loại phí cố định
        "createdDate": "2023-02-02T04:22:56.199278Z",       // Thời điểm tạo chính sách
        "modifiedDate": "2023-02-02T04:22:56.199278Z",      // Thời điểm cập nhật chính sách
        "fixedTradingFee": 2000,        // Phí giao dịch 1 hợp đồng
        "fixedDailyCloseTradingFee": 2000      // Phí giao dịch 1 hợp đồng đóng luôn trong ngày 
      }
    }
  ]
}
```

Với sản phẩm phái sinh, thông thường tài khoản giao dịch của khách hàng chỉ được gắn một gói vay với một bộ tỷ lệ ký quỹ, duy trì và xử lý duy nhất áp dụng cho tất cả các mã phái sinh.

### Deal

Deals hay còn có thể hiểu là danh mục tài sản của khách hàng. Một Deal được hình thành bởi 1 mã chứng khoán và 1 gói vay:

- Với cùng một mã có thể có nhiều Deals độc lập nếu bạn mua cùng mã nhưng chọn gói vay khác nhau
- Việc cho vay, thu nợ, quản trị rủi ro được thực hiện trên từng Deal

Ví dụ:

- Lần 1: Mua 100cp HPG với gói vay “GD Tiền mặt", hệ thống sẽ tạo 1 Deal HPG Tiền mặt, tỷ lệ Ký quỹ 100%
- Lần 2: Mua 500cp HPG với gói vay margin “Tiền mặt 50%”, hệ thống sẽ tạo Deal mới HPG Tiền mặt 50% (khác với Deal bên trên), được hiểu là khách hàng ký quỹ 50% và sử dụng tiền vay 50% tính trên tổng số tiền khớp lệnh mua.
- Lần 3: Mua 200cp HPG với gói vay “Tiền mặt 100%”, do cùng gói vay với lần thứ 1, nên hệ thống gộp khối lượng mua thêm vào Deal HPG Tiền mặt 100%; tổng khối lượng sau mua là 300cp.

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-openapi/doc/img/deal.png)](https://cdn.dnse.com.vn/dnse-openapi/doc/img/deal.png)
</div>

Khách hàng có thể tìm hiểu thêm về sản phẩm giao dịch ký quỹ theo DEAL [tại đây.](https://hdsd.dnse.com.vn/san-pham-dich-vu/sp-giao-dich-ky-quy-theo-deal/thong-tin-chung)