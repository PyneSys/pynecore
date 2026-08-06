---
sidebar_position: 2
---

#  Tài khoản giao dịch
---

Mỗi nhà đầu tư khi mở tài khoản tại DNSE sẽ có các thông tin định danh duy nhất trên hệ thống. Một tài khoản có thể sở hữu một hoặc nhiều tiểu khoản giao dịch. Việc nắm rõ cấu trúc này rất quan trọng để người dùng tích hợp và sử dụng API đúng cách.

Các thông tin này được trả ra trong response của <a
href="https://developers.dnse.com.vn/docs/dnse/get-accounts">/get-accounts</a>


```json lines
{
  "name": "Nguyen Hoang A",         // Họ tên khách hàng
  "custodyCode": "064CAA8386",      // Số tài khoản lưu ký tại VSD
  "investorId": "1002003456",       // Mã định danh khách hàng tại DNSE
  "accounts": [                     // Danh sách tiểu khoản thuộc tài khoản
    {
      "id": "0001009212",           // Số tiểu khoản giao dịch
      "dealAccount": true,          // Tiểu khoản theo Deal hoặc không (true/ false)
      "derivativeAccount": true,    // Tiểu khoản được phép giao dịch phái sinh hoặc không (true/ false)
      "derivative": {
        "status": "ACTIVE"          // Trạng thái tiểu khoản phái sinh (ACTIVE: Hoạt động/ INACTIVE: Ngưng hoạt động)
      }
    },
    {
      "id": "0001177757",           // Số tiểu khoản giao dịch            
      "dealAccount": true,          // Tiểu khoản theo Deal hoặc không (true/ false)
      "derivativeAccount": false,    // Tiểu khoản được phép giao dịch phái sinh hoặc không (true/ false)
      "derivative": {}
    }
  ]
}
```
