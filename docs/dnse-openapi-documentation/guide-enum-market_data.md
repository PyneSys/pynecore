# Enums Dữ liệu thị trường

### boardId
*Định danh bảng giao dịch*

| Enums (BoardID) | Kiểu dữ liệu | Mô tả                        |
|-----------------|--------------|------------------------------|
| UNSPECIFIED     | string       | Không xác định               |
| AL              | string       | Tất cả bảng                  |
| G1              | string       | Lô chẵn                      |
| G3              | string       | Board phiên sau giờ (PLO)    |
| G4              | string       | Lô lẻ                        |
| T1              | string       | Thỏa thuận lô chẵn 9h-14h45  |
| T3              | string       | Thỏa thuận lô chẵn 14h45-15h |
| T4              | string       | Thỏa thuận lô lẻ 9h-14h45    |
| T6              | string       | Thỏa thuận lô lẻ 14h45-15h   |

### eventId
*Mã sự kiện chuyển trạng thái phiên giao dịch*

| Enums (EventID) | Kiểu dữ liệu | Mô tả                                                         |
|-----------------|--------------|---------------------------------------------------------------|
| AA1             | String       | Bắt đầu phiên khớp lệnh định kỳ mở cửa                        |
| AB1             | String       | Bắt đầu phiên khớp lệnh liên tục                              |
| AB2             | String       | Kết thúc giao dịch của bảng                                   |
| AC2             | String       | Thực hiện khớp lệnh định kỳ đóng cửa                          |
| AD1             | String       | Bắt đầu nhận lệnh                                             |
| AD2             | String       | Thực hiện khớp lệnh định kỳ                                   |
| AW8             | String       | Nghỉ trưa                                                     |
| AW9             | String       | Tiếp tục giao dịch sau nghỉ trưa                              |
| AX1             | String       | Bảng giao dịch bắt đầu hoạt động                              |
| BB1             | String       | Bắt đầu phiên khớp lệnh liên tục cho chứng khoán thông thường |
| BC1             | String       | Bắt đầu phiên khớp lệnh định kỳ đóng cửa                      |
| CC1             | String       | Bắt đầu phiên PCA đóng cửa                                    |
| CD1             | String       | Bắt đầu phiên PCA                                             |
| CD3             | String       | Thực hiện khớp lệnh PCA                                       |

---

### indexTypeCode
*Loại chỉ số thị trường*

| Enums (IndexTypeCode) | Kiểu dữ liệu | Mô tả                                                           |
|-----------------------|--------------|-----------------------------------------------------------------|
| 001                   | string       | VNINDEX (Chỉ số sàn HOSE)                                       |
| 002                   | string       | HNX (Chỉ số sàn HNX)                                            |
| 100                   | string       | HNX30 (Top 30 cổ phiếu sàn HNX)                                 |
| 101                   | string       | VN30 (Top 30 cổ phiếu sàn HOSE)                                 |
| 104                   | string       | VN100 (Top 100 cổ phiếu sàn HOSE)                               |
| 151                   | string       | VNXALLSHARE (Chỉ số các cổ phiếu chọn lọc sàn HOSE)             |
| 301                   | string       | UPCOM (Chỉ số sàn UPCOM)                                        |
| 504                   | string       | VNMITECH (Chỉ số nhóm cổ phiếu công nghệ)                       |
| 505                   | string       | VN50GROWTH (Chỉ số nhóm 50 cổ phiếu tăng trưởng sàn HOSE)       |
| 506                   | string       | VNDIVIDEND (Chỉ số nhóm cổ phiếu có tỷ suất cổ tức tăng trưởng) |

---

### marketId
*Mã thị trường*

| Enums (MarketID) | Kiểu dữ liệu | Mô tả                           |
|------------------|--------------|---------------------------------|
| UNSPECIFIED      | string       | Không xác định                  |
| DVX              | string       | Phái sinh sàn HNX               |
| HCX              | string       | Trái phiếu doanh nghiệp sàn HNX |
| STO              | string       | Cổ phiếu sàn HOSE               |
| STX              | string       | Cổ phiếu sàn HNX                |
| UPX              | string       | Cổ phiếu sàn Upcom              |

---

### marketIndex (IndexName)
*Tên chỉ số thị trường*

| Enums (MarketIndex) | Kiểu dữ liệu | Mô tả                                              |
|---------------------|--------------|----------------------------------------------------|
| VNINDEX             | string       | Chỉ số sàn HOSE                                    |
| HNX                 | string       | Chỉ số sàn HNX                                     |
| HNX30               | string       | Chỉ số Top 30 cổ phiếu sàn HNX                     |
| VN30                | string       | Chỉ số Top 30 cổ phiếu sàn HOSE                    |
| VN100               | string       | Chỉ số Top 100 cổ phiếu sàn HOSE                   |
| VNXALLSHARE         | string       | Chỉ số các cổ phiếu chọn lọc sàn HOSE              |
| UPCOM               | string       | Chỉ số sàn UPCOM                                   |
| VNMITECH            | string       | Chỉ số nhóm cổ phiếu công nghệ                     |
| VN50GROWTH          | string       | Chỉ số nhóm 50 cổ phiếu tăng trưởng sàn HOSE       |
| VNDIVIDEND          | string       | Chỉ số nhóm cổ phiếu có tỷ suất cổ tức tăng trưởng |

---

### productGrpId
*Nhóm sản phẩm theo thị trường*

| Enums (ProductGrpID) | Kiểu dữ liệu | Mô tả                         |
|----------------------|--------------|-------------------------------|
| UNSPECIFIED          | string       | Không xác định                |
| FBX                  | string       | Hợp đồng tương lai Trái phiếu |
| FIO                  | string       | Hợp đồng tương lai Chỉ số     |
| HCX                  | string       | Trái phiếu Doanh nghiệp HNX   |
| STO                  | string       | Cổ phiếu sàn HOSE             |
| STX                  | string       | Cổ phiếu sàn HNX              |
| UPX                  | string       | Cổ phiếu sàn Upcom            |

---

### resolution
*Khung thời gian nến*

| Enums (Resolution) | Kiểu dữ liệu | Mô tả   |
|--------------------|-------------|--------|
| 1                  | string      | 1 phút |
| 3                  | string      | 3 phút |
| 5                  | string      | 5 phút |
| 15                 | string      | 15 phút |
| 30                 | string      | 30 phút |
| 1H                 | string      | 1 giờ  |
| 1D                 | string      | 1 ngày |
| 1W                 | string      | 1 tuần |

---

### securityGroupId
*Định danh nhóm chứng khoán*

| Enums (SecurityGroupID) | Kiểu dữ liệu | Mô tả                   |
|-------------------------|--------------|-------------------------|
| UNSPECIFIED             | string       | Không xác định          |
| BS                      | string       | Trái phiếu doanh nghiệp |
| EF                      | string       | Quỹ ETF                 |
| EW                      | string       | Chứng quyền             |
| FU                      | string       | Hợp đồng tương lai      |
| SR                      | string       | Quyền mua               |
| ST                      | string       | Cổ phiếu                |

---

### securityStatus
*Trạng thái của mã chứng khoán*

| Enums (SecurityStatus) | Kiểu dữ liệu | Mô tả                 |
|------------------------|-------------|----------------------|
| HALT                   | string      | Ngừng giao dịch       |
| NO_HALT                | string      | Không ngừng giao dịch |

---

### side
*Chiều xác định mua/bán chủ động*

| Enums (Side) | Kiểu dữ liệu | Mô tả          |
|--------------|--------------|----------------|
| UNSPECIFIED  | string       | Không xác định |
| BUY          | string       | Mua chủ động   |
| SELL         | string       | Bán chủ động   |

---

### symbolAdminStatusCode
*Trạng thái quản lý hành chính mã chứng khoán*

| Enums (SymbolAdminStatusCode) | Kiểu dữ liệu | Mô tả                          |
|-------------------------------|--------------|--------------------------------|
| UNSPECIFIED                   | string       | Không xác định                 |
| CR                            | string       | Kiểm soát và hạn chế giao dịch |
| CTR                           | string       | Kiểm soát                      |
| NRM                           | string       | Bình thường                    |
| RES                           | string       | Hạn chế giao dịch              |
| WFR                           | string       | Cảnh báo do vi phạm BCTC       |
| WID                           | string       | Cảnh báo do vi phạm CBTT       |
| WOV                           | string       | Cảnh báo vi phạm khác          |

---

### symbolTradingMethodStatusCode
*Trạng thái cơ chế giao dịch mã chứng khoán*

| Enums (SymbolTradingMethodStatusCode) | Kiểu dữ liệu | Mô tả                         |
|---------------------------------------|--------------|-------------------------------|
| UNSPECIFIED                           | string       | Không xác định                |
| NRM                                   | string       | Bình thường                   |
| NWE                                   | string       | Niêm yết mới biên độ đặc biệt |
| NWN                                   | string       | Niêm yết mới biên độ thường   |
| SLS                                   | string       | Giao dịch đặc biệt sau halt   |
| SNE                                   | string       | Giao dịch đặc biệt            |

---

### symbolTradingSanctionStatusCode
*Tình trạng giao dịch của mã chứng khoán*

| Enums (SymbolTradingSanctionStatusCode) | Kiểu dữ liệu | Mô tả                      |
|-----------------------------------------|--------------|----------------------------|
| UNSPECIFIED                             | string       | Không xác định             |
| NRM                                     | string       | Bình thường                |
| SUS                                     | string       | Tạm ngừng                  |
| DTL                                     | string       | Hủy niêm yết chuyển sàn    |
| TFR                                     | string       | Ngưng giao dịch do hạn chế |

---

### tradingSessionId
*Mã phiên giao dịch (theo từng mã)*

| Enums (TradingSessionID) | Kiểu dữ liệu | Mô tả          |
|--------------------------|--------------|----------------|
| UNSPECIFIED              | string       | Không xác định |
| 10                       | string       | ATO            |
| 30                       | string       | ATC            |
| 40                       | string       | Phiên liên tục |
| 80                       | string       | PCA mã halt    |
| 99                       | string       | Đóng bảng      |

---

### type
*Loại nhóm thị trường*

| Enums (Type) | Kiểu dữ liệu | Mô tả     |
|--------------|--------------|-----------|
| STOCK        | string       | Cổ phiếu  |
| DERIVATIVE   | string       | Phái sinh |
| INDEX        | string       | Chỉ số    |