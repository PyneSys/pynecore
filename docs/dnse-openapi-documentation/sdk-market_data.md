---
sidebar_position: 3
---
# Market Data SDKs

---

DNSE cung cấp [Market Data SDKs](https://github.com/dnse-tech/openapi-sdk) với đa dạng ngôn ngữ và xây dựng sẵn các function phân tách theo từng loại dữ liệu thị trường để khách hàng có thể sẵn sàng sử dụng được ngay.

- SDK Python: https://github.com/dnse-tech/openapi-sdk/tree/main/python/websocket-marketdata

---

### Thiết lập kết nối WebSocket
Thiết lập kết nối tới Websocket server của DNSE để nhận dữ liệu thị trường, khách hàng cần khai báo thông tin API Key và API Secret trong SDKs.

<details>
  <summary>Connect to WebSocket</summary>

```python
import asyncio
from trading_websocket import TradingClient

async def main():
    encoding = "msgpack"  # hoặc "json"
    client = TradingClient(
        api_key="your_api_key",
        api_secret="your_api_secret",
        base_url="wss://ws-openapi.dnse.com.vn",
        encoding=encoding,
    )

    await client.connect()
    print("Connected")

asyncio.run(main())
```
</details>

#### Thông tin mã chứng khoán (Security Definition)
Cung cấp thông tin về giá trần sàn tham chiếu và trạng thái của mã chứng khoán trong ngày giao dịch. Dữ liệu được hệ thống gửi một lần duy nhất vào 8h sáng đầu ngày giao dịch.

<details>
  <summary>SDK Security Definition</summary>

```python
from trading_websocket.models import SecurityDefinition

def handle_security_definition(sec_def: SecurityDefinition):    

await client.subscribe_sec_def(
    symbols=["ACB", "41I1G2000"],   // Mã chứng khoán nhận thông tin
    on_sec_def=handle_security_definition, 
    encoding=encoding
)
```
</details>

#### Dữ liệu khớp lệnh (Trade)

Phân phối Realtime dữ liệu khớp lệnh (tick) của các mã chứng khoán đã đăng ký.

<details>
  <summary>SDK Trade</summary>

```python
from trading_websocket.models import Trade

def handle_trade(trade: Trade):    

await client.subscribe_trades(
    symbols=["ACB", "41I1G2000"],   // Mã chứng khoán nhận thông tin
    on_trade=handle_trade, 
    encoding=encoding
)
```

</details>

#### Dữ liệu khớp lệnh mở rộng (Trade Extra)

Phân phối Realtime dữ liệu khớp lệnh (tick) và bổ sung thêm một số thông tin do DNSE tự tổng hợp (mua bán chủ động, giá khớp trung bình). Trong trường hợp người dùng không sử dụng đến các thông tin này thì nên dùng function Trade đơn thuần để tối ưu hơn về tốc độ nhận dữ liệu.

<details>
  <summary>SDK Trade Extra</summary>

```python
from trading_websocket.models import TradeExtra

def handle_trade_extra(trade: TradeExtra):

await client.subscribe_trade_extra(
    symbols=["ACB", "41I1G2000"],   // Mã chứng khoán nhận thông tin
    on_trade_extra=handle_trade_extra,
    encoding=encoding,
)
```
</details>

#### Độ sâu thị trường (Quote)

Phân phối Realtime thông tin về các mức giá chào mua và chào bán tốt nhất của mã chứng khoán (bid-ask).
Dữ liệu phản ánh trạng thái cung – cầu tại thời điểm hiện tại.

<details>
  <summary>SDK quote</summary>

```python
from trading_websocket.models import Quote

def handle_quote(quote: Quote):

await client.subscribe_quotes(
    symbols=["ACB", "41I1G2000"],   // Mã chứng khoán nhận thông tin
    on_quote=handle_quote,
    encoding=encoding,
)
```
</details>

#### OHLC

Phân phối thông tin nến theo khung thời gian thực (open, high, low, close, volume) cho Cổ phiếu (stock), Phái sinh (derivative) và Chỉ số thị trường (index). Áp dụng cho nhiều khung thời gian (resolution).

<details>
  <summary>SDK OHLC</summary>

```python
from trading_websocket.models import OHLC

def handle_ohlc(ohlc: Ohlc):

# internal 1 3 5 15 30 1H 1D 1W
await client.subscribe_ohlc(
    symbols=["HPG", "41I1G2000"],
    resolution="1",
    on_bar=handle_bar,
    encoding=encoding,
)
```
</details>

#### Giá khớp dự kiến (Expected Price)

Phân phối thông tin giá đóng cửa (kết thúc phiên), giá khớp dự kiến và khối lượng khớp dự kiến của mã chứng khoán trong các phiên giao dịch khớp lệnh định kỳ ATO và ATC.

<details>
  <summary>SDK Expected Price</summary>

```python
from trading_websocket.models import ExpectedPrice

def handle_expected_price(expected_price: ExpectedPrice):

await client.subscribe_expected_price(
    symbols=["HPG", "41I1G2000"],
    on_expected_price=handle_expected_price,
    encoding=encoding
)
```
</details>

#### Chỉ số thị trường (Market Index)

Cung cấp thông tin chỉ số thị trường bao gồm giá trị chỉ số, mức thay đổi, độ rộng thị trường (số mã tăng/giảm/đi ngang) và thanh khoản. Dữ liệu được cập nhật liên tục trong phiên giao dịch.

<details>
  <summary>SDK Market Index</summary>

```python
from trading_websocket.models import MarketIndex

def handle_market_index(data: MarketIndex):

await client.subscribe_market_index(market_index='HNX', on_market_index=handle_market_index, encoding=encoding)
```
</details>

#### Giao dịch nhà đầu tư nước ngoài (Foreign Investor)

Cung cấp dữ liệu giao dịch của nhà đầu tư nước ngoài theo từng mã chứng khoán, bao gồm khối lượng và giá trị mua/bán, tổng lũy kế trong ngày và room còn lại. Dữ liệu được cập nhật trong phiên giao dịch khi có thay đổi.

<details>
  <summary>SDK Foreign Investor</summary>

```python
from trading_websocket.models import ForeignInvestor

def handle_foreign_trading(data: ForeignInvestor):

await client.subscribe_foreign_trading(["SHS", "FPT"], board_id="G1", on_trade=handle_foreign_trading, encoding=encoding)
```
</details>
