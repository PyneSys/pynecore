<!--
---
weight: 471
title: "currency"
description: "Currency code constants (USD, EUR, etc.)"
icon: "payments"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["currency", "library", "reference"]
---
-->

# currency

The `currency` namespace provides a set of currency code constants used to identify currencies in strategy and indicator configurations. These constants are `Currency` objects and can be passed to parameters like `currency` in `strategy()` declarations.

## Quick Example

```python
from pynecore.lib import close, script, strategy, currency

@script.strategy(title="Currency Example", currency=currency.USD)
def main():
    if close > 100:
        strategy.entry("Long", strategy.long)
```

## Constants

All constants are of type `Currency`.

| Constant           | Description                  |
|--------------------|------------------------------|
| `currency.AED`     | Arab Emirates Dirham         |
| `currency.ARS`     | Argentine Peso               |
| `currency.AUD`     | Australian Dollar            |
| `currency.BDT`     | Bangladeshi Taka             |
| `currency.BHD`     | Bahraini Dinar               |
| `currency.BRL`     | Brazilian Real               |
| `currency.BTC`     | Bitcoin                      |
| `currency.CAD`     | Canadian Dollar              |
| `currency.CHF`     | Swiss Franc                  |
| `currency.CLP`     | Chilean Peso                 |
| `currency.CNY`     | Chinese Yuan                 |
| `currency.COP`     | Colombian Peso               |
| `currency.CZK`     | Czech Koruna                 |
| `currency.DKK`     | Danish Krone                 |
| `currency.EGP`     | Egyptian Pound               |
| `currency.ETH`     | Ethereum                     |
| `currency.EUR`     | Euro                         |
| `currency.GBP`     | Pound Sterling               |
| `currency.HKD`     | Hong Kong Dollar             |
| `currency.HUF`     | Hungarian Forint             |
| `currency.IDR`     | Indonesian Rupiah            |
| `currency.ILS`     | Israeli New Shekel           |
| `currency.INR`     | Indian Rupee                 |
| `currency.ISK`     | Icelandic Krona              |
| `currency.JPY`     | Japanese Yen                 |
| `currency.KES`     | Kenyan Shilling              |
| `currency.KRW`     | South Korean Won             |
| `currency.KWD`     | Kuwaiti Dinar                |
| `currency.LKR`     | Sri Lankan Rupee             |
| `currency.MAD`     | Moroccan Dirham              |
| `currency.MXN`     | Mexican Peso                 |
| `currency.MYR`     | Malaysian Ringgit            |
| `currency.NGN`     | Nigerian Naira               |
| `currency.NOK`     | Norwegian Krone              |
| `currency.NONE`    | Unspecified currency         |
| `currency.NZD`     | New Zealand Dollar           |
| `currency.PEN`     | Peruvian Sol                 |
| `currency.PHP`     | Philippine Peso              |
| `currency.PKR`     | Pakistani Rupee              |
| `currency.PLN`     | Polish Zloty                 |
| `currency.QAR`     | Qatari Riyal                 |
| `currency.RON`     | Romanian Leu                 |
| `currency.RSD`     | Serbian Dinar                |
| `currency.RUB`     | Russian Ruble                |
| `currency.SAR`     | Saudi Riyal                  |
| `currency.SEK`     | Swedish Krona                |
| `currency.SGD`     | Singapore Dollar             |
| `currency.THB`     | Thai Baht                    |
| `currency.TND`     | Tunisian Dinar               |
| `currency.TRY`     | Turkish Lira                 |
| `currency.TWD`     | New Taiwan Dollar            |
| `currency.USD`     | United States Dollar         |
| `currency.USDT`    | Tether                       |
| `currency.VES`     | Venezuelan Bolivar           |
| `currency.VND`     | Vietnamese Dong              |
| `currency.ZAR`     | South African Rand           |