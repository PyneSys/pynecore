from ..types.na import na_float, na_int
from ..types.pine_types import PyneFloat, PyneInt
from ..types.dividends import Dividends

gross = Dividends("gross")
net = Dividends("net")

future_amount: PyneFloat = na_float
future_ex_date: PyneInt = na_int
future_pay_date: PyneInt = na_int
