from ..types.na import na_float, na_int
from ..types.earnings import Earnings
from ..types.pine_types import PyneFloat, PyneInt

actual = Earnings("actual")
estimate = Earnings("estimate")
standardized = Earnings("standardized")

future_eps: PyneFloat = na_float
future_time: PyneInt = na_int
future_revenue: PyneFloat = na_float
future_period_end_time: PyneInt = na_int
