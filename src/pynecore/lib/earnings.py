from ..types.na import NA, na_float, na_int
from ..types.earnings import Earnings
from ..types.pine_types import PyneInt

actual = Earnings("actual")
estimate = Earnings("estimate")
standardized = Earnings("standardized")

future_eps = na_float
future_time: PyneInt = na_int
future_revenue = na_float
future_period_end_time: PyneInt = na_int
