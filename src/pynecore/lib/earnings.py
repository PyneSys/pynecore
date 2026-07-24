from ..types.na import NA, na_float
from ..types.earnings import Earnings

actual = Earnings("actual")
estimate = Earnings("estimate")
standardized = Earnings("standardized")

future_eps = na_float
future_time = NA(int)
future_revenue = na_float
future_period_end_time = NA(int)
