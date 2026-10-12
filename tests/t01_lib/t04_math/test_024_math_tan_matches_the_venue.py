"""
``math.tan`` reproduces the venue's tangent, which is NOT correctly rounded.

The venue evaluates a runtime ``math.tan`` with the JVM's x86 ``Math.tan`` intrinsic
(the Intel LIBM stub): a polynomial for |x| below about 0.1, otherwise a 32-breakpoint
table polynomial on x - n*pi/32 joined with an accurate reciprocal of x modulo pi/2
near the poles, and a fixed-point reduction by 2/pi from |x| = 12800 up. It lands
within an ulp, but on about one argument in 30 on a different double than StrictMath
(fdlibm), which is what a constant ``math.tan`` folds to at parse time, and macOS's
``tan()`` differs from it far more often.

MEASURED on TradingView, CAPITALCOM:EURUSD@60: two probes logging the ulp distance of
a series ``math.tan`` from the StrictMath value, over arguments from the polynomial
range, both table halves, the poles and the huge-argument reduction up to 1e15.
"""
from pynecore.core import pine_math
from pynecore.lib import math

# (argument, venue) where the venue is one ulp away from StrictMath
__test_helper_STRICTMATH_DIFFERS = (
    (0.055729, 0.05578676467818975),
    (-0.057433, -0.0574962319617466),
    (-0.05604, -0.056098737983614304),
    (-0.092595, -0.09286054210996807),
    (0.051696, 0.05174210139618537),
    (0.316603, 0.32762357511400503),
    (0.348064, 0.36283607758921405),
    (0.642341, 0.7481889049984319),
    (-0.645849, -0.7536750582412258),
    (0.549514, 0.6124367259948392),
    (-0.74377, -0.9200266175036002),
    (-0.848539, -1.1349841141190409),
    (-1.053074, -1.7557985883658425),
    (-0.871834, -1.1897452355226001),
    (1.226874, 2.7920781894914417),
    (0.994949, 1.5402403298306104),
    (2.437222, -0.8497874474466812),
    (-2.0891, 1.7534274705648756),
    (2.249322, -1.2403472471115908),
    (-2.283145, 1.1579156277266793),
    (2.442407, -0.8408972501125281),
    (-5.370402, 1.2937849243067092),
    (-162.630362, 0.899350211623792),
    (-44.978413, -1.5441824308593948),
    (104.597651, 1.327235152488367),
    (-158.106871, -1.654871747041121),
    (-6905.769492, -0.6115094940627867),
    (5657.128116, -1.2102878076434649),
    (1053.444871, 1.5969249829253502),
    (3989.235537, -0.6654113852953855),
    (7355.126821, 0.7735746317962332),
    (820971.065607, 0.7579865777357501),
    (-409942.530892, 0.9369946970503816),
    (-108562.125919, -3.0113966240777814),
    (-210527.903647, -0.37087801517189156),
    (228492.90028, -6.433796209120217),
    (-936649000000.0, -0.5552852243696546),
    (414204000000.0, -0.4296414222838911),
    (380400000000.0, 1.5116869830509971),
    (612196000000.0, 0.5995432825674097),
    (926321000000.0, -1.8863971939904025),
    (997337000000000.0, 0.9213623748450263),
    (-245977000000000.0, -1.193542509796435),
    (-221724000000000.0, 0.7129221706158476),
    (544762000000000.0, 0.607662628948073),
    (-380793000000000.0, -0.7322549718579016),
)

# (argument, venue) where the venue and StrictMath agree
__test_helper_STRICTMATH_AGREES = (
    (0.240429, 0.2451714416899434),
    (0.083802, 0.08399872684840834),
    (0.689623, 0.8247024976452325),
    (1.365099, 4.79275192439207),
    (0.813417, 1.0576685712746332),
    (1.716336, -6.822396867745151),
    (3.064605, -0.07714011954885727),
    (2.914216, -0.23137788824011446),
    (26.139668, 1.581394660113928),
    (13.584221, 1.620309939992818),
    (19.390713, 0.6010035506395184),
    (47273.065315, 19.92748305742145),
    (18654.683575, -0.09387633721705549),
    (98139.500407, -0.8638823749183878),
    (988520067.678591, 0.8994489534611113),
    (631397707.744179, 28.297400422122884),
    (714746414.782797, -0.19146179679050535),
)


def __test_tan_matches_the_venue_where_strictmath_does_not__():
    python_tan = pine_math.PYTHON_IMPLEMENTATIONS['tan']
    for x, expected in __test_helper_STRICTMATH_DIFFERS:
        assert math.tan(x) == expected, x
        assert python_tan(x) == expected, x


def __test_tan_matches_the_venue_where_strictmath_does__():
    python_tan = pine_math.PYTHON_IMPLEMENTATIONS['tan']
    for x, expected in __test_helper_STRICTMATH_AGREES:
        assert math.tan(x) == expected, x
        assert python_tan(x) == expected, x
