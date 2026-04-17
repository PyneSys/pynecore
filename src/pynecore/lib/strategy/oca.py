from ...types.strategy import Oca

#
# Constants
#
# The string values here are the Pine-side literals; the broker layer treats
# them as the authoritative enum members of
# :class:`pynecore.core.broker.models.OcaType`. A new OCA semantic must be
# added in both places or the sync engine will reject the intent.

cancel = Oca("cancel")
reduce = Oca("reduce")
none = Oca("none")
