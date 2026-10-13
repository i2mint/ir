"""Import-time dependency check; must be ``ir``'s first import (see ``requirements``)."""

from .requirements import check_requirements

check_requirements()
