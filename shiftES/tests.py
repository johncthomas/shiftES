#!/usr/bin/env python
import pkg_resources
import shiftes
from shiftES.shiftes import effectsize, effectsize_ci
from shiftES.calculate_effectsize import parse_args, run_from_cmdline

print(pkg_resources.resource_filename(__name__, f'test_files'))

