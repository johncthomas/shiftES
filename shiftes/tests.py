import pkg_resources
from shiftes.shiftes import effectsize, effectsize_ci
from shiftes.calculate_effectsize import parse_args, run_from_cmdline

# for argstr in [
#         'test_data.csv ALL ALL',
#         'test_data.nohead.tsv 1,2 3,4 --ci --nboot 200 --no-head',
#         'test.excel A B --type x --sheet "Second sheet"']:
#
#     f = pkg_resources.resource_filename(__name__, f'test_files/{fn}')
#     args =

print(pkg_resources.resource_filename(__name__, f'test_files'))

