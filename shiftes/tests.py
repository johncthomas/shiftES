import pkg_resources

f = pkg_resources.resource_filename(__name__, 'test_files/test_data.csv')
print(f)


