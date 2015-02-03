tests = ['test_autoconnect_simple',
         'test_autoconnect_complex']

threshold = 1e-8

print
for tst in tests:
    test = __import__(tst)
    t = []
    t.append(test.backprop_test)
    t.append(test.grad_test_mean < threshold)
    t.append(test.grad_test_max < threshold)
    if all(t):
        print tst,"passed"
    else:
        print tst,"FAIL"

