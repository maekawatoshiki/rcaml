let make_adder x =
  let adder y = x + y in
  adder in
print_int ((make_adder 3) 7)
