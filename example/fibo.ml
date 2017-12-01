let fibo x =
  let rec fibo_sub x a b = 
    if x = 0 then a
    else fibo_sub (x - 1) (a + b) a in
  fibo_sub x 1 0 in 
print_int (fibo 40) 
