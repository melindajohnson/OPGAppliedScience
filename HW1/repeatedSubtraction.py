"""
The following expressions all result in zero:
    1)  1000 - sum(0.1, i=1..10000)
    2) 10000 - sum(0.1, i=1..100000)
    3) 100000 - sum(0.1, i=1..1000000)
Write an algorithm to compute each of the above repeated subtractions and compare the
answer to the exact answer of zero (i.e. calculate the Absolute Error).

Psuedocode:
 loop through 1000 times
    add 0.1 to a total

Subtract the total from 1000 and calculate the absolute error
 or do repeated subtraction of 0.1 from 1000 and calculate the absolute error

"""

def repeated_subtraction(start, n):
    x = start
    for i in range(n):
        x = x - 0.1
    return x

def repeated_addition(n):
    x = 0
    for i in range(n):
        x = x + 0.1
    return x

# Case 1
result1 = repeated_subtraction(1000, 10000)
error1 = abs(result1 - 0)

# Case 2
result2 = repeated_subtraction(10000, 100000)
error2 = abs(result2 - 0)

# Case 3
result3 = repeated_subtraction(100000, 1000000)
error3 = abs(result3 - 0)

print("Case 1 Error:", error1)
print("Case 2 Error:", error2)
print("Case 3 Error:", error3)


# Case 1
sum1 = repeated_addition(10000)
result1 = 1000 - sum1
error1 = abs(result1 - 0)

# Case 2
sum2 = repeated_addition(100000)
result2 = 10000 - sum2
error2 = abs(result2 - 0)

# Case 3
sum3 = repeated_addition(1000000)
result3 = 100000 - sum3
error3 = abs(result3 - 0)

print("Case 1 Error:", error1)
print("Case 2 Error:", error2)
print("Case 3 Error:", error3)


"""
repeated subtraction results:
Case 1 Error: 1.588713327560498e-10
Case 2 Error: 1.884864639367656e-08
Case 3 Error: 1.3328811345469926e-06
"""
"""
repeated addition results:
Case 1 Error: 1.588205122970976e-10
Case 2 Error: 1.8848368199542165e-08
Case 3 Error: 1.3328826753422618e-06

Learning:
 Due to IEEE 754 floating point representation, 
 0.1 cannot be stored exactly in binary. 
 Small rounding errors accumulate over many iterations, 
 producing non-zero results. More iterations leads to larger accumulated error.
"""