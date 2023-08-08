library(pwr)
ch <- cohen.ES(test = "t", size = "medium")

print(ch)

test <- pwr.t.test(d = 0.5, power = 0.80, sig.level = 0.05)
print(test)