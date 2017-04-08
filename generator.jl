#!/usr/bin/env julia

using Distributions
using DataFrames

function poly(degree)
    coef = [2 * (-1) ^ i * c for (i, c) in enumerate(rand((degree)))]
    coef = convert(Vector{Real}, coef)
    @show coef

    funcs  = [x-> coef[i] * x ^ (i - 1) for i in 1:degree]
    reduce((a, b) -> x -> a(x) + b(x), funcs)
end

fd     = open("./yield.csv", "w+")
degree = 5
count  = 1000
noise  = Normal()

f = poly(degree)
for x in linspace(-2.0, 2.0, count)
    write(fd, "$(x),$(f(x) + rand(noise))\n")
end

close(fd)
