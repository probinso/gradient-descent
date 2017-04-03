#!/usr/bin/env julia
using DataFrames

function MSE(ps, m,  b)
    h, w = size(ps)
    mean( ([ps' ; ones(h)']' * [-m ; 1 ; -b]) .^ 2)
end

function ∂MSE_∂m(ps, m, b)
    h, w = size(ps)
    -2 * mean(ps * [1; 0] .* [ps' ; ones(h)']' * [-m ; 1 ; -b])
end

function ∂MSE_∂b(ps, m, b)
    h, w = size(ps)
    -2 * mean([ps' ; ones(h)']' * [-m ; 1 ; -b])
end

function step_gradient(ps, m, b, γ)
    h,  w  = size(ps)
    ∇b, ∇m = ∂MSE_∂b(ps, m, b), ∂MSE_∂m(ps, m, b)
    m,  b  = m - γ * ∇m, b - γ * ∇b
end

function runner(ps, m₀, b₀, γ, steps)
    m, b, = m₀, b₀
    for i in 1:steps
        m, b = step_gradient(ps, m, b, γ)
        #println(MSE(ps, b, m))
    end
    b, m
end


function main()
    points = Matrix(readtable("./data.csv", header=false))
    γ = 0.0001
    m₀, b₀ = 0, 0
    steps  = 1000
    println("Starting gradient descent at b = $(b₀), m = $(m₀), error = $(MSE(points,m₀,b₀))")
    b, m = runner(points, m₀, b₀, γ, steps)
    println("After $(steps) iterations b = $(b), m = $(m), error = $(MSE(points,m,b))")
end

main()
