#!/usr/bin/env julia

using CSV

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
    ∇m, ∇b = ∂MSE_∂m(ps, m, b), ∂MSE_∂b(ps, m, b)
    Δm, Δb = γ * ∇m, γ * ∇b
    m = Δm == NaN ? m : m - Δm
    b = Δb == NaN ? b : b - Δb
    m, b
end

function runner(ps, m₀, b₀, γ, steps)
    m, b, = m₀, b₀
    for i in 1:steps
        x, y = step_gradient(ps, m, b, γ)
        if (m,b) == (x, y)
            break
        end
        m, b = x, y
    end
    m, b
end

function main()
    points = Matrix(CSV.read("./data.csv", header=false, nullable=false))
    γ = 0.0001
    m₀, b₀ = 0, 0
    steps  = 1000
    println("Starting gradient descent at b = $(b₀), m = $(m₀), error = $(MSE(points,m₀,b₀))")
    m, b = runner(points, m₀, b₀, γ, steps)
    println("After $(steps) iterations b = $(b), m = $(m), error = $(MSE(points,m,b))")
end

main()
