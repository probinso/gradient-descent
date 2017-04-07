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

function ∂MSE_∂j(ps, j, coef)
    degree = size(coef)[1]
    target = [1.0 * convert(Float64, (i == j)) for i in 1:degree]
    others = 1.0 .- target

    Xs = ps * [1, 0]
    Ys = ps * [0, 1]
    M  = [r ^ c for r in Xs, c in 0:degree-1]

    vars = [-1 * others .* coef; 1]
    K  = sum(([M Ys]' .* vars)', 2)

    vars = [target; 0]
    A  = sum(([M Ys]' .* vars)', 2)
    mean((-2.* K .* A) .+ (2 .* A .^ 2 .* coef[j]))
end

#=
@show points = [2 3; 4 7]
STEP = ∂MSE_∂j(points, 2, [9, 10, 11, 12])
@show STEP
=#

function step_gradient(ps, coef, γ)
    h,  w  = size(ps)
    Δcoef = γ * [∂MSE_∂j(ps, j, coef) for (j, _) in enumerate(coef)]
    Δcoef = map((Δ) -> Δ == NaN ? 0 : Δ, Δcoef)

    ∇coef = [coef[i] - Δ for (i, Δ) in enumerate(Δcoef)]
end

function runner(ps, coef, γ, steps)
    for i in 1:steps
        tump = step_gradient(ps, coef, γ)
        if tump == coef
            break
        end
        coef = tump
    end
    coef
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

function mainx()
    points = Matrix(CSV.read("./data.csv", header=false, nullable=false))
    γ = 0.0001
    m₀, b₀ = 0, 0
    steps  = 1000
    println("Starting gradient descent at b = $(b₀), m = $(m₀), error = $(MSE(points,m₀,b₀))")
    m, b = runner(points, m₀, b₀, γ, steps)
    println("After $(steps) iterations b = $(b), m = $(m), error = $(MSE(points,m,b))")
end

function main()
    points = Matrix(CSV.read("./data.csv", header=false, nullable=false))
    γ = 0.0001
    steps = 1000
    coef  = [0, 0]
    println("Starting gradient descent at coef = $(coef)")
    coef  = runner(points, coef, γ, steps)
    println("After $(steps) iterations $(coef)")
end

main()
mainx()
