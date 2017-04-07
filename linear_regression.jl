#!/usr/bin/env julia

using CSV

function MSE{T <: Real}(ps::Matrix{T}, coef::Vector{T})
    h, w = size(ps)
    degree = size(coef)[1]

    x⃗ = ps * [1, 0]
    y⃗ = ps * [0, 1]
    M = [r ^ c for r in x⃗, c in 0:degree-1]

    mean(sum(([M y⃗]' .* [-1 * coef; 1])', 2) .^ 2)
end

function ∂MSE_∂j{T <: Real}(ps::Matrix{T}, j::Int, coef::Vector{T})
    #=
    Gradient wrt coeficient at index j | given points
    =#
    degree = size(coef)[1]
    target = [1.0 * convert(Float64, (i == j)) for i in 1:degree]
    others = 1.0 .- target

    x⃗ = ps * [1, 0]
    y⃗ = ps * [0, 1]
    M = [r ^ c for r in x⃗, c in 0:degree-1]

    vars = [-1 * others .* coef; 1]
    K  = sum(([M y⃗]' .* vars)', 2)

    vars = [target; 0]
    A  = sum(([M y⃗]' .* vars)', 2)
    mean((-2.* K .* A) .+ (2 .* A .^ 2 .* coef[j]))
end

function step_gradient{T <: Real}(ps::Matrix{T}, coef::Vector{T}, γ::T)
    h,  w  = size(ps)
    ∇coef = [∂MSE_∂j(ps, j, coef) for (j, _) in enumerate(coef)]
    Δcoef = map((Δ) -> Δ == NaN ? 0 : Δ, γ * ∇coef)

    convert(Vector{T}, [coef[i] - Δ for (i, Δ) in enumerate(Δcoef)])
end

function runner{T <: Real}(ps::Matrix{T}, coef::Vector{T}, γ::T, steps::Int)
    for i in 1:steps
        tump = step_gradient(ps, coef, γ)
        if tump == coef
            break
        end
        coef = tump
    end
    coef
end

function main()
    points = Matrix(CSV.read("./data.csv", header=false, nullable=false))
    γ      = 0.0001
    steps  = 1000
    degree = 2
    coef   = convert(Vector{Float64}, [0.0 for i in 1:degree])
    println("Starting gradient descent at coef = $(coef), MSE = $(MSE(points, coef))")
    coef  = runner(points, coef, γ, steps)
    println("After $(steps) iterations $(coef), MSE = $(MSE(points, coef))")
end

main()
