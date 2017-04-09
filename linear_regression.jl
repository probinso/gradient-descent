#!/usr/bin/env julia

using DataFrames

len(V::Vector) = size(V)[1]

function MSE{T <: Real}(ps::Matrix{T}, coef::Vector{T})
    # Mean Squared Error
    degree = len(coef)

    x⃗ = ps * [1, 0]
    y⃗ = ps * [0, 1]
    M = [r ^ c for r in x⃗, c in 0:degree-1]

    mean(sum(([M y⃗]' .* [-1 * coef; 1])', 2) .^ 2)
end

function ∂MSE_∂j{T <: Real}(ps::Matrix{T}, j::Int, coef::Vector{T})
    # Gradient of MSE wrt coeficient at index j | given points
    degree = len(coef)
    target = [convert(Real, (i == j)) for i in 1:degree]
    others = 1.0 .- target

    x⃗ = ps * [1, 0]
    y⃗ = ps * [0, 1]
    M = [r ^ c for r in x⃗, c in 0:degree-1]

    vars = [-1 * others .* coef; 1]
    K  = sum(([M y⃗]' .* vars)', 2)

    vars = [target; 0]
    A  = sum(([M y⃗]' .* vars)', 2)

    mean((-2 .* K .* A) .+ (2 * (A .^2) .* coef[j]))
end

function step_gradient{T <: Real}(ps::Matrix{T}, coef::Vector{T}, γ::T)
    ∇coef::Vector{T} = [∂MSE_∂j(ps, j, coef) for (j, _) in enumerate(coef)]
    Δcoef = map((Δ) -> Δ == NaN ? 0 : Δ, rand() * γ * ∇coef)
    convert(Vector{T}, [coef[i] - Δ for (i, Δ) in enumerate(Δcoef)])
end

function runner{T <: Real}(ps::Matrix{T}, coef::Vector{T}, γ::T, steps::Int)
    for i in 1:steps
        if Inf in coef || NaN in coef
            # Break if divergent
            return i, coef
        end
        coef = step_gradient(ps, coef, γ)
    end
    steps, coef
end

function main()
    points = convert(Matrix{Real},
                     DataFrames.readtable("./yield.csv", header=false))

    γ::Real = 0.0001
    Z = convert(Vector{Real}, rand((20)))
    steps = 1000
    for degree = 1:7
        s, coef = runner(points, Z[1:degree], γ, steps)
        coe = convert(Vector{Float64}, coef)
        err = convert(Float64, MSE(points, coef))
        println("After $(s) iterations MSE = $(err) \n::$(coe)")
    end
end

main()
