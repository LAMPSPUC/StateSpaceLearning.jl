@testset "Function: fit_adalasso" begin
    Random.seed!(1234)
    Exogenous_X = hcat(rand(10, 3), vcat(ones(3), zeros(1), ones(6)))
    components_indexes = StateSpaceLearning.get_components_indexes(10, 3, Exogenous_X, true, "Basic Structural", 0)
    Estimation_X = StateSpaceLearning.create_X("Basic Structural", 10, 3, Exogenous_X, true, 0)

    estimation_y = Estimation_X*rand(size(Estimation_X, 2)) + rand(10).*5

    coefs1, ϵ1 = StateSpaceLearning.fit_adalasso(Estimation_X, estimation_y, 0.1, "aic", components_indexes, 0.1, true)
    @test length(coefs1) == 43
    @test length(ϵ1) == 10

    coefs2, ϵ2 = StateSpaceLearning.fit_adalasso(Estimation_X, estimation_y, 0.1, "aic", components_indexes, 10000.0, true)
    coefs_lasso, ϵ_lasso = StateSpaceLearning.fit_lasso(Estimation_X, estimation_y, 0.1, "aic", true, components_indexes; intercept = true)
    @test all(isapprox.(coefs2, coefs_lasso; atol = 1e-3))
    @test all(isapprox.(ϵ2, ϵ_lasso; atol = 1e-3))
end