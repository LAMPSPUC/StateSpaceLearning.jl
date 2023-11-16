Random.seed!(1234)
Estimation_X = rand(30, 3)
estimation_y = rand(30)
α = 0.5
penalty_factor = ones(3)
@testset "Function: get_path_information_criteria" begin
    intercept1 = true
    intercept2 = false

    model1 = glmnet(Estimation_X, estimation_y, alpha = α, penalty_factor = penalty_factor, intercept = intercept1, dfmax=size(Estimation_X, 2), lambda_min_ratio=0.001)
    coefs1, ϵ1 = StateSpaceLearning.get_path_information_criteria(model1, Estimation_X, estimation_y, "aic"; intercept = intercept1)
    @test length(coefs1) == 4
    @test coefs1[1] != 0
    @test all(coefs1[2:end] .== 0)
    @test length(ϵ1) == 30
     
    model2 = glmnet(Estimation_X, estimation_y, alpha = α, penalty_factor = penalty_factor, intercept = intercept2, dfmax=size(Estimation_X, 2), lambda_min_ratio=0.001)
    coefs2, ϵ2 = StateSpaceLearning.get_path_information_criteria(model2, Estimation_X, estimation_y, "aic"; intercept = intercept2)
    @test length(coefs2) == 3
    @test all(coefs2 .== 0)
    @test length(ϵ2) == 30
end

@testset "Function: fit_glmnet" begin
    coefs, ϵ = StateSpaceLearning.fit_glmnet(Estimation_X, estimation_y, α; hyperparameter_selection="aic", penalty_factor=penalty_factor, intercept = true)
    @test length(coefs) == 4
    @test length(ϵ) == 30
end

@testset "Function: fit_lasso" begin
    Random.seed!(1234)
    Exogenous_X = hcat(rand(10, 3), vcat(zeros(3), ones(1), zeros(6)))
    components_indexes = StateSpaceLearning.get_components_indexes(10, 3, Exogenous_X, true, "Basic Structural", 0)
    Estimation_X = StateSpaceLearning.create_X("Basic Structural", 10, 3, Exogenous_X, true, 0)
    estimation_y = Estimation_X*rand(size(Estimation_X, 2)) + rand(10)

    coefs1, ϵ1 = StateSpaceLearning.fit_lasso(Estimation_X, estimation_y, 0.1, "aic", true, components_indexes; intercept = true)
    @test length(coefs1) == 43
    @test length(ϵ1) == 10

    coefs2, ϵ2 = StateSpaceLearning.fit_lasso(Estimation_X, estimation_y, 0.1, "aic", true, components_indexes; intercept = false)
    @test coefs2[1] == mean(estimation_y)
    @test length(coefs2) == 43
    @test length(ϵ2) == 10

    coefs3, ϵ3 = StateSpaceLearning.fit_lasso(Estimation_X, estimation_y, 0.1, "aic", false, components_indexes; intercept = true)
    @test coefs3[components_indexes["o"][4]] == 0
    @test all(coefs3[components_indexes["Exogenous_X"]] .!= 0)
    @test length(coefs3) == 43
    @test length(ϵ3) == 10

    coefs4, ϵ4 = StateSpaceLearning.fit_lasso(Estimation_X, estimation_y, 0.1, "aic", true, components_indexes; penalty_factor = vcat(ones(1), ones(size(Estimation_X,2) - 2).*Inf), intercept = true)
    @test all(coefs4[3:end] .== 0)
    @test length(coefs4) == 43
    @test length(ϵ4) == 10
end