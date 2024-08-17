Random.seed!(1234)
Estimation_X = rand(30, 3)
estimation_y = rand(30)
α = 0.5
penalty_factor = ones(3)
@testset "Function: get_path_information_criteria" begin
    intercept1 = true
    intercept2 = false

    model1 = glmnet(Estimation_X, estimation_y, alpha = α, penalty_factor = penalty_factor, intercept = intercept1, dfmax=size(Estimation_X, 2), lambda_min_ratio=0.001)
    coefs1, ε1 = StateSpaceLearning.get_path_information_criteria(model1, Estimation_X, estimation_y, "aic"; intercept = intercept1)
    @test length(coefs1) == 4
    @test coefs1[1] != 0
    @test all(coefs1[2:end] .== 0)
    @test length(ε1) == 30
     
    model2 = glmnet(Estimation_X, estimation_y, alpha = α, penalty_factor = penalty_factor, intercept = intercept2, dfmax=size(Estimation_X, 2), lambda_min_ratio=0.001)
    coefs2, ε2 = StateSpaceLearning.get_path_information_criteria(model2, Estimation_X, estimation_y, "aic"; intercept = intercept2)
    @test length(coefs2) == 3
    @test all(coefs2 .== 0)
    @test length(ε2) == 30
end

@testset "Function: fit_glmnet" begin
    coefs, ε = StateSpaceLearning.fit_glmnet(Estimation_X, estimation_y, α; information_criteria="aic", penalty_factor=penalty_factor, intercept = true)
    @test length(coefs) == 4
    @test length(ε) == 30
end

@testset "Function: fit_lasso" begin
    Random.seed!(1234)
    Exogenous_X = hcat(rand(10, 3), vcat(zeros(3), ones(1), zeros(6)))
    Basic_Structural = Dict("level" => true, "stochastic_level" => true, "trend" => true, "stochastic_trend" => true, "seasonal" => true, "stochastic_seasonal" => true, "freq_seasonal" => 2, "outlier" => true, "ζ_ω_threshold" => 0)
    Basic_Structural_w_level = Dict("level" => false, "stochastic_level" => true, "trend" => true, "stochastic_trend" => true, "seasonal" => true, "stochastic_seasonal" => true, "freq_seasonal" => 2, "outlier" => true, "ζ_ω_threshold" => 0)

    components_indexes = StateSpaceLearning.get_components_indexes(Exogenous_X, Basic_Structural)
    components_indexes2 = StateSpaceLearning.get_components_indexes(Exogenous_X, Basic_Structural_w_level)

    Estimation_X = StateSpaceLearning.create_X(Basic_Structural, Exogenous_X)
    Estimation_X2 = StateSpaceLearning.create_X(Basic_Structural_w_level, Exogenous_X)
    estimation_y = Estimation_X*rand(size(Estimation_X, 2)) + rand(10)

    coefs1, ε1 = StateSpaceLearning.fit_lasso(Estimation_X, estimation_y, 0.1, "aic", true, components_indexes, ones(size(Estimation_X, 2) - 1); rm_average = true)
    @test length(coefs1) == 43
    @test length(ε1) == 10

    coefs1, ε1 = StateSpaceLearning.fit_lasso(Estimation_X2, estimation_y, 0.1, "aic", true, components_indexes2, ones(size(Estimation_X2, 2)); rm_average = false)
    @test length(coefs1) == 42
    @test length(ε1) == 10

    coefs2, ε2 = StateSpaceLearning.fit_lasso(Estimation_X, estimation_y, 0.1, "aic", true, components_indexes, ones(size(Estimation_X, 2) - 1); rm_average = true)
    @test coefs2[1] == mean(estimation_y)
    @test length(coefs2) == 43
    @test length(ε2) == 10

    coefs3, ε3 = StateSpaceLearning.fit_lasso(Estimation_X, estimation_y, 0.1, "aic", false, components_indexes, ones(size(Estimation_X, 2) - 1); rm_average = true)
    @test coefs3[components_indexes["o"][4]] == 0
    @test all(coefs3[components_indexes["Exogenous_X"]] .!= 0)
    @test length(coefs3) == 43
    @test length(ε3) == 10

    coefs4, ε4 = StateSpaceLearning.fit_lasso(Estimation_X, estimation_y, 0.1, "aic", true, components_indexes, vcat(ones(1), ones(size(Estimation_X,2) - 2).*Inf); rm_average = true)
    @test all(coefs4[3:end] .== 0)
    @test length(coefs4) == 43
    @test length(ε4) == 10
end

@testset "Function: default_estimation_procedure" begin
    Random.seed!(1234)
    Exogenous_X = hcat(rand(10, 3), vcat(ones(3), zeros(1), ones(6)))
    Basic_Structural = Dict("level" => true, "stochastic_level" => true, "trend" => true, "stochastic_trend" => true, "seasonal" => true, "stochastic_seasonal" => true, "freq_seasonal" => 2, "outlier" => true, "ζ_ω_threshold" => 0)
    Basic_Structural_w_level = Dict("level" => false, "stochastic_level" => true, "trend" => true, "stochastic_trend" => true, "seasonal" => true, "stochastic_seasonal" => true, "freq_seasonal" => 2, "outlier" => true, "ζ_ω_threshold" => 0)

    components_indexes = StateSpaceLearning.get_components_indexes(Exogenous_X, Basic_Structural)
    components_indexes2 = StateSpaceLearning.get_components_indexes(Exogenous_X, Basic_Structural_w_level)

    Estimation_X = StateSpaceLearning.create_X(Basic_Structural, Exogenous_X)
    Estimation_X2 = StateSpaceLearning.create_X(Basic_Structural_w_level, Exogenous_X)

    estimation_y = Estimation_X*rand(size(Estimation_X, 2)) + rand(10).*5

    estimation_input1 = Dict("α" => 0.1, "information_criteria" => "aic", "ϵ" => 0.05, "penalize_exogenous" => true, "penalize_initial_states" => true)
    coefs1, ε1 = StateSpaceLearning.default_estimation_procedure(Estimation_X, estimation_y, components_indexes, estimation_input1)
    @test length(coefs1) == 43
    @test length(ε1) == 10

    estimation_input1 = Dict("α" => 0.1, "information_criteria" => "aic", "ϵ" => 0.05, "penalize_exogenous" => true, "penalize_initial_states" => true)
    coefs1, ε1 = StateSpaceLearning.default_estimation_procedure(Estimation_X2, estimation_y, components_indexes2, estimation_input1)
    @test length(coefs1) == 42
    @test length(ε1) == 10

    estimation_input2 = Dict("α" => 0.1, "information_criteria" => "aic", "ϵ" => 0.05, "penalize_exogenous" => true, "penalize_initial_states" => false)
    coefs2, ε2 = StateSpaceLearning.default_estimation_procedure(Estimation_X, estimation_y, components_indexes, estimation_input2)
    @test length(coefs2) == 43
    @test length(ε2) == 10
    @test all(coefs2[components_indexes["initial_states"][2:end] .- 1] .!= 0)
end

@testset "Function: get_dummy_indexes" begin 
    Exogenous_X1 = hcat(rand(10, 3), vcat(zeros(3), ones(1), zeros(6)))
    Exogenous_X2 = hcat(rand(10, 3))

    dummy_indexes1 = StateSpaceLearning.get_dummy_indexes(Exogenous_X1)
    @test dummy_indexes1 == [4]

    dummy_indexes2 = StateSpaceLearning.get_dummy_indexes(Exogenous_X2)
    @test dummy_indexes2 == []
end

@testset "Function: get_outlier_duplicate_columns" begin
    Random.seed!(1234)
    Exogenous_X1 = hcat(rand(10, 3), vcat(zeros(3), ones(1), zeros(6)))
    Exogenous_X2 = rand(10, 3)

    Basic_Structural = Dict("level"=> true, "stochastic_level" => true, "trend" => true, "stochastic_trend" => true, "seasonal" => true, "stochastic_seasonal" => true, "freq_seasonal" => 2, "outlier" => true, "ζ_ω_threshold" => 0)

    components_indexes1 = StateSpaceLearning.get_components_indexes(Exogenous_X1, Basic_Structural)
    components_indexes2 = StateSpaceLearning.get_components_indexes(Exogenous_X2, Basic_Structural)

    Estimation_X1 = StateSpaceLearning.create_X(Basic_Structural, Exogenous_X1)
    outlier_duplicate_columns1 = StateSpaceLearning.get_outlier_duplicate_columns(Estimation_X1, components_indexes1)
    @test outlier_duplicate_columns1 == [32]

    Estimation_X2 = StateSpaceLearning.create_X(Basic_Structural, Exogenous_X2)
    outlier_duplicate_columns2 = StateSpaceLearning.get_outlier_duplicate_columns(Estimation_X2, components_indexes2)
    @test outlier_duplicate_columns2 == []
end