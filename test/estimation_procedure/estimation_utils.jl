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
    components_indexes1 = StateSpaceLearning.get_components_indexes(10, 3, Exogenous_X1, true, "Basic Structural", 0)
    components_indexes2 = StateSpaceLearning.get_components_indexes(10, 3, Exogenous_X2, true, "Basic Structural", 0)

    Estimation_X1 = StateSpaceLearning.create_X("Basic Structural", 10, 3, Exogenous_X1, true, 0)
    outlier_duplicate_columns1 = StateSpaceLearning.get_outlier_duplicate_columns(Estimation_X1, components_indexes1)
    @test outlier_duplicate_columns1 == [32]

    Estimation_X2 = StateSpaceLearning.create_X("Basic Structural", 10, 3, Exogenous_X2, true, 0)
    outlier_duplicate_columns2 = StateSpaceLearning.get_outlier_duplicate_columns(Estimation_X2, components_indexes2)
    @test outlier_duplicate_columns2 == []
end

@testset "Function: fit_estimation_procedure" begin
    Random.seed!(1234)
    Exogenous_X = hcat(rand(10, 3), vcat(ones(3), zeros(1), ones(6)))
    components_indexes = StateSpaceLearning.get_components_indexes(10, 3, Exogenous_X, true, "Basic Structural", 0)
    Estimation_X = StateSpaceLearning.create_X("Basic Structural", 10, 3, Exogenous_X, true, 0)
    estimation_y = Estimation_X*rand(size(Estimation_X, 2)) + rand(10).*5

    coefs1, ϵ1 = StateSpaceLearning.fit_estimation_procedure("Lasso", Estimation_X, estimation_y, 0.1, "aic", components_indexes, 0.1, true, true)
    @test length(coefs1) == 43
    @test length(ϵ1) == 10

    coefs2, ϵ2 = StateSpaceLearning.fit_estimation_procedure("AdaLasso", Estimation_X, estimation_y, 0.1, "aic", components_indexes, 0.1, true, true)
    @test length(coefs2) == 43
    @test length(ϵ2) == 10
end