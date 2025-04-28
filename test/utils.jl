@testset "Function: build_components" begin
    exog1 = rand(10, 3)
    exog2 = zeros(10, 0)

    Basic_Structural = StateSpaceLearning.StructuralModel(
        rand(10);
        level="stochastic",
        slope="stochastic",
        seasonal="stochastic",
        freq_seasonal=2,
        outlier=true,
        ζ_threshold=0,
        ω_threshold=0,
        exog=exog1,
    )
    Local_Level = StateSpaceLearning.StructuralModel(
        rand(10);
        level="stochastic",
        slope="none",
        seasonal="none",
        outlier=true,
        exog=exog1,
    )
    Local_Linear_Trend1 = StateSpaceLearning.StructuralModel(
        rand(10);
        level="stochastic",
        slope="stochastic",
        seasonal="none",
        outlier=false,
        exog=exog1,
    )
    Local_Linear_Trend2 = StateSpaceLearning.StructuralModel(
        rand(10);
        level="stochastic",
        slope="stochastic",
        seasonal="none",
        outlier=false,
        exog=exog2,
        ζ_threshold=0,
    )

    models = [Basic_Structural, Local_Level, Local_Linear_Trend1, Local_Linear_Trend2]

    for idx in eachindex(models)
        model = models[idx]
        X = model.X

        components_indexes = StateSpaceLearning.get_components_indexes(model)
        coefs = rand(size(X, 2))
        components = StateSpaceLearning.build_components(X, coefs, components_indexes)

        for key in keys(components)
            @test "Values" in keys(components[key])
            @test "Coefs" in keys(components[key])
            @test "Indexes" in keys(components[key])
            @test key == "exog" ? "Selected" in keys(components[key]) : true
        end
    end
end

@testset "Function: get_fit_and_residuals" begin
    coefs = rand(10)
    X = rand(30, 10)
    T = 30

    valid_indexes1 = setdiff(collect(1:30), [11, 12])
    estimation_ε1 = rand(length(valid_indexes1))
    ε1, fitted1 = StateSpaceLearning.get_fit_and_residuals(
        estimation_ε1, coefs, X, valid_indexes1, T
    )
    @test all(isnan.(ε1[11:12]))
    @test !all(isnan.(ε1[valid_indexes1]))
    @test !all(isnan.(fitted1))

    valid_indexes2 = collect(1:30)
    estimation_ε2 = rand(length(valid_indexes2))
    ε2, fitted2 = StateSpaceLearning.get_fit_and_residuals(
        estimation_ε2, coefs, X, valid_indexes2, T
    )
    @test !all(isnan.(ε2[valid_indexes2]))
    @test !all(isnan.(fitted2))
end

@testset "Function: has_intercept" begin
    X = rand(10, 3)
    @test !StateSpaceLearning.has_intercept(X)

    X = [ones(10) rand(10, 2)]
    @test StateSpaceLearning.has_intercept(X)
end

@testset "Function: handle_missing_values" begin
    y = rand(10)
    X = rand(10, 3)
    y[1] = NaN
    X[1, :] .= NaN
    y_treated, X_treated, valid_indexes = StateSpaceLearning.handle_missing_values(X, y)
    @test y_treated == y[2:end]
    @test X_treated == X[2:end, :]
    @test valid_indexes == 2:10
end

@testset "Function: get_stochastic_values" begin
    Random.seed!(1234)

    estimated_stochastic = rand(10)
    steps_ahead = 5
    T = 10
    start_idx = 1
    final_idx = 10
    seasonal_innovation_simulation1 = 0 
    seasonal_innovation_simulation2 = 2
    
    st_values1 = StateSpaceLearning.get_stochastic_values(estimated_stochastic, steps_ahead, T, start_idx, final_idx, seasonal_innovation_simulation1)
    st_values2 = StateSpaceLearning.get_stochastic_values(estimated_stochastic, steps_ahead, T, start_idx, final_idx, seasonal_innovation_simulation2)

    @test length(st_values1) == steps_ahead
    @test length(st_values2) == steps_ahead

    @test all(isapprox.(st_values1, [  0.6395615996802734
                                        -0.8396219340580711
                                        0.6395615996802734
                                        -0.5798621201341324
                                        0.967142768915383], atol=1e-6))
    @test all(isapprox.(st_values2, [   0.520354993723718
                                        -0.014908849285099945
                                        -0.13102565622085904
                                        -0.6395615996802734
                                        -0.520354993723718], atol=1e-6))
end

