@testset "Function: build_components" begin
    Exogenous_X1 = rand(10, 3)
    Exogenous_X2 = zeros(10, 0)

    Basic_Structural = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=true,
        seasonal=true,
        stochastic_seasonal=true,
        freq_seasonal=2,
        outlier=true,
        ζ_ω_threshold=0,
        Exogenous_X=Exogenous_X1,
    )
    Local_Level = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=false,
        stochastic_trend=false,
        seasonal=false,
        stochastic_seasonal=false,
        freq_seasonal=2,
        outlier=true,
        ζ_ω_threshold=0,
        Exogenous_X=Exogenous_X1,
    )
    Local_Linear_Trend1 = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=true,
        seasonal=false,
        stochastic_seasonal=false,
        freq_seasonal=2,
        outlier=false,
        ζ_ω_threshold=0,
        Exogenous_X=Exogenous_X1,
    )
    Local_Linear_Trend2 = StateSpaceLearning.StructuralModel(
        rand(10);
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=true,
        seasonal=false,
        stochastic_seasonal=false,
        freq_seasonal=2,
        outlier=false,
        ζ_ω_threshold=0,
        Exogenous_X=Exogenous_X2,
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
            @test key == "Exogenous_X" ? "Selected" in keys(components[key]) : true
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

@testset "Function: fill_innovation_coefs" begin
    model = StateSpaceLearning.StructuralModel(rand(100))
    StateSpaceLearning.fit!(model)
    components = ["ξ", "ζ", "ω"]

    valid_indexes = model.output.valid_indexes

    inov_comp1 = StateSpaceLearning.fill_innovation_coefs(model, components[1], valid_indexes)
    inov_comp2 = StateSpaceLearning.fill_innovation_coefs(model, components[2], valid_indexes)
    inov_comp3 = StateSpaceLearning.fill_innovation_coefs(model, components[3], valid_indexes)

    @test length(inov_comp1) == 100
    @test length(inov_comp2) == 100
    @test length(inov_comp3) == 100

    model = StateSpaceLearning.StructuralModel(rand(100, 3))
    StateSpaceLearning.fit!(model)
    components = ["ξ", "ζ", "ω"]

    valid_indexes = model.output[1].valid_indexes

    inov_comp1 = StateSpaceLearning.fill_innovation_coefs(model, components[1], valid_indexes)
    inov_comp2 = StateSpaceLearning.fill_innovation_coefs(model, components[2], valid_indexes)
    inov_comp3 = StateSpaceLearning.fill_innovation_coefs(model, components[3], valid_indexes)

    @test size(inov_comp1) == (100, 3)
    @test size(inov_comp2) == (100, 3)
    @test size(inov_comp3) == (100, 3)
    
end
