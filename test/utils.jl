@testset "Function: build_components" begin
    Exogenous_X1 = rand(10, 3)
    Exogenous_X2 = zeros(10, 0)

    Basic_Structural = Dict("stochastic_level" => true, "trend" => true, "stochastic_trend" => true, "seasonal" => true, "stochastic_seasonal" => true, "freq_seasonal" => 2)
    Local_Level = Dict("stochastic_level" => true, "trend" => false, "stochastic_trend" => false, "seasonal" => false, "stochastic_seasonal" => false, "freq_seasonal" => 2)
    Local_Linear_Trend = Dict("stochastic_level" => true, "trend" => true, "stochastic_trend" => true, "seasonal" => false, "stochastic_seasonal" => false, "freq_seasonal" => 2)
    parameter_combination = [
        [Basic_Structural, true, Exogenous_X1],
        [Local_Level, true, Exogenous_X1],
        [Local_Linear_Trend, true, Exogenous_X1],
        [Basic_Structural, false, Exogenous_X1],
        [Basic_Structural, true, Exogenous_X2],
    ]

    for param in parameter_combination
        X = StateSpaceLearning.create_X_unobserved_components(param[1], param[3], param[2], 0, 10, 0)

        components_indexes = StateSpaceLearning.get_components_indexes(10, param[3], param[1], param[2], 0)
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
    estimation_ϵ1 = rand(length(valid_indexes1))
    ϵ1, fitted1 = StateSpaceLearning.get_fit_and_residuals(estimation_ϵ1, coefs, X, valid_indexes1, T)
    @test all(isnan.(ϵ1[11:12]))
    @test !all(isnan.(ϵ1[valid_indexes1]))
    @test !all(isnan.(fitted1))

    valid_indexes2 = collect(1:30)
    estimation_ϵ2 = rand(length(valid_indexes2))
    ϵ2, fitted2 = StateSpaceLearning.get_fit_and_residuals(estimation_ϵ2, coefs, X, valid_indexes2, T)
    @test !all(isnan.(ϵ2[valid_indexes2]))
    @test !all(isnan.(fitted2))
end