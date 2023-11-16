@testset "Innovation matrices" begin
    @test StateSpaceLearning.ξ_size(10) == 8
    @test StateSpaceLearning.ζ_size(10, 2) == 7
    @test StateSpaceLearning.ζ_size(10, 0) == 9
    @test StateSpaceLearning.ω_size(10, 2) == 8
    @test StateSpaceLearning.o_size(10) == 10

    X_ξ1 = StateSpaceLearning.create_ξ(5, 0)
    X_ξ2 = StateSpaceLearning.create_ξ(5, 2)

    @test X_ξ1 == [0.0  0.0  0.0;
                   1.0  0.0  0.0;
                   1.0  1.0  0.0;
                   1.0  1.0  1.0;
                   1.0  1.0  1.0]

    @test X_ξ2 == [0.0  0.0  0.0;
                   1.0  0.0  0.0;
                   1.0  1.0  0.0;
                   1.0  1.0  1.0;
                   1.0  1.0  1.0;
                   1.0  1.0  1.0;
                   1.0  1.0  1.0]

    X_ζ1 = StateSpaceLearning.create_ζ(5, 0, 0)
    X_ζ2 = StateSpaceLearning.create_ζ(5, 2, 0)
    X_ζ3 = StateSpaceLearning.create_ζ(5, 2, 2)

    @test X_ζ1 == [0.0  0.0  0.0  0.0;
                   2.0  0.0  0.0  0.0;
                   3.0  2.0  0.0  0.0;
                   4.0  3.0  2.0  0.0;
                   5.0  4.0  3.0  2.0]

    @test X_ζ2 == [0.0  0.0  0.0  0.0;
                   2.0  0.0  0.0  0.0;
                   3.0  2.0  0.0  0.0;
                   4.0  3.0  2.0  0.0;
                   5.0  4.0  3.0  2.0;
                   6.0  5.0  4.0  3.0;
                   7.0  6.0  5.0  4.0]

    @test X_ζ3 == [0.0  0.0;
                   2.0  0.0;
                   3.0  2.0;
                   4.0  3.0;
                   5.0  4.0;
                   6.0  5.0;
                   7.0  6.0]

    X_ω1 = StateSpaceLearning.create_ω(5, 2, 0)
    X_ω2 = StateSpaceLearning.create_ω(5, 2, 2)

    @test X_ω1 == [0.0   0.0   0.0;
                  -1.0   0.0   0.0;
                   1.0  -1.0   0.0;
                  -1.0   1.0  -1.0;
                   1.0  -1.0   1.0]

    @test X_ω2 == [0.0   0.0   0.0;
                  -1.0   0.0   0.0;
                   1.0  -1.0   0.0;
                  -1.0   1.0  -1.0;
                   1.0  -1.0   1.0;
                  -1.0   1.0  -1.0;
                   1.0  -1.0   1.0]

    X_o1 = StateSpaceLearning.create_o_matrix(3, 0)
    X_o2 = StateSpaceLearning.create_o_matrix(3, 2)

    @test X_o1 == Matrix(1.0 * I, 3, 3)
    @test X_o2 == vcat(Matrix(1.0 * I, 3, 3), zeros(2, 3))
end


@testset "Initial State Matrix" begin 
    X1 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 0, "Basic Structural")
    X2 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 2, "Basic Structural")

    @test X1 == [1.0  1.0  1.0  0.0;
                   1.0  2.0  0.0  1.0;
                   1.0  3.0  1.0  0.0;
                   1.0  4.0  0.0  1.0;
                   1.0  5.0  1.0  0.0]

    @test X2 == [1.0  1.0  1.0  0.0;
                   1.0  2.0  0.0  1.0;
                   1.0  3.0  1.0  0.0;
                   1.0  4.0  0.0  1.0;
                   1.0  5.0  1.0  0.0;
                   1.0  6.0  0.0  1.0;
                   1.0  7.0  1.0  0.0]

    X3 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 0, "Local Linear Trend")
    X4 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 2, "Local Linear Trend")

    @test X3 == [1.0  1.0;
                 1.0  2.0;
                 1.0  3.0;
                 1.0  4.0;
                 1.0  5.0]

    @test X4 == [1.0  1.0;
                 1.0  2.0;
                 1.0  3.0;
                 1.0  4.0;
                 1.0  5.0;
                 1.0  6.0;
                 1.0  7.0] 
    X5 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 0, "Local Level")
    X6 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 2, "Local Level")

    @test X5 == ones(5, 1)
    @test X6 == ones(7, 1)
end

@testset "Create X matrix" begin
    Exogenous_X1 = rand(5, 3)
    Exogenous_X2 = zeros(5, 0)

    Exogenous_forecast1 = rand(2, 3)

    param_combination = [
        Any[true, 0, 0], 
        Any[true, 2, 0],
        Any[true, 2, 2],
        Any[false, 0, 0],
        Any[false, 2, 0],
        Any[false, 2, 2]
    ]

    size_vec1=[(5, 22), (5, 20), (7, 20), (5, 17), (5, 15), (7, 15), (5, 12), (5, 12), (7, 12), (5, 7), (5, 7), (7, 7), (5, 17), (5, 15), (7, 15), (5, 12), (5, 10), (7, 10)]
    size_vec2=[(5, 19), (5, 17), (7, 17), (5, 14), (5, 12), (7, 12), (5, 9), (5, 9), (7, 9), (5, 4), (5, 4), (7, 4), (5, 14), (5, 12), (7, 12), (5, 9), (5, 7), (7, 7)]
    count = 1
    for model_type in ["Basic Structural", "Local Level", "Local Linear Trend"]
        for param in param_combination
            if param[3] != 0
                X1 = StateSpaceLearning.create_X(model_type, 5, 2, Exogenous_X1, param[1], param[2], param[3], Exogenous_forecast1)
            else
                X1 = StateSpaceLearning.create_X(model_type, 5, 2, Exogenous_X1, param[1], param[2], param[3])
            end
            X2 = StateSpaceLearning.create_X(model_type, 5, 2, Exogenous_X2, param[1], param[2], param[3])
            @test size(X1) == size_vec1[count]
            @test size(X2) == size_vec2[count]
            count += 1
        end
    end

end

@testset "Function: get_components_indexes" begin
    Exogenous_X1 = rand(10, 3)
    Exogenous_X2 = zeros(10, 0)

    parameter_combination = [
        ["Basic Structural", true, Exogenous_X1],
        ["Local Level", true, Exogenous_X1],
        ["Local Linear Trend", true, Exogenous_X1],
        ["Basic Structural", false, Exogenous_X1],
        ["Basic Structural", true, Exogenous_X2],
    ]

    for param in parameter_combination
        
        components_indexes = StateSpaceLearning.get_components_indexes(10, 3, param[3], param[2], param[1], 0)
        
        for key in keys(components_indexes)
            if param[1] == "Basic Structural"
                @test !(key in ["Exogenous_X", "o"]) ? !isempty(components_indexes[key]) : true
            elseif param[1] == "Local Level"
                @test !(key in ["ω", "γ₁", "ζ", "ν₁", "o", "Exogenous_X"]) ? !isempty(components_indexes[key]) : true
                @test !(key in ["o", "Exogenous_X", "μ₁", "ξ", "initial_states"]) ? isempty(components_indexes[key]) : true
            elseif param[1] == "Local Linear Trend"
                @test !(key in ["ω", "γ₁", "o", "Exogenous_X"]) ? !isempty(components_indexes[key]) : true
                @test !(key in ["μ₁", "ξ", "ζ", "ν₁", "o", "Exogenous_X", "initial_states"]) ? isempty(components_indexes[key]) : true
            end
            @test (param[2] && key == "o") ? !isempty(components_indexes[key]) : true
            @test (!param[2] && key == "o") ? isempty(components_indexes[key]) : true
            @test key == "Exogenous_X" ? length(components_indexes[key]) == size(param[3], 2) : true
        end
    end
    
end

@testset "Function: get_variances" begin
    Exogenous_X2 = zeros(10, 0)
    parameter_combination = [
        ["Basic Structural", true, Exogenous_X2, ["ξ", "ζ", "ω", "ϵ"]],
        ["Local Level", true, Exogenous_X2, ["ξ", "ϵ"]],
        ["Local Linear Trend", true, Exogenous_X2, ["ξ", "ζ", "ϵ"]]
    ]
    for param in parameter_combination
        components_indexes = StateSpaceLearning.get_components_indexes(10, 3, param[3], param[2], param[1], 0)
        variances = StateSpaceLearning.get_variances(rand(100), rand(39), components_indexes)
        @test all([key in keys(variances) for key in param[4]])
    end
end 

@testset "Function: forecast_model" begin
    Exogenous_X = rand(30, 3)
    X = StateSpaceLearning.create_X("Basic Structural", 30, 2, Exogenous_X, true, 0) 
    coefs = rand(size(X, 2))
    components_indexes  = StateSpaceLearning.get_components_indexes(30, 2, Exogenous_X, true, "Basic Structural", 0)
    components          = StateSpaceLearning.build_components(X, coefs, components_indexes)

    output = StateSpaceLearning.Output("Basic Structural", X, coefs, rand(30), rand(30), components, Dict(), 3, 30, true, collect(1:30), 0)
    @test length(StateSpaceLearning.forecast_model(output, 5, rand(5, 3))) == 5
end