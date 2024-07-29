@testset "Innovation matrices" begin
    @test StateSpaceLearning.ξ_size(10) == 8
    @test StateSpaceLearning.ζ_size(10, 2) == 6
    @test StateSpaceLearning.ζ_size(10, 0) == 8
    @test StateSpaceLearning.ω_size(10, 2, 0) == 9
    @test StateSpaceLearning.ω_size(10, 2, 2) == 7
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

    @test X_ζ1 == [0.0  0.0  0.0;
                   0.0  0.0  0.0;
                   1.0  0.0  0.0;
                   2.0  1.0  0.0;
                   3.0  2.0  1.0]

    @test X_ζ2 == [0.0  0.0  0.0;
                   0.0  0.0  0.0;
                   1.0  0.0  0.0;
                   2.0  1.0  0.0;
                   3.0  2.0  1.0;
                   4.0  3.0  2.0;
                   5.0  4.0  3.0]

    @test X_ζ3 == reshape([0.0;
                           0.0;
                           1.0;
                           2.0;
                           3.0;
                           4.0;
                           5.0], 7, 1)

    X_ω1 = StateSpaceLearning.create_ω(5, 2, 0, 0)
    X_ω2 = StateSpaceLearning.create_ω(5, 2, 2, 0)

    @test X_ω1 == [0.0   0.0   0.0 0.0;
                   0.0   0.0   0.0 0.0;
                  -1.0   1.0   0.0 0.0;
                   0.0  -1.0   1.0 0.0;
                  -1.0   1.0  -1.0 1.0]

    @test X_ω2 == [0.0   0.0   0.0   0.0;
                   0.0   0.0   0.0   0.0;
                  -1.0   1.0   0.0   0.0;
                   0.0  -1.0   1.0   0.0;
                  -1.0   1.0  -1.0   1.0;
                   0.0  -1.0   1.0  -1.0;
                  -1.0   1.0  -1.0   1.0]

    X_o1 = StateSpaceLearning.create_o_matrix(3, 0)
    X_o2 = StateSpaceLearning.create_o_matrix(3, 2)

    @test X_o1 == Matrix(1.0 * I, 3, 3)
    @test X_o2 == vcat(Matrix(1.0 * I, 3, 3), zeros(2, 3))
end


@testset "Initial State Matrix" begin 
    X1 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 0, "Basic Structural")
    X2 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 2, "Basic Structural")

    @test X1 == [1.0  0.0  1.0  0.0;
                 1.0  1.0  0.0  1.0;
                 1.0  2.0  1.0  0.0;
                 1.0  3.0  0.0  1.0;
                 1.0  4.0  1.0  0.0]

    @test X2 == [1.0  0.0  1.0  0.0;
                 1.0  1.0  0.0  1.0;
                 1.0  2.0  1.0  0.0;
                 1.0  3.0  0.0  1.0;
                 1.0  4.0  1.0  0.0;
                 1.0  5.0  0.0  1.0;
                 1.0  6.0  1.0  0.0]

    X3 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 0, "Local Linear Trend")
    X4 = StateSpaceLearning.create_initial_states_Matrix(5, 2, 2, "Local Linear Trend")

    @test X3 == [1.0  0.0;
                 1.0  1.0;
                 1.0  2.0;
                 1.0  3.0;
                 1.0  4.0]

    @test X4 == [1.0  0.0;
                 1.0  1.0;
                 1.0  2.0;
                 1.0  3.0;
                 1.0  4.0;
                 1.0  5.0;
                 1.0  6.0] 

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

    size_vec1=[(5, 22), (5, 18), (7, 18), (5, 17), (5, 13), (7, 13), (5, 12), (5, 12), (7, 12), (5, 7), (5, 7), (7, 7), (5, 16), (5, 14), (7, 14), (5, 11), (5, 9), (7, 9)]
    size_vec2=[(5, 19), (5, 15), (7, 15), (5, 14), (5, 10), (7, 10), (5, 9), (5, 9), (7, 9), (5, 4), (5, 4), (7, 4), (5, 13), (5, 11), (7, 11), (5, 8), (5, 6), (7, 6)]
    count = 1
    for model_type in ["Basic Structural", "Local Level", "Local Linear Trend"]
        for param in param_combination
            if param[3] != 0
                X1 = StateSpaceLearning.create_X_unobserved_components(model_type, 5, 2, Exogenous_X1, param[1], param[2], param[3], Exogenous_forecast1)
            else
                X1 = StateSpaceLearning.create_X_unobserved_components(model_type, 5, 2, Exogenous_X1, param[1], param[2], param[3])
            end
            X2 = StateSpaceLearning.create_X_unobserved_components(model_type, 5, 2, Exogenous_X2, param[1], param[2], param[3])
            @test size(X1) == size_vec1[count]
            @test size(X2) == size_vec2[count]
            count += 1
        end
    end
    @test_throws ErrorException StateSpaceLearning.create_X_unobserved_components("none", 5, 2, Exogenous_X1, param_combination[1][1], param_combination[1][2], param_combination[1][3], Exogenous_forecast1)
end

@testset "Function: get_components_indexes_unobserved_components" begin
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
        
        components_indexes = StateSpaceLearning.get_components_indexes_unobserved_components(10, 3, param[3], param[2], param[1], 0)
        
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
    @test_throws ErrorException StateSpaceLearning.get_components_indexes_unobserved_components(10, 3, Exogenous_X1, true, "none", 0)
end

@testset "Function: get_variances_unobserved_components" begin
    Exogenous_X2 = zeros(10, 0)
    parameter_combination = [
        ["Basic Structural", true, Exogenous_X2, ["ξ", "ζ", "ω", "ϵ"]],
        ["Local Level", true, Exogenous_X2, ["ξ", "ϵ"]],
        ["Local Linear Trend", true, Exogenous_X2, ["ξ", "ζ", "ϵ"]]
    ]
    for param in parameter_combination
        components_indexes = StateSpaceLearning.get_components_indexes_unobserved_components(10, 3, param[3], param[2], param[1], 0)
        variances = StateSpaceLearning.get_variances_unobserved_components(rand(100), rand(39), components_indexes)
        @test all([key in keys(variances) for key in param[4]])
    end
end 