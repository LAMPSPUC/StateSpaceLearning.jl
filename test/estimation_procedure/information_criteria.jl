@testset "Function: get_information"  begin 
    ϵ = [1.1, 2.2, 3.3, 4.4, 5.5]
    T = 5
    K = 3
    p = 10
    bic = StateSpaceLearning.get_information(T, K, ϵ; hyperparameter_selection = "bic", p = p)
    aic = StateSpaceLearning.get_information(T, K, ϵ; hyperparameter_selection = "aic", p = p)
    aicc = StateSpaceLearning.get_information(T, K, ϵ; hyperparameter_selection = "aicc", p = p)
    EBIC = StateSpaceLearning.get_information(T, K, ϵ; hyperparameter_selection = "EBIC", p = p)
    @test bic == 10.362869194716325
    @test aic == 11.534555457414024
    @test aicc == 35.53455545741402
    @test EBIC == 12.340528691778138
end