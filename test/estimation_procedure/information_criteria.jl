@testset "Function: get_information"  begin 
    ϵ = [1.1, 2.2, 3.3, 4.4, 5.5]
    T = 5
    K = 3
    p = 10
    bic = StateSpaceLearning.get_information(T, K, ϵ; hyperparameter_selection = "bic", p = p)
    aic = StateSpaceLearning.get_information(T, K, ϵ; hyperparameter_selection = "aic", p = p)
    aicc = StateSpaceLearning.get_information(T, K, ϵ; hyperparameter_selection = "aicc", p = p)
    EBIC = StateSpaceLearning.get_information(T, K, ϵ; hyperparameter_selection = "EBIC", p = p)
    @test round(bic, digits = 5) == 10.36287
    @test round(aic, digits = 5) == 11.53456
    @test round(aicc, digits = 5) == 35.53456
    @test round(EBIC, digits = 5) == 19.93785
end