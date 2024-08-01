@testset "Function: get_information"  begin 
    ϵ = [1.1, 2.2, 3.3, 4.4, 5.5]
    T = 5
    K = 3
    bic = StateSpaceLearning.get_information(T, K, ϵ; information_criteria = "bic")
    aic = StateSpaceLearning.get_information(T, K, ϵ; information_criteria = "aic")
    aicc = StateSpaceLearning.get_information(T, K, ϵ; information_criteria = "aicc")
    @test round(bic, digits = 5) == 10.36287
    @test round(aic, digits = 5) == 11.53456
    @test round(aicc, digits = 5) == 35.53456
end