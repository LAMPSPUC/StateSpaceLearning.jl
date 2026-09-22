@testset "Function: get_information" begin
    ε = [1.1, 2.2, 3.3, 4.4, 5.5]
    T = 5
    K = 3
    bic = StateSpaceLearning.get_information(T, K, ε; information_criteria="bic")
    aic = StateSpaceLearning.get_information(T, K, ε; information_criteria="aic")
    aicc = StateSpaceLearning.get_information(T, K, ε; information_criteria="aicc")
    @test round(bic; digits=5) == 10.36287
    @test round(aic; digits=5) == 11.53456
    @test round(aicc; digits=5) == 35.53456

    # ebic with the default p (= K) has no extra penalty and falls back to bic
    ebic_default = StateSpaceLearning.get_information(T, K, ε; information_criteria="ebic")
    @test ebic_default == bic

    # ebic = bic + 2γ log(binomial(p, K)); binomial(10, 3) = 120
    ebic = StateSpaceLearning.get_information(T, K, ε; information_criteria="ebic", p=10)
    @test round(ebic; digits=5) == round(bic + 2 * log(120); digits=5)

    ebic_half = StateSpaceLearning.get_information(
        T, K, ε; information_criteria="ebic", p=10, ebic_γ=0.5
    )
    @test round(ebic_half; digits=5) == round(bic + log(120); digits=5)

    # K = 0 and K = p give log(binomial) = 0, and large p must not overflow
    @test StateSpaceLearning.get_information(T, 0, ε; information_criteria="ebic", p=10) ==
        StateSpaceLearning.get_information(T, 0, ε; information_criteria="bic")
    @test StateSpaceLearning.log_binomial(10, 10) == 0.0
    @test isfinite(StateSpaceLearning.log_binomial(2000, 1000))

    @test_throws ArgumentError StateSpaceLearning.get_information(
        T, K, ε; information_criteria="abc"
    )
end
