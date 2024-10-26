function sMAPE(y_test::Vector, prediction::Vector)
    H = length(y_test)

    denominator = abs.(y_test) + abs.(prediction)
    return (200 / H) * sum(abs(y_test[i] - prediction[i]) / (denominator[i]) for i in 1:H)
end

function MASE(y_train::Vector, y_test::Vector, prediction::Vector; m::Int64=12)
    T = length(y_train)
    H = length(y_test)

    numerator = (1 / H) * sum(abs(y_test[i] - prediction[i]) for i in 1:H)
    denominator = (1 / (T - m)) * sum(abs(y_train[j] - y_train[j - m]) for j in (m + 1):T)
    return numerator / denominator
end

function OWA(MASE1, MASE2, sMAPE1, sMAPE2)
    return 0.5 * (((MASE1) / (MASE2)) + ((sMAPE1) / (sMAPE2)))
end
