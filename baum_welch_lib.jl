module Hmm

  type BaumWelch
    A::Array{Float64,2}
    B::Array{Float64,2}
    ρ::Array{Float64,2}
    state_num::Int64
    symbol_num::Int64
    symbol::Array{Int64,2}
    pA::Array{Float64,2}
    pB::Array{Float64,2}
    pρ::Array{Float64,2}
    A_numer::Array{Float64,2}
    A_denom::Array{Float64,2}
    B_numer::Array{Float64,2}
    B_denom::Array{Float64,2}
    newρ::Array{Float64,2}
    α::Array{Float64,2}
    β::Array{Float64,2}
    c::Array{Float64,1}
  end

  function hmm_initialization(A, B, ρ)

    state_num = length(A[1,:])
    symbol_num = length(B[2,:])
    symbol = [0 1]
    # pA = A
    # pB = B
    # pρ = ρ
    pA = zeros(state_num, state_num)
    pB = zeros(state_num, symbol_num)
    pρ = zeros(1,state_num)
    A_numer = zeros(state_num, state_num)
    A_denom = zeros(state_num, state_num)
    B_numer = zeros(state_num, symbol_num)
    B_denom = zeros(state_num, symbol_num)
    newρ = zeros(1,state_num)
    α = zeros(0, state_num)
    β = zeros(0, state_num)
    c = zeros(0)

    hmm = BaumWelch(A, B, ρ, state_num, symbol_num, symbol, pA, pB, pρ,
                    A_numer, A_denom, B_numer, B_denom, newρ, α, β, c)

    return hmm
  end

  function init_variables(hmm)
    hmm.A_numer = zeros(hmm.state_num, hmm.state_num)
    hmm.A_denom = zeros(hmm.state_num, hmm.state_num)
    hmm.B_numer = zeros(hmm.state_num, hmm.symbol_num)
    hmm.B_denom = zeros(hmm.state_num, hmm.symbol_num)
    hmm.newρ = zeros(1,hmm.state_num)
  end

  function forward(hmm,obs)
    # 観測系列の長さ
    n = length(obs)
    # scaled forwardアルゴリズム
    # 変数の初期化
    α = zeros(n, hmm.state_num)
    c = zeros(n)

    # 初期化
    α[1, :] = hmm.ρ[:] .* hmm.B[:, Int(obs[1])+1]
    c[1] = 1.0 / sum(α[1, :])
    α[1, :] = c[1] * α[1, :]

    # 再帰的計算
    for t in 2:n
      α[t, :] = (α[t-1, :]' * hmm.A)' .* hmm.B[:, Int(obs[t])+1]
      c[t] = 1.0 / sum(α[t, :])
      α[t, :] = c[t] * α[t, :]
    end

    hmm.α = α
    hmm.c = c
  end

  function backward(hmm, obs)
    # 観測系列の長さ
    n = length(obs)
    # scaled backwardアルゴリズム
    # 変数の初期化
    β = zeros(n, hmm.state_num)

    # 初期化
    β[n, :] = hmm.c[n]

    # 再帰的計算
    for t in n:-1:2
      β[t-1, :] = hmm.A * (hmm.B[:, Int(obs[t]+1)] .* β[t, :])
      β[t-1, :] = hmm.c[t-1] * β[t-1, :]
    end

    hmm.β = β
  end

  function maximization_step(hmm, obs)
    # 観測系列の長さ
    n = length(obs)

    # update A
    for i in 1:hmm.state_num
      for j in 1:hmm.state_num
        A_numer = A_denom = 0.0
        for t in 1:n-1
          A_numer += hmm.α[t,i] * hmm.A[i,j] * hmm.B[j, Int(obs[t+1])+1] * hmm.β[t+1,j]
          A_denom += hmm.α[t,i] * hmm.β[t,i] / hmm.c[t]
        end
        hmm.A_numer[i, j] += A_numer
        hmm.A_denom[i, j] += A_denom
      end
    end

    # update B
    for j in 1:hmm.state_num
      for k in 1:hmm.symbol_num
        B_numer = B_denom = 0.0
        for t in 1:n
          B_numer += (obs[t] == hmm.symbol[k]) * hmm.α[t, j] * hmm.β[t, j] / hmm.c[t]
          B_denom +=  hmm.α[t, j] * hmm.β[t, j] / hmm.c[t]
        end
        hmm.B_numer[j, k] += B_numer
        hmm.B_denom[j, k] += B_denom
      end
    end

    # update ρ
    hmm.newρ += hmm.α[1, :]' .* hmm.β[1, :]' / hmm.c[1]

  end

  function check_convergence(hmm, eps)
    diff = 0.0
    diff += sum((hmm.A - hmm.pA).^2)
    diff += sum((hmm.B - hmm.pB).^2)
    diff += sum((hmm.ρ - hmm.pρ).^2)

    return sqrt(diff) < eps
  end

  function train(hmm, obs, eps = 1e-9, max_iter = 10000)

    # init
    seq_num = length(obs)
    loglik = zeros(seq_num)
    iter = 0

    for count in 1:max_iter
      if count % 10 == 0
        println("iter: [", count, "]", )
      end

      init_variables(hmm)

      # calc α, β, c each sequences
      for s in 1:seq_num
        s_obs = obs[s]
        # E-Step
        forward(hmm, s_obs)
        backward(hmm, s_obs)
        # M-Step
        maximization_step(hmm, s_obs)
        # calc log-likelihood
        loglik[s] = -sum(log.(hmm.c))
      end

      # update parameter
      hmm.A = hmm.A_numer ./ hmm.A_denom
      hmm.B = hmm.B_numer ./ hmm.B_denom
      hmm.ρ = hmm.newρ ./ seq_num

      # convergence check
      if check_convergence(hmm, eps)
        println("convergence !!")
        break
      end

      # save previous parameter
      hmm.pA = hmm.A
      hmm.pB = hmm.B
      hmm.pρ = hmm.ρ

      iter = count
    end

    print("iteration: ", iter)
    println(" log-likelihood: ", mean(loglik))

  end
end
