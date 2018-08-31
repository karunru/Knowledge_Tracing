# -*- coding: utf-8 -*-

#
# Knowledge Tracing
# ベースとなるKnowledge Tracingのプログラムです．
# データ(カラムは['result', 'user_id', 'item_id', 'skill', 'date'])を読み込み，
# KTのパラメータを学習，スキル習得確率と設問に対する応答結果を予測します．
# 読み込む前に日付順にソートする必要があります．これも別スクリプトで可能です．
# AUCや正答率は別ファイルのevalution.pyで求めてください．
#

using DataFrames
include("./baum_welch_lib.jl")
using Hmm

## パラメータ学習の初期値
p_init = 0.1
ρ = [p_init 1-p_init]
p_learn = 0.1
p_forget = 0
A = [1-p_learn p_learn;  p_forget 1-p_forget]
p_slip = 0.1
p_guess = 0.1
B = [1-p_guess p_guess; p_slip 1-p_slip]



# データファイルの読み込み
function read_data(path)

  # データの読み込み
  df = readtable(path)

  # このカラムは必要ない
  delete!(df, :date)
  delete!(df, :item_id)

  # 正解:1、不正解:0　→ 正解:0、不正解:1　にする
  df2 = df[:result]
  df2[df2[:] .== 0] = 2
  df2[df2[:] .== 1] = 0
  df2[df2[:] .== 2] = 1
  df[:result] = df2

  # ユーザー一覧
  # 先行研究だとmost_commonでやってるから回答数の多いユーザーから順になってる
  # リストになってるけど俺のはなってない
  user_list = unique(df[:user_id])

  # スキル一覧
  skill_list = sort(unique(df[:skill]))

  return df, user_list, skill_list
end

# データの分割
function parse_data(df, user, skill)

  println("Parsing data ...")
  # 同じスキルを使う問題を抽出
  s_data = df[df[:skill] .== skill, :]

  # ユーザー数
  n = length(user)

  # 結果格納用
  test_data_df = df[1,:]
  deleterows!(test_data_df, 1)
  train_data_df = df[1,:]
  deleterows!(train_data_df, 1)
  train_data_p_u = []

  for i in 1:n
    # 同じユーザーが回答したものを抽出
    u_data = s_data[s_data[:user_id] .== user[i],:]
    # それをカウント
    u_data_count = length(u_data[:user_id])

    # テストデータを作る場合
    # 回答数が2回以下なら使えないデータ
    if (u_data_count <= 2)
      continue
    end

    # トレーニングデータ用の回答結果をリストに格納
    push!(train_data_p_u, u_data[1:end-1, :result])
    # トレーニングデータを抽出
    append!(train_data_df, u_data[1:end-1, :])
    # テストデータを抽出
    append!(test_data_df, u_data[end, :])
  end

  # テストデータが作れなかった場合は，仕方ないのでスキップする
  if length(test_data_df) == 0
    print("テストデータを作るためのデータが足りません...")
  end

  return train_data_p_u, test_data_df, train_data_df
end

# 成績予測
function predicter(train_data, test_data, hmm_ρ, hmm_A, hmm_B)

  # パラメータをセット
  pinit = hmm_ρ[1]
  plearn = hmm_A[1, 2]
  pguess = hmm_B[1, 2]
  pslip = hmm_B[2, 1]

  # 予測するユーザー数をカウント(テストデータの行数と一致)
  s_num = length(test_data[:result])

  # 正解:0、不正解:1　→ 正解:1、不正解:0　に戻す
  # トレーニングデータについて
  df2 = train_data[:result]
  df2[df2[:] .== 0] = 2
  df2[df2[:] .== 1] = 0
  df2[df2[:] .== 2] = 1
  train_data[:result] = df2
  # # テストデータについて
  df2 = test_data[:result]
  df2[df2[:] .== 0] = 2
  df2[df2[:] .== 1] = 0
  df2[df2[:] .== 2] = 1
  test_data[:result] = df2


  # 予測結果格納用
  # カラム名をつけておく
  p_te_all = DataFrame(result=[], user_id=[], skill=[], predict=[], probability=[])
  p_tr_all =  DataFrame(result=[], user_id=[], skill=[], predict=[], probability=[])

  # 各ユーザーについて求める
  for s in 1:s_num
    # ユーザー名
    user_name = test_data[s, :user_id]

    # そのユーザーの回答結果(トレーニングデータの中身)
    user_data = train_data[train_data[:user_id] .== user_name, :]
    cases = user_data[:result]
    # 回答回数
    n = length(cases)

    # スキル習得確率を計算
    pL = [pinit]
    for i in 1:n
      if cases[i] == 1.0
        num = (pL[i] * (1 - pslip)) / ((pL[i] * (1-pslip)) + (1-pL[i]) * pguess)
      else
        num = (pL[i] * pslip) / ((pL[i] * pslip) + (1-pL[i]) * (1-pguess))
      end
      numpL = num + (1-num) * plearn
      pL = vcat(pL, numpL)
    end

    # 設問に正解する確率を計算
    pC = []  # 確率
    CoI = [] # Correct or Incorrect
    for i in 1:n+1
      numC = (pL[i] * (1-pslip)) + ((1-pL[i]) * pguess)
      append!(pC, numC)
      append!(CoI, ((pC[i] > 0.5)?1:0))
    end

    # 各ユーザーに対する予測結果(トレーニングデータ)
    p_tr = hcat(user_data, CoI[1:end-1], pC[1:end-1])
    rename!(p_tr, :x1, :predict)
    rename!(p_tr, :x1_1, :probability)
    # 各ユーザーに対する予測結果(テストデータ)
    p_te = hcat(test_data[s, :], CoI[end], pC[end])
    rename!(p_te, :x1, :predict)
    rename!(p_te, :x1_1, :probability)

    # すべてのユーザーの予測結果に追加(トレーニングデータ)
    p_tr_all = vcat(p_tr_all, p_tr)
    # すべてのユーザーの予測結果に追加(テストデータ)
    p_te_all = vcat(p_te_all, p_te)
  end

  return p_tr_all, p_te_all
end


### MAIN ###
function main()

  # コマンドライン引数を読み取る
  # argc = length(ARGS)
  #
  # # 引数の個数チェック
  # if (argc-1 < 1)
  #   print("Usage: \$ julia", ARGS[1], "datafile.csv")
  #   exit()
  # end

  # データファイルの読み込み
  filepath = ARGS[1]
  df, user_list, skill_list = read_data(filepath)

  # HMMパラメータの読み込み
  HMM_model_flag = false
  # if argc-1 == 2
  #   HMM_model_flag = True
  #   HMM_model = pd.read_csv(argvs[2])
  # end

  # スキル数
  s_num = length(skill_list)

  # モデルのパラメータを格納する変数
  model = DataFrame(skill = [], init = [], learn = [], guess = [], slip = [])
  global predict_test = DataFrame([])
  global predict_train = DataFrame([])
  for i in 1:s_num
    println("+--------------------------------------------+**...")
    println("|  TRAINING SKILL ", skill_list[i], "")
    println("+--------------------------------------------+**...")

    # データの分割
    global train = 0
    train, test_df, train_df = parse_data(df, user_list, skill_list[i])
    if length(test_df) == 0
      println("skipping...")
      continue
    end

    # HMMのパラメータ学習
    if HMM_model_flag == true
      println("Loading HMM Parameters from Model File...")
      println("ここはまだやってない")
      # hmm = BaumWelch(A, B, pi)
      # params = HMM_model[HMM_model['skill'] == skill_list[i]]
      # # init
      # hmm.pi[0] = float(params['init'].values)
      # # learn
      # hmm.A[1, 0] = float(params['learn'].values)
      # # slip
      # hmm.B[0, 1] = float(params['slip'].values)
      # # guess
      # hmm.B[1, 0] = float(params['guess'].values)
    else
      # HMMのパラメータ学習
      println("Training HMM Parameters ...")
      hmm = Hmm.hmm_initialization(A, B, ρ)
      Hmm.train(hmm,train,1e-6,1000)
    end

    println("***  Learned Params  ***")
    println("init:", hmm.ρ[1], " learn:", hmm.A[1, 2], " guess:", hmm.B[1, 2], " slip:", hmm.B[2, 1])

    # 予測
    println("Predicting Future Student Performance ...")
    tr_res, te_res = predicter(train_df, test_df, hmm.ρ, hmm.A, hmm.B)

    # 結果を保持
    predict_test = vcat([predict_test, te_res])
    predict_train = vcat([predict_train, tr_res])

    # パラメータの記録
    tmp =  [skill_list[i], hmm.ρ[1], hmm.A[1, 2], hmm.B[1, 2], hmm.B[2, 1]]
    push!(model, tmp)
  end

  # # パラメータをファイルに保存
  # time = datetime.now().strftime("%Y%m%d-%H%M%S")
  # if  HMM_model_flag == False:
  #     model.to_csv("model_" + time + ".csv", index = False)
	writetable("model_params.csv", model)
  #
  # # 予測結果の出力
  # predict_test.to_csv("predict_test_" + time + ".csv", index = False)
  # predict_train.to_csv("predict_train_" + time + ".csv", index = False)
  println(predict_train)
  println(predict_test)
  println(model)
end

main()
