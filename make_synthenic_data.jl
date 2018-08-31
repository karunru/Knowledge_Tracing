using DataFrames
using Distributions

USER = 1000
SKILL = 5
ITEM = SKILL*5
RECORD = 200000
INCREASE = 400 / (RECORD / USER)
DECREASE = 200 / (RECORD / USER)

srand(1)

# 項目の作成
function make_items()
  item = DataFrame(item = Int[], skill = Char[], a = Float64[], β = Float64[], c = Float64[])

  for i in 1:ITEM
    p = 0.1
    a = rand(LogNormal(0, 0.5))
    β = rand(Normal(0,2))
    c = rand(Beta(20*p+1, 20*(1-p)+1))
    push!(item, [i, Char(Int('A') + (i-1) % SKILL), a, β, c])
  end

  return item
end

# 能力値の初期値
function init_trait()
  mat = rand(Normal(-10,2), USER, SKILL)
  return mat
end

# Raschモデル
function rasch(Θ, β, a, c)
  numer = 1 - c
  denom = 1 + exp(-1.7 * a * (Θ - β))
  return c + numer / denom
end

# 忘却型Raschモデル
function decay_rasch(Θ, β, Θ̃, β̃, t, h, c)
  numer = (1 + h * t)^(-exp(Θ̃ - β̃)) - c
  denom = 1 + exp(-1.7(Θ - β))
  return c + numer / denom
end

# 回答結果
function calc_result(trait, item)
  prob = rasch(trait, item[:β][1], item[:a][1], item[:c][1])
  if prob[1] > rand()
    return 1
  else
    return 0
  end
end

# 経過時間(hour)
function passed_hour(second1, second2)
  diff = second1 - second2
  return round(Int(Dates.Second(diff)) / (60*60))
end

# 能力値を成長させる
function grow_trait(trait)
  trait += rand(Beta(10,10)) * INCREASE
  return trait
end

# 能力値を衰退させる
function decline_trait(trait, time)
  trait -= rand(Beta(10,10)) * (DECREASE * time)
  return trait
end

# 初期化
items = make_items()
trait = init_trait()

# レコードを生成
data = DataFrame(result = Int[], user_id = Int[], item_id = Int[], skill = Char[], date = DateTime[])
u_num = zeros(USER)
i_num = zeros(ITEM)

# スタート時刻
date = DateTime(2017, 5, 5, 12, 0, 0)

# 最後に回答した日時を記録
user_last_update = fill(date, USER, SKILL)

for record in 1:RECORD
  user_id = rand(1:USER)
  u_num[user_id] += 1
  item_id = rand(1:ITEM)
  i_num[item_id] += 1
  skill_id = items[:skill][item_id] - 'A' + 1

  passed_time = passed_hour(date, user_last_update[user_id, skill_id])
  trait[user_id, skill_id] = decline_trait(trait[user_id, skill_id], passed_time)

  result = calc_result(trait[user_id, skill_id], items[item_id, :])

  # ランダムな秒数だけ経過させる
  date += Dates.Second(round(rand(Uniform(1,60))))
  user_last_update[user_id, skill_id] = date

  push!(data, [result, user_id, item_id, items[:skill][item_id], date])

  trait[user_id, skill_id] = grow_trait(trait[user_id, skill_id])
end

writetable("synthetic_data.csv", data)
