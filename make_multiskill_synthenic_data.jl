# "A", "B", ..., "E", "A,B", ..., "A,B,C,D,E"

using DataFrames
using Distributions

USER = 1000
SKILL = 5
ITEM = 2^SKILL - 1
RECORD = 20000
INCREASE = 100 / (RECORD / USER)
# DECREASE = 200 / (RECORD / USER)
DECREASE = 0

srand(0)

# 項目の作成
function make_items()
  item = DataFrame(item_id = Int[], skill = Char[], a = Float64[], β = Float64[], c = Float64[])

  for i in 1:ITEM
    for j in SKILL:-1:1
      if bin(i, SKILL)[j] == '1'
        p = 0.1
        a = rand(LogNormal(0, 0.5))
        β = rand(Normal(0,2))
        c = rand(Beta(20*p+1, 20*(1-p)+1))
        push!(item, [i, Char(Int('A') + (SKILL-j)), a, β, c])
      end
    end
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
function calc_result(skills,traits, items)
  prob = 1
  for skill in skills
    Θ = traits[skill]
    β = items[items[:skill] .== Char(skill + 'A' - 1), :β][1]
    a = items[items[:skill] .== Char(skill + 'A' - 1), :a][1]
    c = items[items[:skill] .== Char(skill + 'A' - 1), :c][1]
    prob *= rasch(Θ, β, a, c)
  end

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
data = DataFrame(result = Int[], user_id = Int[], item_id = Int[], skill = String[], date = DateTime[])
u_num = zeros(USER)
i_num = zeros(ITEM)

# スタート時刻
date = DateTime(2016, 4, 1, 12, 0, 0)

# 最後に回答した日時を記録
user_last_update = fill(date, USER, ITEM)

for record in 1:RECORD
  user_id = rand(1:USER)
  u_num[user_id] += 1
  item_id = rand(1:ITEM)
  i_num[item_id] += 1
  skills = [skill - 'A' + 1 for skill in items[items[:item_id] .== item_id, :skill]]

  passed_time = passed_hour(date, user_last_update[user_id, item_id])
  for skill_id in skills
    trait[user_id, skill_id] = decline_trait(trait[user_id, skill_id], passed_time)
  end

  result = calc_result(skills, trait[user_id, :], items[items[:item_id] .== item_id, :])
  print(result)

  # ランダムな秒数だけ経過させる
  date += Dates.Second(round(rand(Uniform(1,60))))
  user_last_update[user_id, item_id] = date

  push!(data, [result, user_id, item_id,  join([Char(i) for i in (skills + Int('A') - 1)], ","), date])

  for skill_id in skills
    trait[user_id, skill_id] = grow_trait(trait[user_id, skill_id])
  end
end

writetable("synthetic_multiskill_data.csv", data)
