<p align="center"><img src="assets/banner.gif" alt="stochgame" width="720"></p>
<p align="center"><img src="assets/divider.gif" width="720"></p>

# stochgame

Движок стохастических игр и симуляция теории игр для автономных ИИ-агентов.
**Один файл, одна зависимость — `numpy`.**

```bash
pip install numpy
python stochgame.py selftest   # 8 проверок
python stochgame.py demo       # экскурсия по всему (~20 с)
```

## Что внутри

| Блок | Что есть |
|---|---|
| Решатели | точный LP для антагонистических игр (симплекс), все Нэш-равновесия биматричной игры (перебор носителей), `nash_gap` |
| `StochasticGame` | N игроков, S состояний, `P[s,a0..aN,s']`, `R[s,a0..aN,i]`, терминальные состояния |
| Планирование | итерация Шепли (`shapley`) для 2-игроковых zero-sum игр |
| Агенты | `RandomAgent`, `Always`, `QLearner`, `MinimaxQ`, `RegretMatcher`, `TitForTat`, `GrimTrigger`, `Pavlov` |
| Симуляция | `play`, `tournament` (round-robin), `replicator` (RK4) |
| Игры | дилемма заключённого, охота на оленя, chicken, камень-ножницы-бумага, орёл-решка, повторяющиеся игры с памятью, `commons` (трагедия общин), `random_game` |

## Пример

```python
import stochgame as sg

g = sg.commons()                                   # стохастическая игра с общим ресурсом
agents = [sg.QLearner(), sg.QLearner()]
res = sg.play(g, agents, episodes=300, horizon=50)
print(res.returns[-30:].mean(0), res.freq[-30:, :, 0].mean())   # доход, доля "бережного" действия

names, M = sg.tournament(sg.prisoners_dilemma(),
                         {"TFT": sg.TitForTat, "ALLD": lambda: sg.Always(1), "Pavlov": sg.Pavlov})
print(sg.format_matrix(names, M))

V, policies = sg.shapley(sg.random_game(4, 3, 3, seed=7))        # точное решение
```

Свой агент — наследуйте `Agent`, реализуйте `act(s)` и `observe(s, acts, r, s2, done)`.
Свою игру — передайте тензоры `P` и `R` в `StochasticGame`.

## Баннеры

`python tools/make_banner.py` пересоздаёт `assets/banner.gif`, `banner.png`, `divider.gif`.
Тоже только numpy + stdlib: свой растровый шрифт, свои GIF (LZW) и PNG энкодеры.
Кометы на баннере — реальные орбиты replicator dynamics для игры «камень-ножницы-бумага», посчитанные самим движком.

## Лицензия

MIT
