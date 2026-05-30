import pandas as pd
import matplotlib.pyplot as plt
import math

#base de dados
dados = {
    "tempo_segundos": [
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
        100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
        200, 210, 220, 230, 240, 250, 260, 270, 280, 290
    ],
    "pressao_kpa": [
        101.2, 100.6, 99.8, 99.1, 98.6, 97.9, 97.2, 96.5, 95.9, 95.4,
        94.6, 94.0, 93.3, 92.9, 92.1, 91.5, 90.8, 90.2, 89.5, 88.9,
        88.3, 87.6, 87.1, 86.4, 85.8, 85.2, 84.5, 83.9, 83.3, 82.6
    ]
}

df = pd.DataFrame(dados)

x = df["tempo_segundos"]
y = df["pressao_kpa"]

#medias
media_x = sum(x) / len(x)
media_y = sum(y) / len(y)

#calculo do coeficiente angular pelo metodo OLS
numerador = sum((x[i] - media_x) * (y[i] - media_y) for i in range(len(x)))
denominador = sum((x[i] - media_x) ** 2 for i in range(len(x)))

b1 = numerador / denominador
b0 = media_y - b1 * media_x

#previsoes
df["pressao_prevista"] = [b0 + b1 * valor for valor in x]

#residuos
df["residuo"] = df["pressao_kpa"] - df["pressao_prevista"]

#correlacao manual
soma_x = sum((valor - media_x) ** 2 for valor in x)
soma_y = sum((valor - media_y) ** 2 for valor in y)
correlacao = numerador / math.sqrt(soma_x * soma_y)

#erro medio absoluto
erro_medio_absoluto = sum(abs(valor) for valor in df["residuo"]) / len(df)

#resultados
print("Média de X:", round(media_x, 2))
print("Média de Y:", round(media_y, 2))
print("Coeficiente angular b1:", round(b1, 4))
print("Coeficiente linear b0:", round(b0, 4))
print("Equação da reta: y =", round(b0, 4), "+", round(b1, 4), "* x")
print("Correlação:", round(correlacao, 4))
print("Erro médio absoluto:", round(erro_medio_absoluto, 4))

print("\nTabela com previsões e resíduos:")
print(df)

#grafico de dispersão e reta de regressao
plt.scatter(df["tempo_segundos"], df["pressao_kpa"], label="Dados simulados")
plt.plot(df["tempo_segundos"], df["pressao_prevista"], label="Reta de regressão")

plt.title("Regressão Linear: Tempo x Pressão Interna do Módulo")
plt.xlabel("Tempo após a pane (segundos)")
plt.ylabel("Pressão interna (kPa)")
plt.legend()
plt.grid(True)
plt.show()