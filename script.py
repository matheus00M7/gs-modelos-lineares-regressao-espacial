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
        101.10, 100.60, 99.90, 99.30, 98.50, 98.00, 97.20, 96.50, 96.00, 95.20,
        94.60, 94.10, 93.20, 92.80, 92.00, 91.50, 90.70, 90.10, 89.50, 88.80,
        88.20, 87.40, 86.90, 86.20, 85.50, 84.90, 84.30, 83.60, 83.00, 82.30
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