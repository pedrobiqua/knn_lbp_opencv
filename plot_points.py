import pandas as pd
import matplotlib.pyplot as plt
import sys

# Verificar se os parâmetros foram passados corretamente
if len(sys.argv) != 3:
    print("Uso incorreto. Exemplo: python script.py points.csv output_image.png")
    sys.exit(1)

# Obter os argumentos
csv_file = sys.argv[1]
image_name = sys.argv[2]

# Ler o arquivo CSV
data = pd.read_csv(csv_file)

# Criar o gráfico de dispersão
for label in data['label'].unique():
    subset = data[data['label'] == label]
    plt.scatter(subset['x'], subset['y'], label=f"Class {label}")

# Adicionar rótulos e título
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Scatter Plot with Labels")
plt.legend()

# Mostrar o gráfico
plt.savefig(image_name)
print(f"Gráfico salvo como {image_name}")
