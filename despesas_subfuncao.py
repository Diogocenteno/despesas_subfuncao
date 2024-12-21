#%% Importando as bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Configuração de estilo para os gráficos
sns.set_theme(style="whitegrid")

# Carregar a planilha Excel
file_path = 'C:/Users/diogo/Desktop/leitura-escrita/data/despesas_subfuncao.xlsx'  
df = pd.read_excel(file_path)

# Verificar as primeiras linhas para garantir que os dados foram carregados corretamente
print("Primeiras linhas antes de renomear as colunas:")
print(df.head())

# Garantir que as colunas estejam corretamente nomeadas
df.columns = ['Mes_Ano', 'Area_Atuacao', 'Subfuncao', 'Valor_Empenhado', 'Valor_Liquidado', 'Valor_Pago', 'Valor_Restos_a_Pagar_Pagos']

# Verificar novamente os dados após renomear as colunas
print("\nPrimeiras linhas após renomear as colunas:")
print(df.head())

# Converter a coluna 'Mes_Ano' para um tipo de dado datetime para facilitar a análise por mês
df['Mes_Ano'] = pd.to_datetime(df['Mes_Ano'], format='%m-%Y', errors='coerce')

# Agrupar os dados por mês (ano, mês) e calcular as métricas
df_grouped = df.groupby(df['Mes_Ano'].dt.to_period('M')).agg(
    Total_Empenhado=('Valor_Empenhado', 'sum'),
    Total_Liquidado=('Valor_Liquidado', 'sum'),
    Total_Pago=('Valor_Pago', 'sum'),
    Total_Restos_a_Pagar_Pagos=('Valor_Restos_a_Pagar_Pagos', 'sum')
)

# Remover duplicatas do DataFrame agrupado
df_grouped = df_grouped.drop_duplicates()

# Calcular os percentuais, evitando divisão por zero
df_grouped['Percentual_Execucao'] = df_grouped.apply(
    lambda row: (row['Total_Liquidado'] / row['Total_Empenhado']) * 100 if row['Total_Empenhado'] > 0 else 0, axis=1
)
df_grouped['Percentual_Pagamento'] = df_grouped.apply(
    lambda row: (row['Total_Pago'] / row['Total_Liquidado']) * 100 if row['Total_Liquidado'] > 0 else 0, axis=1
)

# Salvar o índice para fins de gráficos
df_grouped.reset_index(inplace=True)
df_grouped['Mes_Ano_Str'] = df_grouped['Mes_Ano'].astype(str)

# Formatar os valores em reais e percentuais
df_grouped['Total_Empenhado'] = df_grouped['Total_Empenhado'].apply(lambda x: f'R${x:,.2f}')
df_grouped['Total_Liquidado'] = df_grouped['Total_Liquidado'].apply(lambda x: f'R${x:,.2f}')
df_grouped['Total_Pago'] = df_grouped['Total_Pago'].apply(lambda x: f'R${x:,.2f}')
df_grouped['Total_Restos_a_Pagar_Pagos'] = df_grouped['Total_Restos_a_Pagar_Pagos'].apply(lambda x: f'R${x:,.2f}')
df_grouped['Percentual_Execucao'] = df_grouped['Percentual_Execucao'].apply(lambda x: f'{x:.2f}%')
df_grouped['Percentual_Pagamento'] = df_grouped['Percentual_Pagamento'].apply(lambda x: f'{x:.2f}%')

# Exibindo o DataFrame com a análise formatada
print("\nDataFrame com os resultados agrupados e formatados:")
print(df_grouped)

# Criar gráficos
output_path = 'C:/Users/diogo/Desktop/'

# Gráfico de linhas para valores financeiros
plt.figure(figsize=(12, 8))
plt.plot(df_grouped['Mes_Ano_Str'], [float(v.strip('R$').replace(',', '')) for v in df_grouped['Total_Empenhado']], label='Empenhado', marker='o', linestyle='-', color='blue')
plt.plot(df_grouped['Mes_Ano_Str'], [float(v.strip('R$').replace(',', '')) for v in df_grouped['Total_Liquidado']], label='Liquidado', marker='o', linestyle='--', color='green')
plt.plot(df_grouped['Mes_Ano_Str'], [float(v.strip('R$').replace(',', '')) for v in df_grouped['Total_Pago']], label='Pago', marker='o', linestyle=':', color='orange')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.title('Evolução dos Valores Financeiros por Mês', fontsize=14, fontweight='bold')
plt.xlabel('Mês/Ano', fontsize=12)
plt.ylabel('Valores em R$', fontsize=12)
plt.legend(title='Categoria', fontsize=10)
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(output_path + 'grafico_valores_financeiros.png', dpi=300)
plt.show()

# Gráfico de barras para percentuais
plt.figure(figsize=(12, 8))
width = 0.4
x = range(len(df_grouped))
exec_values = [float(p.strip('%')) for p in df_grouped['Percentual_Execucao']]
pag_values = [float(p.strip('%')) for p in df_grouped['Percentual_Pagamento']]
plt.bar(x, exec_values, width=width, label='Execução (%)', color='blue', alpha=0.7)
plt.bar([p + width for p in x], pag_values, width=width, label='Pagamento (%)', color='orange', alpha=0.7)
plt.xticks([p + width/2 for p in x], df_grouped['Mes_Ano_Str'], rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.title('Percentuais de Execução e Pagamento por Mês', fontsize=14, fontweight='bold')
plt.xlabel('Mês/Ano', fontsize=12)
plt.ylabel('Percentual (%)', fontsize=12)
plt.legend(title='Categoria', fontsize=10)
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(output_path + 'grafico_percentuais.png', dpi=300)
plt.show()

# Gráfico de pizza para distribuição de valores empenhados
plt.figure(figsize=(10, 10))
values = [float(v.strip('R$').replace(',', '')) for v in df_grouped['Total_Empenhado']]
labels = df_grouped['Mes_Ano_Str']
colors = sns.color_palette('pastel')[0:len(labels)]
wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 10})
plt.title('Distribuição de Valores Empenhados por Mês', fontsize=14, fontweight='bold')
plt.legend(wedges, labels, title="Mês/Ano", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
plt.tight_layout()
plt.savefig(output_path + 'grafico_pizza_empenhado.png', dpi=300)
plt.show()

# Converter os valores financeiros para números (sem o prefixo R$) para análise de previsão
df_grouped['Total_Empenhado_Num'] = df_grouped['Total_Empenhado'].apply(lambda x: float(x.strip('R$').replace(',', '')))
df_grouped['Total_Liquidado_Num'] = df_grouped['Total_Liquidado'].apply(lambda x: float(x.strip('R$').replace(',', '')))
df_grouped['Total_Pago_Num'] = df_grouped['Total_Pago'].apply(lambda x: float(x.strip('R$').replace(',', '')))
df_grouped['Total_Restos_a_Pagar_Pagos_Num'] = df_grouped['Total_Restos_a_Pagar_Pagos'].apply(lambda x: float(x.strip('R$').replace(',', '')))

# Definir as variáveis independentes (X) e dependentes (y)
X = df_grouped[['Total_Empenhado_Num', 'Total_Liquidado_Num', 'Total_Restos_a_Pagar_Pagos_Num']]  # Incluindo a variável de Restos a Pagar

y = df_grouped['Total_Pago_Num']

# Dividir o conjunto de dados em 80% para treinamento e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar o tamanho dos conjuntos de treino e teste
print(f"\nTamanho do conjunto de treinamento: {X_train.shape[0]}")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")

# Inicializar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Exibir os resultados de avaliação
print(f"\nErro quadrático médio (MSE): {mse:.2f}")
print(f"Coeficiente de determinação (R²): {r2:.2f}")

# Exibir os coeficientes da regressão linear
print("\nCoeficientes da regressão linear:")
print(f"Coeficiente de Total_Empenhado: {model.coef_[0]:.2f}")
print(f"Coeficiente de Total_Liquidado: {model.coef_[1]:.2f}")
print(f"Coeficiente de Restos a Pagar Pagos: {model.coef_[2]:.2f}")
print(f"Intercepto: {model.intercept_:.2f}")

# Exibir algumas previsões comparadas com os valores reais
results = pd.DataFrame({'Mes_Ano': df_grouped.loc[X_test.index, 'Mes_Ano_Str'], 
                        'Real': y_test, 
                        'Previsto': y_pred})

# Remover duplicatas dos resultados de previsões
results = results.drop_duplicates()

# Formatar os resultados como valores monetários
results['Real'] = results['Real'].apply(lambda x: f'R${x:,.2f}')
results['Previsto'] = results['Previsto'].apply(lambda x: f'R${x:,.2f}')

print("\nComparação de valores reais vs previstos:")
print(results.head())

# Remover as colunas especificadas antes de salvar
df_grouped = df_grouped.drop(columns=['Mes_Ano_Str', 'Total_Empenhado_Num', 'Total_Liquidado_Num', 'Total_Pago_Num', 'Total_Restos_a_Pagar_Pagos_Num'])

# Salvar os dados em uma nova planilha Excel
output_file_path = 'C:/Users/diogo/Desktop/resultado_previsao_sem_colunas.xlsx'
with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
    df_grouped.to_excel(writer, sheet_name='Dados_Agrupados', index=False)
    
    # Adicionar os resultados da comparação de previsões
    results.to_excel(writer, sheet_name='Comparacao_Real_vs_Previsto', index=False)

    # Acessar o objeto do ExcelWriter
    workbook  = writer.book
    worksheet = writer.sheets['Dados_Agrupados']
    
    # Definir o formato para as colunas de valores (monetários)
    money_format = workbook.add_format({'num_format': 'R$#,##0.00'})
    
    # Aplicar o formato para as colunas que ainda contêm valores monetários
    worksheet.set_column('B:E', 20, money_format)
    
    print(f"Arquivo Excel com as colunas removidas e resultados adicionados salvo em: {output_file_path}")
