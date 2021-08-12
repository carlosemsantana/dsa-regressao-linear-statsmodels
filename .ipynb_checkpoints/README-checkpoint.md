# Regressão Linear Simples


## Definindo o Problema de Negócio

Nosso objetivo é construir um modelo de Machine Learning que seja capaz de fazer previsões sobre a taxa média de ocupação de casas na região de Boston, EUA, por proprietários. A variável a ser prevista é um valor numérico que representa a mediana da taxa de ocupação das casas em Boston. Para cada casa temos diversas variáveis explanatórias. Sendo assim, podemos resolver este problema empregando Regressão Linear Simples ou Múltipla.


## Definindo o Dataset 

Usaremos o Boston Housing Dataset, que é um conjunto de dados que tem a taxa média de ocupação das casas, juntamente com outras 13 variáveis que podem estar relacionadas aos preços das casas. Esses são os fatores como condições socioeconômicas, condições ambientais, instalações educacionais e alguns outros fatores semelhantes. Existem 506 observações nos dados para 14 variáveis. Existem 12 variáveis numéricas em nosso conjunto de dados e 1 variável categórica. O objetivo deste projeto é construir um modelo de regressão linear para estimar a taxa média de ocupação das casas pelos proprietários em Boston.


Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/housing/


## Descrição do dataset

1. CRIM: per capita crime rate by town 
2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft. 
3. INDUS: proportion of non-retail business acres per town 
4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
5. NOX: nitric oxides concentration (parts per 10 million) 
6. RM: average number of rooms per dwelling 
7. AGE: proportion of owner-occupied units built prior to 1940 
8. DIS: weighted distances to five Boston employment centres 
9. RAD: index of accessibility to radial highways 
10. TAX: full-value property-tax rate per 10,000 
11. PTRATIO: pupil-teacher ratio by town 
12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
13. LSTAT: % lower status of the population 
14. TARGET: Median value of owner-occupied homes in $1000's

```python
# Carregando o Dataset Boston Houses
from sklearn.datasets import load_boston
boston = load_boston() 

# Carregando Bibliotecas Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline 
```

## Análise Exploratória

```python
# Convertendo o dataset em um dataframe com Pandas
dataset = pd.DataFrame(boston.data, columns = boston.feature_names)
dataset['target'] = boston.target
```

```python
dataset.head(5)
```

Não somos obrigados usar todas as variáveis deste conjunto de dados, vamos procurar quais são as variáveis mais relevantes.

```python
# Calculando a média da variável de resposta
valor_medio_esperado_na_previsao = dataset['target'].mean()
```

```python
valor_medio_esperado_na_previsao
```

Provavelmente 22.5 representa o valor médio da taxa de ocupação, e é bem provável que as previsões estejam em torno deste valor.

```python
# Calculando (simulando) o SSE
# O SSE é a diferença ao quadrado entre o valor previsto e o valor observado.
# Considerando que o valor previsto seja igual a média, podemos considerar que 
# y = média da variável target (valores observados).

# Estamos apenas simulando o SSE, uma vez que a regressão ainda não foi criada e os valores previstos 
# ainda não foram calculados.

squared_errors = pd.Series(valor_medio_esperado_na_previsao - dataset['target'])**2 
SSE = np.sum(squared_errors)
print ('Soma dos Quadrados dos Erros (SSE): %01.f' % SSE)
```

```python
# Histograma dos erros
# Temos mais erros "pequenos", ou seja, mais valores próximos à média.
hist_plot = squared_errors.plot(kind='hist')
```

Os dados provavelmente estão seguindo uma distribuição normal. Saber como os dados estão distribuídos nos ajuda a escolher quais técnicas podemos usar para solução do problema.


Para Regressão Linear Simples usaremos como variável explanatória a variável RM que representa o número médio de quartos nas casas.

```python
# Função para calcular o desvio padrão (a partir da fórmula)
def calc_desvio_padrao(variable, bias = 0):
    observations = float(len(variable))
    return np.sqrt(np.sum((variable - np.mean(variable))**2) / (observations - min(bias, 1)))
```

```python
# Imprimindo o desvio padrão via fórmula e via NumPy da variável RM 
print ('Resultado da Função: %0.5f Resultado do Numpy: %0.5f' % (calc_desvio_padrao(dataset['RM']), \
                                                                 np.std(dataset['RM'])))
```

```python
# Funções para calcular a variância da variável RM e a correlação com a variável target
def covariance(variable_1, variable_2, bias = 0):
    observations = float(len(variable_1))
    return np.sum((variable_1 - np.mean(variable_1)) * (variable_2 - np.mean(variable_2))) / (observations - min(bias,1))

def standardize(variable):
    return (variable - np.mean(variable)) / np.std(variable)

def correlation(var1, var2, bias = 0):
    return covariance(standardize(var1), standardize(var2), bias)
```

```python
# Compara o resultado das nossas funções com a função pearsonr do SciPy
from scipy.stats.stats import pearsonr
print ('Nossa estimativa de Correlação: %0.5f' % (correlation(dataset['RM'], dataset['target'])))
print ('Correlação a partir da função pearsonr do SciPy: %0.5f' % pearsonr(dataset['RM'], dataset['target'])[0])
```

### Analisando o resultado

Próximo de [-1] indica alta correlação negativa, próximo de [1] alta correlção positiva e próximo de [0] que não existe correlação. Olhando o resultado concluímos que quanto mais quartos tem uma casa, maior a taxa de ocupação. 

```python
# Definindo o range dos valores de x e y
x_range = [dataset['RM'].min(),dataset['RM'].max()]
y_range = [dataset['target'].min(),dataset['target'].max()]
```

```python
# Plot dos valores de x e y com a média
scatter_plot = dataset.plot(kind = 'scatter', x = 'RM', y = 'target', xlim = x_range, ylim = y_range)

# Cálculo da média
meanY = scatter_plot.plot(x_range, [dataset['target'].mean(),dataset['target'].mean()], '--', color = 'red', linewidth = 1)
meanX = scatter_plot.plot([dataset['RM'].mean(), dataset['RM'].mean()], y_range, '--', color = 'red', linewidth = 1)
```

Existe correlação positiva entre as variáveis RM e target, por isso, iremos contruir o modelos com base nestas informações.


## Regressão Linear com o StatsModels


https://www.statsmodels.org/stable/index.html

```python
# Importando as funções
import statsmodels.api as sm
```

```python
# Gerando X e Y. Vamos adicionar a constante ao valor de X, gerando uma matrix.
y = dataset['target']
X = dataset['RM']
```

```python
# Esse comando adiciona os valores dos coefientes à variável X (o bias será calculado internamente pela função)
X = sm.add_constant(X)
X.head()
```

```python
# Criando o modelo de regressão
modelo = sm.OLS(y, X)

# Treinando o modelo
modelo_v1 = modelo.fit()
```

```python
# Resumo do modelo criado
print(modelo_v1.summary())
```

```python
print(modelo_v1.params)
```

```python
# Gerando os valores previstos
valores_previstos = modelo_v1.predict(X)
valores_previstos
```

```python
# Fazendo previsões com o modelo treinado
RM = 6
Xp = np.array([1, RM])
print ("Se RM = %01.f nosso modelo prevê que a mediana da taxa de ocupação é %0.1f" % (RM, modelo_v1.predict(Xp)))
```

### Gerando um ScatterPlot com a Linha de Regressão

```python
# Range de valores para x e y
x_range = [dataset['RM'].min(), dataset['RM'].max()]
y_range = [dataset['target'].min(), dataset['target'].max()]
```

```python
# Primeira camada do Scatter Plot
scatter_plot = dataset.plot(kind = 'scatter', x = 'RM', y = 'target', xlim = x_range, ylim = y_range)

# Segunda camada do Scatter Plot (médias)
meanY = scatter_plot.plot(x_range, [dataset['target'].mean(),dataset['target'].mean()], '--', color = 'red', linewidth = 1)
meanX = scatter_plot.plot([dataset['RM'].mean(),dataset['RM'].mean()], y_range, '--', color = 'red', linewidth = 1)

# Terceira camada do Scatter Plot (linha de regressão)
regression_line = scatter_plot.plot(dataset['RM'], valores_previstos, '-', color = 'orange', linewidth = 2)
```

```python
# Gerando os resíduos
residuos = dataset['target'] - valores_previstos
residuos_normalizados = standardize(residuos)
```

```python
# ScatterPlot dos resíduos (O resíduos são os erros)
residual_scatter_plot = plt.plot(dataset['RM'], residuos_normalizados,'bp')
plt.xlabel('RM') 
plt.ylabel('Resíduos Normalizados') 
mean_residual = plt.plot([int(x_range[0]),round(x_range[1],0)], [0,0], '-', color = 'red', linewidth = 3)
upper_bound = plt.plot([int(x_range[0]),round(x_range[1],0)], [3,3], '--', color = 'red', linewidth = 2)
lower_bound = plt.plot([int(x_range[0]),round(x_range[1],0)], [-3,-3], '--', color = 'red', linewidth = 2)
plt.grid()
```

## Regressão Linear com Scikit-Learn


https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

```python
from sklearn import linear_model
```

```python
# Cria o objeto
modelo_v2 = linear_model.LinearRegression(normalize = False, fit_intercept = True)
```

```python
# Define os valores de x e y
num_observ = len(dataset)
X = dataset['RM'].values.reshape((num_observ, 1)) # X deve sempre ser uma matriz e nunca um vetor
y = dataset['target'].values # y pode ser um vetor
```

```python
# Treinamento do modelo - fit()
modelo_v2.fit(X,y)
```

```python
# Imprime os coeficientes
print (modelo_v2.coef_)
print (modelo_v2.intercept_)
```

```python
# Fazendo previsões com o modelo treinado
RM = 6
# Xp = np.array(RM)
Xp = np.array(RM).reshape(-1, 1)
print ("Se RM = %01.f nosso modelo prevê que a mediana da taxa de ocupação é %0.1f" % (RM, modelo_v2.predict(Xp)))
```

### Comparação StatsModels x ScikitLearn

```python
from sklearn.datasets import make_regression
HX, Hy = make_regression(n_samples = 10000000, n_features = 1, n_targets = 1, random_state = 101)
```

```python
%%time
sk_linear_regression = linear_model.LinearRegression(normalize=False, fit_intercept=True)
sk_linear_regression.fit(HX,Hy)
```

```python
%%time
sm_linear_regression = sm.OLS(Hy, sm.add_constant(HX))
sm_linear_regression.fit()
```

## Resultado


Obtivemos o mesmo resultado, porque a implementação conceitual é a mesma nos dois pacotes, enentretanto em termos de performance o ScikitLearn apresenta um desempenho melhor que StatsModels. Se você precisa de interpretação do modelo StatsModels pode ser uma boa opção, se precisa de velocidade na hora de treinar e preparar o seu modelo,  ScikitLearn será nesse caso a melhor escolha.  


## Cost Function de um Modelo de Regressão Linear


O objetivo da regressão linear é buscar a equação de uma linha de regressão que minimize a soma dos erros ao quadrado, da diferença entre o valor observado de y e o valor previsto.

Existem alguns métodos para minimização da Cost Function tais como: Pseudo-inversão, Fatorização e Gradient Descent.


## Por Que Usamos o Erro ao Quadrado?

```python
# Definindo 2 conjuntos de dados
import numpy as np
x = np.array([9.5, 8.5, 8.0, 7.0, 6.0])
```

```python
# Função para cálculo da Cost Function
def squared_cost(v, e):
    return np.sum((v - e) ** 2)
```

```python
# A função fmin() tenta descobrir o valor do somatório mínimo dos quadrados
from scipy.optimize import fmin
xopt = fmin(squared_cost, x0 = 0, xtol = 1e-8, args = (x,))
```

```python
print ('Resultado da Otimização: %0.1f' % (xopt[0]))
print ('Média: %0.1f' % (np.mean(x)))
print ('Mediana: %0.1f' % (np.median(x)))
```

```python
def absolute_cost(v,e):
    return np.sum(np.abs(v - e))
```

```python
xopt = fmin(absolute_cost, x0 = 0, xtol = 1e-8, args = (x,))
```

```python
print ('Resultado da Otimização: %0.1f' % (xopt[0]))
print ('Média %0.1f' % (np.mean(x)))
print ('Mediana %0.1f' % (np.median(x)))
```

### Resposta


Quando usamos o cálculo da diferença ao quadrado a otimização se aproxima da média. Quando usamos o cálculo da difereça apenas com valores absoluto, a otimização fica próxima da mediana. Devemos escolher a média com base em uma fundamentação matemática estatística. Por esse motivo quando na amostra temos valores outliers precisamos aplicar técnicas para remover esses valores.


----


[Carlos Eugênio](https://carlosemsantana.github.io/)

Graduando Engenharia Mecatrônica


### Referências
- Data Science Academy - <a href="https://www.datascienceacademy.com.br">https://www.datascienceacademy.com.br</a>
